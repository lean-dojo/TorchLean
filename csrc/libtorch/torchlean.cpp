// Lean–LibTorch interface: ownership, operations, and explicit backward calls.
#include "torchlean_libtorch.h"
#include "unstable.h"

// Buffer ownership and runtime controls

#include <lean/mimalloc.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <string>

namespace {

using namespace torchlean;

struct Counter {
  std::atomic<uint64_t> live{0}, peak{0}, created{0}, retired{0};

  void add(uint64_t amount) {
    created.fetch_add(1, std::memory_order_relaxed);
    const uint64_t next = live.fetch_add(amount, std::memory_order_relaxed) + amount;
    uint64_t previous = peak.load(std::memory_order_relaxed);
    while (previous < next &&
           !peak.compare_exchange_weak(previous, next, std::memory_order_relaxed)) {}
  }

  void remove(uint64_t amount) {
    retired.fetch_add(1, std::memory_order_relaxed);
    live.fetch_sub(amount, std::memory_order_relaxed);
  }
};

// Logical ownership counters deliberately exclude ATen temporaries and shared-storage aliases.
// The separate upstream allocator counters include storage allocated inside ATen kernels.
Counter payloads;
Counter wrappers;
std::once_flag initialization;
std::atomic<int> selected_device{0};

// The bytes of `x`'s elements.
size_t nbytes(const Tensor& x) {
  return static_cast<size_t>(x.numel()) * static_cast<size_t>(x.element_size());
}

bool release_data(torchlean_cuda_buffer* buffer) {
  if (!buffer || !buffer->tensor) return false;
  const size_t size = buffer->size;
  const size_t bytes = nbytes(*buffer->tensor);
  buffer->tensor.reset();
  buffer->size = 0;
  if (size != 0) payloads.remove(bytes);
  return size != 0;
}

void finalize(void* pointer) {
  auto* buffer = static_cast<torchlean_cuda_buffer*>(pointer);
  if (!buffer) return;
  release_data(buffer);
  if (buffer->format) lean_dec(buffer->format);
  delete buffer;
  wrappers.remove(1);
}

void foreach_reference(void* pointer, b_lean_obj_arg visitor) {
  auto* buffer = static_cast<torchlean_cuda_buffer*>(pointer);
  if (buffer && buffer->format) {
    lean_inc(visitor);
    lean_inc(buffer->format);
    lean_dec(lean_apply_1(visitor, buffer->format));
  }
}

lean_external_class* buffer_class() {
  static lean_external_class* result =
      lean_register_external_class(finalize, foreach_reference);
  return result;
}

int64_t signed_bits(uint64_t value) {
  int64_t result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

Tensor shift_right(const Tensor& value, int64_t amount) {
  // ATen int64 shifts are arithmetic; masking recovers the unsigned SplitMix shift.
  const auto mask = static_cast<int64_t>(UINT64_MAX >> amount);
  return aten<"bitwise_and.Tensor">(
      aten<"bitwise_right_shift.Tensor">(value, scalar_for(value, amount)),
      scalar_for(value, mask));
}

Tensor splitmix_draws(uint32_t n, uint64_t key) {
  const auto counter = arange(static_cast<int64_t>(n), torchlean::options().dtype(kLong));
  auto value = add(counter, scalar_for(counter, signed_bits(key + UINT64_C(0x9e3779b97f4a7c15))));
  value = aten<"mul.Tensor">(aten<"bitwise_xor.Tensor">(value, shift_right(value, 30)),
                             scalar_for(value, signed_bits(UINT64_C(0xbf58476d1ce4e5b9))));
  value = aten<"mul.Tensor">(aten<"bitwise_xor.Tensor">(value, shift_right(value, 27)),
                             scalar_for(value, signed_bits(UINT64_C(0x94d049bb133111eb))));
  return aten<"bitwise_and.Tensor">(aten<"bitwise_xor.Tensor">(value, shift_right(value, 31)),
                                    scalar_for(value, INT64_C(0xffffffff)));
}

ScalarType scalar_type(uint8_t dtype) {
  TORCHLEAN_CHECK(dtype <= 1, "LibTorch: unsupported buffer dtype");
  return dtype == 0 ? kFloat : kDouble;
}

Tensor uniform(uint32_t n, uint64_t key, ScalarType dtype) {
  const auto draws = aten<"to.dtype">(splitmix_draws(n, key), kDouble);
  return aten<"to.dtype">(aten<"div.Tensor">(draws, scalar_for(draws, 4294967296.0)), dtype);
}

Tensor bernoulli(uint32_t n, double probability, uint64_t key, ScalarType dtype) {
  const double p = rounded(probability, dtype);
  if (!(p > 0.0)) return aten<"zeros">({n}, torchlean::options().dtype(dtype));
  if (!(1.0 > p)) return aten<"ones">({n}, torchlean::options().dtype(dtype));
  const auto draws = uniform(n, key, dtype);
  return aten<"to.dtype">(aten<"lt.Tensor">(draws, scalar_for(draws, p)), dtype);
}

Tensor upload(b_lean_obj_arg object, ScalarType dtype) {
  const size_t n = lean_sarray_size(object);
  TORCHLEAN_CHECK(n <= INT64_MAX, "LibTorch upload: input is too large");
  if (n == 0) return aten<"empty.memory_format">({0}, torchlean::options().dtype(dtype));
  // The blocking copy ends before Lean can release or mutate the borrowed host array.
  const auto host = host_view(lean_float_array_cptr(object), static_cast<int64_t>(n), kDouble);
  return device_copy(host, dtype);
}

lean_obj_res download(b_lean_obj_arg object) {
  const auto* buffer = torchlean_cuda_buffer_unbox(object);
  std::optional<Tensor> host;
  if (buffer->size != 0) host = host_copy(torchlean::tensor(object), kDouble);
  lean_object* out = lean_mk_empty_float_array(lean_box(buffer->size));
  lean_sarray_set_size(out, buffer->size);
  if (buffer->size != 0)
    std::memcpy(lean_float_array_cptr(out), host->const_data_ptr<double>(),
                checked_bytes_size(buffer->size, sizeof(double), "FloatArray size overflow"));
  return out;
}

Tensor upload_bytes(b_lean_obj_arg object, ScalarType dtype, ScalarType source_dtype) {
  const size_t bytes = lean_sarray_size(object);
  const size_t width = element_size(source_dtype);
  TORCHLEAN_CHECK(bytes % width == 0, "checkpoint payload length does not match its dtype");
  const size_t n = bytes / width;
  const auto host = aten<"empty.memory_format">({static_cast<int64_t>(n)},
                                                tensor_options{}.dtype(source_dtype));
  const auto* source = reinterpret_cast<const uint8_t*>(lean_sarray_cptr(object));
  auto* values = static_cast<uint8_t*>(host.mutable_data_ptr());
  // Encoded arithmetic already supplies byte-ordered words, independent of host endianness.
  if (width == 1) {
    if (bytes) std::memcpy(values, source, bytes);
  } else for (size_t i = 0; i != n; ++i) {
    uint64_t bits = 0;
    for (size_t j = 0; j != width; ++j) bits |= uint64_t(source[width * i + j]) << (8 * j);
    if (width == 4) {
      const uint32_t word = static_cast<uint32_t>(bits);
      std::memcpy(values + width * i, &word, width);
    } else std::memcpy(values + width * i, &bits, width);
  }
  return device_copy(host, dtype);
}

lean_obj_res download_bytes(b_lean_obj_arg object, ScalarType dtype) {
  const auto* buffer = torchlean_cuda_buffer_unbox(object);
  const size_t bytes =
      checked_bytes_size(buffer->size, element_size(dtype), "checkpoint size overflow");
  std::optional<Tensor> host;
  if (bytes != 0) host = host_copy(torchlean::tensor(object), dtype);
  lean_object* out = lean_alloc_sarray(1, bytes, bytes);
  auto* destination = reinterpret_cast<uint8_t*>(lean_sarray_cptr(out));
  const size_t width = element_size(dtype);
  for (size_t i = 0; i != buffer->size; ++i) {
    uint64_t bits = 0;
    const auto* values = static_cast<const uint8_t*>(host->const_data_ptr());
    if (width == 4) {
      uint32_t word;
      std::memcpy(&word, values + width * i, width);
      bits = word;
    } else std::memcpy(&bits, values + width * i, width);
    for (size_t j = 0; j != width; ++j)
      destination[width * i + j] = static_cast<uint8_t>(bits >> (8 * j));
  }
  return out;
}

// Initial policy from TORCHLEAN_CUDA_DETERMINISTIC_REDUCTIONS: unset, empty, or "0" means off.
bool deterministic_reductions_requested() {
  const char* value = std::getenv("TORCHLEAN_CUDA_DETERMINISTIC_REDUCTIONS");
  return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
}

}  // namespace

namespace torchlean {

void initialize() {
  std::call_once(initialization, [] {
    unstable::initialize(deterministic_reductions_requested());
    // An empty tensor on the device initializes torch's CUDA state, as
    // `at::globalContext().lazyInitDevice(kCUDA)` does.
    aten<"empty.memory_format">({0}, tensor_options{}.dtype(kByte).device(device()));
  });
}

std::int32_t device_index() { return selected_device.load(); }

tensor_options options() { return tensor_options{}.dtype(kFloat).device(device()); }

// A buffer's tensor never requires gradients: the bridge never requests them, the entry points
// other than `of_handle` compute in no-grad mode, and the native tests check what they box.
const Tensor& tensor(b_lean_obj_arg object) {
  const auto* buffer = torchlean_cuda_buffer_unbox(object);
  TORCHLEAN_CHECK(buffer->tensor.has_value(), "LibTorch: buffer has been released");
  const auto& value = *buffer->tensor;
  TORCHLEAN_CHECK(value.is_cuda() &&
                  (value.scalar_type() == kFloat || value.scalar_type() == kDouble),
                  "LibTorch: expected a CUDA real buffer");
  TORCHLEAN_CHECK(value.numel() == static_cast<int64_t>(buffer->size),
                  "LibTorch: buffer size disagrees with storage");
  return value;
}

torchlean_cuda_buffer* owned(Tensor value) {
  TORCHLEAN_CHECK(value.defined() && value.is_cuda() &&
                  (value.scalar_type() == kFloat || value.scalar_type() == kDouble),
                  "LibTorch: expected a CUDA real result");
  auto buffer = std::make_unique<torchlean_cuda_buffer>();
  buffer->tensor = aten<"contiguous">(aten<"reshape">(value, {-1}));
  buffer->size = static_cast<size_t>(buffer->tensor->numel());
  const auto bytes = checked_bytes_size(buffer->size, buffer->tensor->element_size(),
                                         "LibTorch buffer byte size overflow");
  if (buffer->size != 0) payloads.add(bytes);
  return buffer.release();
}

lean_obj_res box(Tensor value) {
  return torchlean_cuda_buffer_box(owned(std::move(value)));
}

lean_obj_res encoded(Tensor value, b_lean_obj_arg format, size_t width) {
  TORCHLEAN_CHECK(value.is_cuda() && value.scalar_type() == kByte && width > 0 &&
                  value.numel() % width == 0, "encoded buffer: invalid storage extent");
  TORCHLEAN_CHECK(static_cast<uint64_t>(value.numel()) / width <= UINT32_MAX,
                  "encoded buffer: scalar count exceeds the tape ABI");
  auto buffer = std::make_unique<torchlean_cuda_buffer>();
  buffer->tensor = aten<"contiguous">(aten<"reshape">(value, {-1}));
  buffer->size = buffer->tensor->numel() / width;
  buffer->width = width;
  buffer->format = format;
  lean_inc(format);
  if (buffer->size) payloads.add(nbytes(*buffer->tensor));
  return torchlean_cuda_buffer_box(buffer.release());
}

// Context.divNat on the device: each entry divided once by the exact count, through a device
// tensor divisor. ATen turns a CPU scalar divisor into a product with its rounded reciprocal, which
// changes bits: binary32 147 * 2^-149 / 98 rounds to 2^-148, the product to 2^-149. Below 2^53 a
// binary64 count is exact and the quotient is one IEEE division. binary32 divides in binary64 and
// then rounds to binary32; for finite s and 1 <= n < 2^29, RN32(RN64(s / n)) = RN32(s / n). Since
// |s| >= |s / n|, s is a multiple of the binary32 spacing D at s / n and n * M of D / 2 for a
// binary32 midpoint M, so a quotient other than M lies at least D / (2n) > D * 2^-30 from M, more
// than binary64's half spacing there: the binary64 rounding cannot reach or cross M, and a midpoint
// quotient is exact in binary64 and takes binary32's tie rule. Subnormal quotients have
// D = 2^-149, and a nonzero quotient exceeds 2^-178, a normal binary64 number. Zeros keep their
// sign; infinities and NaNs pass both roundings; the count zero is the IEEE division by zero.
Tensor divide_by_count(const Tensor& x, uint64_t count) {
  const auto dtype = x.scalar_type();
  if (dtype == kDouble) {
    require(count < (uint64_t{1} << 53), "divCount: a binary64 count must be below 2^53");
    return aten<"div.Tensor">(x, scalar_tensor(static_cast<double>(count), torchlean::options(x)));
  }
  require(dtype == kFloat, "divCount: a binary32 or binary64 buffer is required");
  require(count < (uint64_t{1} << 29), "divCount: a binary32 count must be below 2^29");
  const auto wide = aten<"to.dtype">(x, kDouble);
  return aten<"to.dtype">(
      aten<"div.Tensor">(wide, scalar_tensor(static_cast<double>(count), torchlean::options(wide))),
      kFloat);
}

}  // namespace torchlean

extern "C" torchlean_cuda_buffer* torchlean_cuda_buffer_unbox(b_lean_obj_arg object) {
  torchlean::require(lean_is_external(object), "LibTorch: expected an external buffer");
  TORCHLEAN_CHECK(lean_get_external_class(object) == buffer_class(),
                  "LibTorch: external object is not a tensor buffer");
  return static_cast<torchlean_cuda_buffer*>(lean_get_external_data(object));
}

extern "C" lean_obj_res torchlean_cuda_buffer_box(torchlean_cuda_buffer* buffer) {
  auto* result = lean_alloc_external(buffer_class(), buffer);
  wrappers.add(1);
  return result;
}

extern "C" void torchlean_cuda_buffer_drop_unboxed(torchlean_cuda_buffer* buffer) {
  if (!buffer) return;
  release_data(buffer);
  if (buffer->format) lean_dec(buffer->format);
  delete buffer;
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_of_handle(AtenTensorHandle handle) {
  return torchlean::box(Tensor(handle));
}

extern "C" LEAN_EXPORT AtenTensorHandle torchlean_cuda_buffer_handle(b_lean_obj_arg object) {
  return torchlean::tensor(object).get();
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_runtime_status() {
  return lean_io_result_mk_ok(lean_box_uint32(device_count() > 0 ? 1u : 2u));
}

// Ownership counters are relaxed atomic loads, so reading them never initializes CUDA.
#define TORCHLEAN_COUNTER(NAME, VALUE)                                             \
  extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_##NAME() {                     \
    return lean_io_result_mk_ok(                                                   \
        lean_box_uint64((VALUE).load(std::memory_order_relaxed)));                 \
  }
TORCHLEAN_COUNTER(allocator_live_bytes, payloads.live)
TORCHLEAN_COUNTER(allocator_peak_bytes, payloads.peak)
TORCHLEAN_COUNTER(allocator_alloc_count, payloads.created)
TORCHLEAN_COUNTER(allocator_free_count, payloads.retired)
TORCHLEAN_COUNTER(wrapper_live_count, wrappers.live)
TORCHLEAN_COUNTER(wrapper_peak_count, wrappers.peak)
TORCHLEAN_COUNTER(wrapper_alloc_count, wrappers.created)
TORCHLEAN_COUNTER(wrapper_finalize_count, wrappers.retired)
#undef TORCHLEAN_COUNTER

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_allocator_device_free_bytes() {
  return io([] { return lean_box_uint64(memory_info(false)); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_allocator_device_total_bytes() {
  return io([] { return lean_box_uint64(memory_info(true)); });
}

#define TORCHLEAN_ALLOCATOR_STAT(NAME, FIELD)                                         \
  extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_##NAME() {                 \
    return io([] {                                                                  \
      return lean_box_uint64(unstable::allocator_stats(torchlean::device_index()).FIELD); \
    });                                                                             \
  }
TORCHLEAN_ALLOCATOR_STAT(allocated_bytes, allocated)
TORCHLEAN_ALLOCATOR_STAT(reserved_bytes, reserved)
TORCHLEAN_ALLOCATOR_STAT(peak_allocated_bytes, peak_allocated)
TORCHLEAN_ALLOCATOR_STAT(peak_reserved_bytes, peak_reserved)
#undef TORCHLEAN_ALLOCATOR_STAT

extern "C" LEAN_EXPORT uint32_t torchlean_cuda_buffer_size(b_lean_obj_arg object) {
  const auto* buffer = torchlean_cuda_buffer_unbox(object);
  torchlean::require(buffer->size <= UINT32_MAX, "LibTorch: buffer size exceeds UInt32");
  return static_cast<uint32_t>(buffer->size);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_size_io(b_lean_obj_arg object) {
  return lean_io_result_mk_ok(lean_box_uint32(torchlean_cuda_buffer_size(object)));
}

extern "C" LEAN_EXPORT uint32_t torchlean_cuda_buffer_release(b_lean_obj_arg object) {
  return release_data(torchlean_cuda_buffer_unbox(object)) ? 1 : 0;
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_release_io(
    b_lean_obj_arg object) {
  return lean_io_result_mk_ok(lean_box_uint32(torchlean_cuda_buffer_release(object)));
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_release_then(
    b_lean_obj_arg scratch, b_lean_obj_arg keep) {
  torchlean_cuda_buffer_release(scratch);
  lean_inc(keep);
  return keep;
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_runtime_collect_allocator() {
  mi_collect(false);
  return lean_io_result_mk_ok(lean_box(0));
}

#define TORCHLEAN_CONSTRUCTOR(NAME, PARAMETERS, EXPRESSION)                         \
  extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_##NAME PARAMETERS {     \
    return torchlean::invoke([&] { return torchlean::box(EXPRESSION); });             \
  }                                                                               \
  extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_##NAME##_io PARAMETERS { \
    return io([&] { return torchlean::box(EXPRESSION); });                            \
  }
TORCHLEAN_CONSTRUCTOR(zeros, (uint32_t n, uint8_t dtype),
                     aten<"zeros">({n}, torchlean::options().dtype(scalar_type(dtype))))
TORCHLEAN_CONSTRUCTOR(full, (uint32_t n, double v, uint8_t dtype),
                     full({n}, rounded(v, scalar_type(dtype)),
                          torchlean::options().dtype(scalar_type(dtype))))
TORCHLEAN_CONSTRUCTOR(of_float_array, (b_lean_obj_arg object, uint8_t dtype),
                     upload(object, scalar_type(dtype)))
#undef TORCHLEAN_CONSTRUCTOR

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_rand_uniform_io(
    uint32_t n, uint64_t key, uint8_t dtype) {
  return io([&] { return torchlean::box(uniform(n, key, scalar_type(dtype))); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_bernoulli_mask_io(
    uint32_t n, double p, uint64_t key, uint8_t dtype) {
  return io([&] { return torchlean::box(bernoulli(n, p, key, scalar_type(dtype))); });
}

extern "C" LEAN_EXPORT uint8_t torchlean_cuda_buffer_dtype(b_lean_obj_arg object) {
  return torchlean::invoke([&]() -> uint8_t {
    const auto* buffer = torchlean_cuda_buffer_unbox(object);
    if (buffer->format) {
      TORCHLEAN_CHECK(buffer->tensor.has_value(), "encoded buffer: released handle");
      return 2;
    }
    return torchlean::tensor(object).scalar_type() == kFloat ? 0 : 1;
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_fail(b_lean_obj_arg message) {
  return torchlean::invoke([&]() -> lean_obj_res {
    fail(lean_string_cstr(message));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_zeros_like(
    b_lean_obj_arg reference, uint32_t count) {
  return torchlean::invoke([&] {
    auto* buffer = torchlean_cuda_buffer_unbox(reference);
    TORCHLEAN_CHECK(buffer->tensor.has_value(), "zeros: released buffer");
    const uint64_t width = buffer->format ? buffer->width : 1;
    TORCHLEAN_CHECK(width > 0 && count <= INT64_MAX / width, "zeros: invalid extent");
    auto value = aten<"zeros">({static_cast<int64_t>(count * width)},
                               torchlean::options(*buffer->tensor));
    return buffer->format ? torchlean::encoded(value, buffer->format, width) : torchlean::box(value);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_copy(b_lean_obj_arg reference) {
  return torchlean::invoke([&] {
    auto* buffer = torchlean_cuda_buffer_unbox(reference);
    TORCHLEAN_CHECK(buffer->tensor.has_value(), "copy: released buffer");
    auto value = aten<"clone">(*buffer->tensor);
    return buffer->format ? torchlean::encoded(value, buffer->format, buffer->width)
                          : torchlean::box(value);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_duplicate(b_lean_obj_arg reference) {
  return torchlean::invoke([&] {
    auto* buffer = torchlean_cuda_buffer_unbox(reference);
    TORCHLEAN_CHECK(buffer->tensor.has_value(), "duplicate: released buffer");
    auto first = aten<"clone">(*buffer->tensor);
    auto second = aten<"clone">(*buffer->tensor);
    auto* result = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(result, 0, buffer->format
      ? torchlean::encoded(first, buffer->format, buffer->width) : torchlean::box(first));
    lean_ctor_set(result, 1, buffer->format
      ? torchlean::encoded(second, buffer->format, buffer->width) : torchlean::box(second));
    return result;
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_to_float_array(b_lean_obj_arg object) {
  return torchlean::invoke([&] { return download(object); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_to_float_array_io(b_lean_obj_arg object) {
  return io([&] { return download(object); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_kernel_run_buffer(
    b_lean_obj_arg source, uint8_t dtype, b_lean_obj_arg objects, uint64_t count) {
  return io([&] {
    std::vector<Tensor> inputs;
    inputs.reserve(lean_array_size(objects));
    for (size_t i = 0; i < lean_array_size(objects); ++i)
      inputs.push_back(torchlean::tensor(lean_array_uget(objects, i)));
    return torchlean::box(run_kernel(source, inputs, scalar_type(dtype), count));
  });
}

// The source's scalar layout and stride are generated together by Lean. Byte tensors
// preserve wide words, signed zeros and NaN payloads without numerical conversion.
extern "C" LEAN_EXPORT lean_obj_res torchlean_kernel_run_bytes(
    b_lean_obj_arg source, uint32_t format, uint64_t width,
    b_lean_obj_arg arrays, uint64_t count) {
  return io([&] {
    TORCHLEAN_CHECK(format <= 2 && width > 0, "kernel: invalid encoded scalar layout");
    const auto scalar = format == 0 ? kFloat : format == 1 ? kDouble : kByte;
    TORCHLEAN_CHECK(format == 2 || width == element_size(scalar),
                    "kernel: native scalar width mismatch");
    std::vector<Tensor> inputs;
    inputs.reserve(lean_array_size(arrays));
    for (size_t i = 0; i < lean_array_size(arrays); ++i) {
      const auto object = lean_array_uget(arrays, i);
      const auto bytes = lean_sarray_size(object);
      TORCHLEAN_CHECK(bytes <= INT64_MAX && bytes % width == 0,
                      "kernel: invalid input byte extent");
      const auto elements = bytes / element_size(scalar);
      if (!bytes) {
        inputs.push_back(aten<"empty.memory_format">({0}, torchlean::options().dtype(scalar)));
      } else {
        const auto host =
            host_view(lean_sarray_cptr(object), static_cast<int64_t>(elements), scalar);
        inputs.push_back(device_copy(host, scalar));
      }
    }
    const auto result = run_kernel(source, inputs, scalar, count, format == 2 ? width : 1);
    const auto host = host_copy(result);
    TORCHLEAN_CHECK(count <= SIZE_MAX / width, "kernel: host byte size overflow");
    const auto bytes = static_cast<size_t>(count * width);
    auto output = lean_alloc_sarray(1, bytes, bytes);
    if (bytes) std::memcpy(lean_sarray_cptr(output), host.const_data_ptr(), bytes);
    return output;
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_format(b_lean_obj_arg object) {
  auto* buffer = torchlean_cuda_buffer_unbox(object);
  if (!buffer->format) return lean_box(0);
  auto* result = lean_alloc_ctor(1, 1, 0);
  lean_inc(buffer->format);
  lean_ctor_set(result, 0, buffer->format);
  return result;
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_of_encoded_io(
    b_lean_obj_arg bytes, b_lean_obj_arg format, uint64_t width) {
  return io([&] {
    TORCHLEAN_CHECK(width && lean_sarray_size(bytes) % width == 0,
                    "encoded buffer: truncated scalar word");
    return torchlean::encoded(upload_bytes(bytes, kByte, kByte), format, width);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_to_encoded_io(
    b_lean_obj_arg object) {
  return io([&] {
    auto* buffer = torchlean_cuda_buffer_unbox(object);
    TORCHLEAN_CHECK(buffer->format && buffer->tensor, "encoded buffer: invalid handle");
    const auto host = host_copy(*buffer->tensor);
    const size_t bytes = nbytes(host);
    auto* result = lean_alloc_sarray(1, bytes, bytes);
    if (bytes) std::memcpy(lean_sarray_cptr(result), host.const_data_ptr(), bytes);
    return result;
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_kernel_run_encoded(
    b_lean_obj_arg source, b_lean_obj_arg objects, b_lean_obj_arg format,
    uint64_t width, uint64_t count) {
  return torchlean::invoke([&] {
    std::vector<Tensor> inputs;
    for (size_t i = 0; i < lean_array_size(objects); ++i) {
      auto* buffer = torchlean_cuda_buffer_unbox(lean_array_uget(objects, i));
      TORCHLEAN_CHECK(buffer->format && buffer->width == width && buffer->tensor,
                      "encoded buffer: incompatible input storage");
      inputs.push_back(*buffer->tensor);
    }
    return torchlean::encoded(run_kernel(source, inputs, kByte, count, width), format, width);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_to_bytes_io(
    b_lean_obj_arg object, uint8_t dtype) {
  return io([&] { return download_bytes(object, scalar_type(dtype)); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_of_bytes_io(
    b_lean_obj_arg object, uint8_t dtype, uint8_t source) {
  return io([&] { return torchlean::box(upload_bytes(object, scalar_type(dtype), scalar_type(source))); });
}

// The release of the loaded torch runtime, which may be later than the release the bridge is
// built against.
extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_version() {
  const uint64_t version = aoti_torch_abi_version();
  const std::string release = std::to_string(version >> 56) + "." +
                              std::to_string((version >> 48) & 0xff) + "." +
                              std::to_string((version >> 40) & 0xff);
  return lean_io_result_mk_ok(lean_mk_string(release.c_str()));
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_device_count() {
  return lean_io_result_mk_ok(lean_box_uint32(static_cast<uint32_t>(device_count())));
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_get_device() {
  return lean_io_result_mk_ok(lean_box_uint32(static_cast<uint32_t>(selected_device.load())));
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_set_device(uint32_t index) {
  return io([&] {
    TORCHLEAN_CHECK(index < static_cast<uint32_t>(device_count()),
                    "LibTorch: selected CUDA device does not exist");
    TORCHLEAN_CHECK(wrappers.live.load() == 0,
                    "LibTorch: select the device before creating tensor buffers");
    selected_device.store(static_cast<int>(index));
    return lean_box(0);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_get_setting(uint32_t setting) {
  return io([&] { return lean_box_uint32(unstable::setting(setting) ? 1 : 0); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_set_setting(
    uint32_t setting, uint32_t enabled) {
  return io([&] {
    TORCHLEAN_CHECK(enabled <= 1, "LibTorch: runtime boolean must be zero or one");
    unstable::set_setting(setting, enabled != 0);
    return lean_box(0);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_get_memory_fraction() {
  return io([] {
    return lean_box_float(unstable::memory_fraction(torchlean::device_index()));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_set_memory_fraction(double fraction) {
  return io([&] {
    TORCHLEAN_CHECK(std::isfinite(fraction) && fraction > 0.0 && fraction <= 1.0,
                    "LibTorch: memory fraction must be finite and in (0, 1]");
    unstable::set_memory_fraction(fraction, torchlean::device_index());
    return lean_box(0);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_synchronize() {
  return io([] {
    synchronize();
    return lean_box(0);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_empty_cache() {
  return io([] {
    unstable::empty_cache();
    return lean_box(0);
  });
}

// Elementwise operations


#include <algorithm>
#include <tuple>

namespace torchlean::elementwise {

namespace {

using torchlean::box;
using torchlean::divide_by_count;
using torchlean::invoke;
using torchlean::pair;
using torchlean::require;
using torchlean::tensor;
using torchlean::triple;

Tensor flat(b_lean_obj_arg object) {
  return aten<"reshape">(tensor(object), {-1});
}

void same_size(const Tensor& a, const Tensor& b, const char* message) {
  require(a.numel() == b.numel(), message);
  require(a.scalar_type() == b.scalar_type(), "LibTorch buffer operation: dtype mismatch");
}

Tensor selected_sqrt(const Tensor& x) {
  // Ordered comparison preserves NaN and selects positive zero for both signed zeros.
  return aten<"sqrt">(
      aten<"where.self">(aten<"le.Tensor">(x, scalar_for(x, 0.0)), scalar_for(x, 0.0), x));
}

Tensor selected_relu(const Tensor& x) {
  // Match Activation.relu's ordered maximum: preserve NaN, but normalize both zeros to +0.
  return aten<"where.self">(aten<"le.Tensor">(x, scalar_for(x, 0.0)), scalar_for(x, 0.0), x);
}

// a + b c with c rounded to the dtype, as fma(b, c, a): one rounding. CUDA addcmul evaluates this
// std::fma when its value is exactly one (DeviceAddCmulCdiv.cuh), but that value is a Scalar, which
// the stable dispatcher does not pass, and add with alpha fuses only if the SDK's compiler
// contracted it, so a generated kernel calls the fma. TorchLeanTest/CUDA/Numerics.lean checks it.
Tensor axpy(const Tensor& a, const Tensor& b, double c) {
  const auto dtype = a.scalar_type();
  TORCHLEAN_CHECK((dtype == kFloat || dtype == kDouble) && b.scalar_type() == dtype &&
                  b.numel() == a.numel(), "axpy: operands of one size and real dtype expected");
  const auto x = aten<"contiguous">(a);
  const auto y = aten<"contiguous">(b);
  const auto result = aten<"empty_like">(x);
  const std::string type = dtype == kFloat ? "float" : "double";
  const std::string source =
      "extern \"C\" __global__ void torchlean_kernel(const " + type + "* a, const " + type +
      "* b, " + type + "* out, double c, unsigned long long n) {\n"
      "  unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n"
      "  if (index < n) out[index] = " + (dtype == kFloat ? "fmaf" : "fma") + "(b[index], (" +
      type + ")c, a[index]);\n"
      "}\n";
  // The coefficient travels as the bits of a binary64 argument.
  const double coefficient = rounded(c, dtype);
  uint64_t bits = 0;
  std::memcpy(&bits, &coefficient, sizeof bits);
  const auto n = static_cast<uint64_t>(x.numel());
  std::vector<launch> launches{{n, {reinterpret_cast<uint64_t>(x.const_data_ptr()),
                                    reinterpret_cast<uint64_t>(y.const_data_ptr()),
                                    reinterpret_cast<uint64_t>(result.mutable_data_ptr()),
                                    bits, n}}};
  launch_rows(source, launches);
  return result;
}

// Native buffers are real: an ordered magnitude bound rejects both infinities and NaNs.
Tensor finite(const Tensor& x) {
  return aten<"lt.Tensor">(aten<"abs">(x), scalar_for(x, std::numeric_limits<double>::infinity()));
}

Tensor mean_update(const Tensor& mean, const Tensor& sample, double weight) {
  require(std::isfinite(weight) && weight > 0.0 && weight <= 1.0,
          "LibTorch running mean: weight must be finite and in (0, 1]");
  const double previous_weight = rounded(1.0 - weight, mean.scalar_type());
  const auto weighted = aten<"mul.Tensor">(mean, scalar_for(mean, previous_weight));
  add_(weighted, aten<"mul.Tensor">(sample, scalar_for(sample, weight)));
  // Subtraction is safe for same-sign finite values; opposite-sign weighted terms cannot
  // overflow their sum. Bound the finite result by its operands to contain rounding overshoot.
  const auto candidate = [&] {
    const auto shifted = sub(sample, mean);
    add_(aten<"mul_.Tensor">(shifted, scalar_for(shifted, weight)), mean);
    return aten<"where.self">(
        aten<"eq.Tensor">(aten<"signbit">(mean), aten<"signbit">(sample)), shifted, weighted);
  }();
  // Only fresh workspaces are mutated; keep peak storage bounded for large parameter tensors.
  aten<"maximum.out">(candidate, aten<"minimum">(mean, sample), candidate);
  aten<"minimum.out">(candidate, aten<"maximum">(mean, sample), candidate);
  return aten<"where.self">(
      aten<"logical_and">(aten<"isfinite">(mean), aten<"isfinite">(sample)), candidate, weighted);
}

// The constants of `Activation.geluTanh`: `0.044715`, rounded to the dtype by `rounded`
// (binary32 0x3d372713, binary64 the value nearest 0.044715), and `sqrt(2 / pi)` computed in the
// dtype, the square root of the rounded quotient of 2 by the rounded pi: binary32 0x3f4c4229, one
// unit in the last place below PyTorch's kBeta rounded to binary32; binary64 the value nearest.
constexpr double kGeluKappa = 0.044715;

double gelu_beta(ScalarType dtype) {
  if (dtype == kFloat) {
    const float pi = static_cast<float>(M_PI);
    return static_cast<double>(std::sqrt(2.0f / pi));
  }
  require(dtype == kDouble, "gelu: the LibTorch buffers compute binary32 or binary64 only");
  return std::sqrt(2.0 / M_PI);
}

Tensor stable_sigmoid(const Tensor& x) {
  const auto z = aten<"exp">(aten<"neg">(aten<"abs">(x)));
  const auto denominator = add(z, scalar_for(z, 1.0));
  return aten<"where.self">(aten<"gt.Tensor">(x, scalar_for(x, 0.0)),
                            aten<"reciprocal">(denominator), aten<"div.Tensor">(z, denominator));
}

Tensor activation_backward(const Tensor& x, const Tensor& g, bool is_tanh) {
  const auto magnitude = aten<"abs">(x);
  const auto z = aten<"exp">(aten<"neg">(magnitude));
  const auto denominator = add(is_tanh ? aten<"mul.Tensor">(z, z) : z, scalar_for(z, 1.0));
  std::optional<Tensor> slope;
  if (is_tanh) {
    const auto reciprocal_cosh = aten<"div.Tensor">(add(z, z), denominator);
    slope = aten<"mul.Tensor">(reciprocal_cosh, reciprocal_cosh);
  } else {
    slope = aten<"div.Tensor">(z, aten<"mul.Tensor">(denominator, denominator));
  }
  const auto ordinary = aten<"mul.Tensor">(*slope, g);

  const auto half_argument = aten<"mul.Tensor">(magnitude, scalar_for(magnitude, -0.5));
  const auto q = aten<"exp">(half_argument);
  const auto gradient_magnitude = aten<"abs">(g);
  const double min_normal = x.scalar_type() == kFloat
      ? std::numeric_limits<float>::min() : std::numeric_limits<double>::min();
  std::optional<Tensor> repaired;
  if (x.scalar_type() == kFloat) {
    // The replay of TorchLean.Numeric.applySlope, q^2 g / (1 + q^2)^2 or
    // 4 q^4 g / (1 + q^4)^2, evaluated in binary64 and rounded once to binary32. Its binary64
    // error, below 8 units of 2^-53 relative, decides that rounding only next to a midpoint.
    // A binary32 q below the least normal sends the rule to its enclosure of the exact slope,
    // for which the binary64 exp stands in.
    const auto wide_magnitude = aten<"to.dtype">(magnitude, kDouble);
    const auto wide_q = aten<"where.self">(
        aten<"lt.Tensor">(q, scalar_for(q, min_normal)),
        aten<"exp">(aten<"mul.Tensor">(wide_magnitude, scalar_for(wide_magnitude, -0.5))),
        aten<"to.dtype">(q, kDouble));
    auto power = aten<"mul.Tensor">(wide_q, wide_q);
    if (is_tanh) power = aten<"mul.Tensor">(power, power);
    const auto wide_denominator = add(power, scalar_for(power, 1.0));
    auto weighted = aten<"div.Tensor">(aten<"mul.Tensor">(power, aten<"to.dtype">(g, kDouble)),
                                       aten<"mul.Tensor">(wide_denominator, wide_denominator));
    if (is_tanh) {
      weighted = aten<"mul.Tensor">(weighted, scalar_for(weighted, 4.0));
    } else {
      // At a nonzero x whose half argument rounds below the least normal, the rule encloses the
      // exact slope, which lies below the replay's 1/4 by a relative x^2/4 < 2^-250. One binary64
      // step toward zero rounds as the exact value does: both lie below g/4 and above every
      // binary32 number or midpoint below it. weighted * 0 keeps the sign of a zero.
      const auto enclosed = aten<"logical_and">(
          aten<"gt.Tensor">(magnitude, scalar_for(magnitude, 0.0)),
          aten<"gt.Tensor">(half_argument, scalar_for(half_argument, -min_normal)));
      weighted = aten<"where.self">(
          enclosed,
          aten<"nextafter">(weighted, aten<"mul.Tensor">(weighted, scalar_for(weighted, 0.0))),
          weighted);
    }
    repaired = aten<"to.dtype">(weighted, kFloat);
  } else {
    // Without a wider device type, share the incoming magnitude between the exponential
    // factors before squaring, so the weighted gradient fits where a rounded slope is zero.
    auto amplitude = aten<"mul.Tensor">(aten<"sqrt">(gradient_magnitude), q);
    if (is_tanh) {
      amplitude = aten<"mul.Tensor">(aten<"mul.Tensor">(amplitude, q), scalar_for(amplitude, 2.0));
    }
    amplitude = aten<"div.Tensor">(amplitude, denominator);
    repaired = aten<"copysign.Tensor">(aten<"mul.Tensor">(amplitude, amplitude), g);
  }
  const auto lost_range = aten<"logical_or">(
      aten<"lt.Tensor">(aten<"abs">(*slope), scalar_for(*slope, min_normal)),
      aten<"lt.Tensor">(aten<"abs">(ordinary), scalar_for(ordinary, min_normal)));
  const auto finite_gradient = aten<"lt.Tensor">(
      gradient_magnitude,
      scalar_for(gradient_magnitude, std::numeric_limits<double>::infinity()));
  const auto repair =
      aten<"logical_and">(lost_range, aten<"logical_and">(finite(x), finite_gradient));
  return aten<"where.self">(repair, *repaired, ordinary);
}

// The limits that the Lean definitions of GELU, SiLU and their slopes return at the infinities
// before their formulas, which compute `inf * 0` there: `at_positive` at `+inf`, `-0` at `-inf`.
Tensor infinity_limits(const Tensor& x, const Tensor& at_positive, const Tensor& formula) {
  const auto limit = aten<"where.self">(aten<"gt.Tensor">(x, scalar_for(x, 0.0)), at_positive,
                                        aten<"neg">(aten<"zeros_like">(x)));
  return aten<"where.self">(aten<"isinf">(x), limit, formula);
}

Tensor staged_silu(const Tensor& x) {
  // Bound the inactive exponential before evaluating either ATen branch.
  const auto negative = aten<"minimum">(x, aten<"zeros_like">(x));
  const auto r = aten<"exp">(aten<"mul.Tensor">(negative, scalar_for(negative, 0.5)));
  const auto denominator = add(aten<"mul.Tensor">(r, r), scalar_for(r, 1.0));
  const auto tail =
      aten<"div.Tensor">(aten<"mul.Tensor">(aten<"mul.Tensor">(x, r), r), denominator);
  return infinity_limits(x, x, aten<"where.self">(aten<"gt.Tensor">(x, scalar_for(x, 0.0)),
                                                  aten<"mul.Tensor">(x, stable_sigmoid(x)), tail));
}

Tensor staged_silu_backward(const Tensor& x, const Tensor& g) {
  const auto negative = aten<"minimum">(x, aten<"zeros_like">(x));
  const auto r = aten<"exp">(aten<"mul.Tensor">(negative, scalar_for(negative, 0.5)));
  const auto denominator = add(aten<"mul.Tensor">(r, r), scalar_for(r, 1.0));
  const auto tail = aten<"div.Tensor">(
      aten<"mul.Tensor">(aten<"mul.Tensor">(add(x, denominator), r), r),
      aten<"mul.Tensor">(denominator, denominator));
  const auto z = aten<"exp">(aten<"neg">(aten<"abs">(x)));
  const auto positive_denominator = add(z, scalar_for(z, 1.0));
  const auto positive_slope =
      aten<"div.Tensor">(z, aten<"mul.Tensor">(positive_denominator, positive_denominator));
  const auto positive =
      add(aten<"reciprocal">(positive_denominator), aten<"mul.Tensor">(x, positive_slope));
  return aten<"mul.Tensor">(
      infinity_limits(x, aten<"ones_like">(x),
                      aten<"where.self">(aten<"gt.Tensor">(x, scalar_for(x, 0.0)), positive, tail)),
      g);
}

Tensor gelu_inner(const Tensor& x) {
  // Each ATen call materializes one rounded stage in the input dtype: beta (x + kappa x^3).
  const auto cube = aten<"mul.Tensor">(aten<"mul.Tensor">(x, x), x);
  const auto inner =
      add(x, aten<"mul.Tensor">(cube, scalar_for(cube, rounded(kGeluKappa, x.scalar_type()))));
  return aten<"mul.Tensor">(inner, scalar_for(inner, gelu_beta(x.scalar_type())));
}

// `Activation.geluTanh` in its order, `(0.5 x) (1 + tanh (beta (x + kappa x^3)))`, one rounded
// ATen stage per operation, with its limits at the infinities; the device's `tanh` is its own
// approximation.
Tensor staged_gelu(const Tensor& x) {
  const auto gate = aten<"tanh">(gelu_inner(x));
  return infinity_limits(x, x, aten<"mul.Tensor">(aten<"mul.Tensor">(x, scalar_for(x, 0.5)),
                                                  add(gate, scalar_for(gate, 1.0))));
}

Tensor staged_gelu_backward(const Tensor& x, const Tensor& g) {
  const auto inner = gelu_inner(x);
  const auto gate = stable_sigmoid(add(inner, inner));
  const auto z = aten<"exp">(aten<"neg">(aten<"abs">(inner)));
  const auto reciprocal_cosh =
      aten<"div.Tensor">(add(z, z), add(aten<"mul.Tensor">(z, z), scalar_for(z, 1.0)));
  const auto sech_term = aten<"mul.Tensor">(reciprocal_cosh, reciprocal_cosh);
  const double scaled_coeff = rounded(3.0 * rounded(kGeluKappa, x.scalar_type()), x.scalar_type());
  const auto quadratic0 = aten<"mul.Tensor">(x, scalar_for(x, scaled_coeff));
  const auto quadratic = aten<"mul.Tensor">(quadratic0, x);
  const auto inner_deriv = aten<"mul.Tensor">(add(quadratic, scalar_for(quadratic, 1.0)),
                                              scalar_for(quadratic, gelu_beta(x.scalar_type())));
  const auto derivative_term = aten<"where.self">(
      aten<"eq.Tensor">(sech_term, scalar_for(sech_term, 0.0)), aten<"zeros_like">(x),
      aten<"mul.Tensor">(aten<"mul.Tensor">(x, sech_term), inner_deriv));
  const auto numerator = add(add(gate, gate), derivative_term);
  return aten<"mul.Tensor">(
      infinity_limits(x, aten<"ones_like">(x),
                      aten<"mul.Tensor">(numerator, scalar_for(numerator, 0.5))),
      g);
}

// Where a and b are equal zeros, of either sign.
Tensor equal_zeros(const Tensor& a, const Tensor& b) {
  return aten<"logical_and">(aten<"eq.Tensor">(a, b),
                             aten<"eq.Tensor">(a, scalar_for(a, std::int64_t{0})));
}

// IEEE 754-2019 9.6 `maximum` from ATen pieces: `aten::maximum` returns a NaN when an operand is
// one, and at equal zeros `a + b` is the larger zero in round to nearest, `+0` unless both are
// `-0`.
Tensor ieee_maximum(const Tensor& a, const Tensor& b) {
  return aten<"where.self">(equal_zeros(a, b), add(a, b), aten<"maximum">(a, b));
}

// 9.6 `minimum`: at equal zeros `-((-a) + (-b))` is the smaller zero, `-0` unless both are `+0`.
Tensor ieee_minimum(const Tensor& a, const Tensor& b) {
  return aten<"where.self">(equal_zeros(a, b), aten<"neg">(add(aten<"neg">(a), aten<"neg">(b))),
                            aten<"minimum">(a, b));
}

lean_obj_res minmax_backward(const Tensor& a, const Tensor& b, const Tensor& g, bool maximum) {
  const auto a_wins = maximum ? aten<"gt.Tensor">(a, b) : aten<"gt.Tensor">(b, a);
  const auto b_wins = maximum ? aten<"gt.Tensor">(b, a) : aten<"gt.Tensor">(a, b);
  // The CPU rule, by selection: the strict winner receives g bit for bit, the strict loser +0 at
  // every g, and a tie or an unordered pair (a NaN operand) half of g each.
  const auto zero = aten<"zeros_like">(g);
  const auto half = aten<"mul.Tensor">(full_like(g, 0.5), g);
  auto da = aten<"where.self">(a_wins, g, aten<"where.self">(b_wins, zero, half));
  auto db = aten<"where.self">(b_wins, g, aten<"where.self">(a_wins, zero, half));
  return pair(std::move(da), std::move(db));
}

Tensor deterministic_sum(const Tensor& input) {
  // Preserve the specified 256-lane tree and capped grid-stride accumulation
  // order using upstream tensor operations. A different reduction tree can
  // change finite results even when it is itself deterministic.
  constexpr int64_t block_size = 256;
  constexpr int64_t max_blocks = 65535;
  if (input.numel() == 0) {
    return aten<"zeros">({1}, options(input));
  }
  auto current = input;
  do {
    const int64_t n = current.numel();
    const int64_t blocks = std::min((n + block_size - 1) / block_size, max_blocks);
    const int64_t stride = blocks * block_size;
    const auto lanes = aten<"zeros">({stride}, options(input));
    for (int64_t offset = 0; offset < n; offset += stride) {
      const int64_t count = std::min(stride, n - offset);
      // Only participating lanes add an input; inactive lanes keep their bits.
      // Each lane's first addition starts at +0.
      add_(aten<"narrow">(lanes, 0, 0, count), aten<"narrow">(current, 0, offset, count));
    }
    auto tree = aten<"reshape">(lanes, {blocks, block_size});
    for (int64_t width = block_size / 2; width != 0; width /= 2) {
      tree = add(aten<"narrow">(tree, 1, 0, width), aten<"narrow">(tree, 1, width, width));
    }
    current = aten<"reshape">(tree, {blocks});
  } while (current.numel() > 1);
  return current;
}

Tensor reduce_sum(const Tensor& x) {
  if (unstable::deterministic()) {
    return deterministic_sum(x);
  }
  return aten<"reshape">(aten<"sum">(x), {1});
}

Tensor reduce_mean(const Tensor& x) {
  // Op.mean over every axis: the sum divided once by the element count, by one for an empty
  // buffer, whose sum is zero.
  return divide_by_count(reduce_sum(x), std::max<uint64_t>(static_cast<uint64_t>(x.numel()), 1));
}

// Tensor.meanSquaredError of the differences d: the sum of the squares divided once by the count
// n, or, where that sum overflows while every difference is finite, the sum of d * (d / n), whose
// exact terms do not exceed the exact mean. Both sums are reduce_sum's.
Tensor mse_mean(const Tensor& d, uint64_t count) {
  const auto total = reduce_sum(aten<"mul.Tensor">(d, d));
  const auto repair =
      aten<"logical_and">(aten<"logical_not">(finite(total)), aten<"all">(finite(d)));
  return aten<"where.self">(repair, reduce_sum(aten<"mul.Tensor">(d, divide_by_count(d, count))),
                            divide_by_count(total, count));
}

// Tensor.meanSquaredErrorBackward's slope 2d / n at each difference d: a finite d + d divided once
// by n, and otherwise d / n doubled. Both are 2d / n rounded once: a finite d + d is exact, and
// for n >= 2 so is the doubling of d / n, which stays normal at every count divide_by_count
// accepts (the quotient of a d with an overflowing double is subnormal only past 2^(emax - emin)).
Tensor mse_slope(const Tensor& d, uint64_t count) {
  const auto twice = add(d, d);
  const auto quotient = divide_by_count(d, count);
  return aten<"where.self">(finite(twice), divide_by_count(twice, count), add(quotient, quotient));
}

// Huber.term at each prediction p and target t with the threshold delta, every operation in
// the buffers' dtype and in the rule's order: (|d| / 2) |d| below delta, else delta (|d| - delta /
// 2), or |delta p - delta t| - (delta / 2) delta where d = p - t is not finite, delta > 0 and that
// weighted difference is finite.
Tensor huber_terms(const Tensor& p, const Tensor& t, double delta) {
  const auto threshold = scalar_tensor(delta, options(p));
  const auto half = aten<"mul.Tensor">(threshold, scalar_for(threshold, 0.5));
  const auto difference = sub(p, t);
  const auto x = aten<"abs">(difference);
  const auto direct = aten<"mul.Tensor">(threshold, sub(x, half));
  const auto weighted =
      aten<"abs">(sub(aten<"mul.Tensor">(threshold, p), aten<"mul.Tensor">(threshold, t)));
  const auto positive = aten<"gt.Tensor">(threshold, scalar_for(threshold, 0.0));
  const auto keep = aten<"logical_or">(
      aten<"logical_or">(finite(difference), aten<"logical_not">(positive)),
      aten<"logical_not">(finite(weighted)));
  const auto linear =
      aten<"where.self">(keep, direct, sub(weighted, aten<"mul.Tensor">(half, threshold)));
  return aten<"where.self">(aten<"lt.Tensor">(x, threshold),
                            aten<"mul.Tensor">(aten<"mul.Tensor">(x, scalar_for(x, 0.5)), x),
                            linear);
}

// Tensor.huberLossBackward's slope at each difference d before the mean's division: d
// where |d| < delta, else delta, -delta or +0 by the sign of d (+0 at a NaN).
Tensor huber_slope(const Tensor& d, double delta) {
  const auto threshold = scalar_tensor(delta, options(d));
  const auto positive = aten<"gt.Tensor">(d, scalar_for(d, 0.0));
  const auto magnitude = aten<"where.self">(positive, d, aten<"neg">(d));
  const auto outer = aten<"where.self">(positive, threshold,
      aten<"where.self">(aten<"lt.Tensor">(d, scalar_for(d, 0.0)), aten<"neg">(threshold),
                         aten<"zeros_like">(d)));
  return aten<"where.self">(aten<"lt.Tensor">(magnitude, threshold), d, outer);
}

// All generated calls share conversion, validation, device selection, and result ownership.
Tensor argument(b_lean_obj_arg object) { return flat(object); }
double argument(double scalar) { return scalar; }

void check_argument(int64_t& size, ScalarType& dtype, const Tensor& value) {
  if (size < 0) { size = value.numel(); dtype = value.scalar_type(); }
  require(value.numel() == size, "LibTorch buffer operation: size mismatch");
  require(value.scalar_type() == dtype, "LibTorch buffer operation: dtype mismatch");
}
void check_argument(int64_t&, ScalarType&, double) {}

const Tensor& rounded_argument(const Tensor& value, ScalarType) { return value; }
double rounded_argument(double value, ScalarType dtype) { return rounded(value, dtype); }

template <typename Function, typename... Args>
lean_obj_res call_buffer(Function&& function, Args... args) {
  return invoke([&]() {
    auto converted = std::make_tuple(argument(args)...);
    return std::apply([&](const auto&... values) {
      int64_t size = -1;
      ScalarType dtype = kFloat;
      (check_argument(size, dtype, values), ...);
      return box(function(rounded_argument(values, dtype)...));
    }, converted);
  });
}

}  // namespace

#define TORCHLEAN_BUFFER_EXPORT(NAME, PARAMETERS, ARGUMENTS, FUNCTION) \
  extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_##NAME PARAMETERS { \
    return call_buffer(FUNCTION, TORCHLEAN_ARGUMENTS ARGUMENTS); \
  }

// Parentheses keep argument lists intact while expanding the signature macros.
#define TORCHLEAN_ARGUMENTS(...) __VA_ARGS__
#define TORCHLEAN_UNARY_EXPORT(NAME, EXPRESSION) \
  TORCHLEAN_BUFFER_EXPORT(NAME, (b_lean_obj_arg x), (x), \
      ([](const auto& x) { return EXPRESSION; }))
#define TORCHLEAN_BINARY_EXPORT(NAME, EXPRESSION) \
  TORCHLEAN_BUFFER_EXPORT(NAME, (b_lean_obj_arg a, b_lean_obj_arg b), \
      (a, b), ([](const auto& a, const auto& b) { return EXPRESSION; }))
#define TORCHLEAN_UNARY_SCALAR_EXPORT(NAME, EXPRESSION) \
  TORCHLEAN_BUFFER_EXPORT(NAME, (b_lean_obj_arg x, double scalar), \
      (x, scalar), ([](const auto& x, double scalar) { return EXPRESSION; }))
#define TORCHLEAN_BINARY_SCALAR_EXPORT(NAME, EXPRESSION) \
  TORCHLEAN_BUFFER_EXPORT(NAME, (b_lean_obj_arg a, b_lean_obj_arg b, double scalar), \
      (a, b, scalar), \
      ([](const auto& a, const auto& b, double scalar) { return EXPRESSION; }))
#define TORCHLEAN_VJP_EXPORT(NAME, EXPRESSION) \
  TORCHLEAN_BUFFER_EXPORT(NAME##_bwd, (b_lean_obj_arg x, b_lean_obj_arg g), \
      (x, g), ([](const auto& x, const auto& g) { return EXPRESSION; }))

#include "operations.h"

#undef TORCHLEAN_VJP_EXPORT
#undef TORCHLEAN_BINARY_SCALAR_EXPORT
#undef TORCHLEAN_UNARY_SCALAR_EXPORT
#undef TORCHLEAN_BINARY_EXPORT
#undef TORCHLEAN_UNARY_EXPORT
#undef TORCHLEAN_ARGUMENTS
#undef TORCHLEAN_BUFFER_EXPORT

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_mse_mean(
    b_lean_obj_arg input, uint64_t count) {
  return invoke([&] { return box(mse_mean(tensor(input), count)); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_mse_slope(
    b_lean_obj_arg input, uint64_t count) {
  return invoke([&] { return box(mse_slope(tensor(input), count)); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_abs_bwd(
    b_lean_obj_arg XObj, b_lean_obj_arg GObj) {
  return invoke([&]() {
    const auto x = flat(XObj);
    const auto g = flat(GObj);
    same_size(x, g, "torchlean_cuda_buffer_abs_bwd: size mismatch");
    const auto sign = aten<"where.self">(
        aten<"gt.Tensor">(x, scalar_for(x, 0.0)), scalar_for(x, 1.0),
        aten<"where.self">(aten<"lt.Tensor">(x, scalar_for(x, 0.0)), scalar_for(x, -1.0), x));
    // Context.sign: x itself at a zero or NaN, so a NaN propagates and a zero keeps its sign.
    return box(aten<"mul.Tensor">(sign, g));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_sqrt_bwd(
    b_lean_obj_arg XObj, b_lean_obj_arg GObj) {
  return invoke([&]() {
    const auto x = flat(XObj);
    const auto g = flat(GObj);
    same_size(x, g, "torchlean_cuda_buffer_sqrt_bwd: size mismatch");
    const auto positive = aten<"gt.Tensor">(x, scalar_for(x, 0.0));
    const auto root = aten<"sqrt">(aten<"where.self">(positive, x, scalar_for(x, 1.0)));
    // The rule of Op.sqrt .clampNegative rounds g / (2 root) once and multiplies a zero factor
    // where x > 0 fails, which gives -0 at a negative g and a NaN at a nonfinite one.
    return box(aten<"where.self">(
        positive, aten<"div.Tensor">(g, aten<"mul.Tensor">(root, scalar_for(root, 2.0))),
        aten<"mul.Tensor">(aten<"zeros_like">(g), g)));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_clamp(
    b_lean_obj_arg BObj, double lo, double hi) {
  return invoke([&]() {
    const auto x = flat(BObj);
    const auto lower = scalar_tensor(rounded(lo, x.scalar_type()), options(x));
    const auto upper = scalar_tensor(rounded(hi, x.scalar_type()), options(x));
    // `Context.clamp`: minimum (maximum x lo) hi with 9.6's operations, so a NaN operand or bound
    // gives a NaN and, for a number x, lo > hi gives hi.
    return box(ieee_minimum(ieee_maximum(x, lower), upper));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_clamp_bwd(
    b_lean_obj_arg XObj, b_lean_obj_arg GObj, double lo, double hi) {
  return invoke([&]() {
    const auto x = flat(XObj);
    const auto g = flat(GObj);
    same_size(x, g, "torchlean_cuda_buffer_clamp_bwd: size mismatch");
    const auto interior = aten<"logical_and">(
        aten<"gt.Tensor">(x, scalar_for(x, rounded(lo, x.scalar_type()))),
        aten<"lt.Tensor">(x, scalar_for(x, rounded(hi, x.scalar_type()))));
    // `clampDerivative x lo hi * g`: the coefficient 1 or 0 times the cotangent.
    return box(aten<"mul.Tensor">(
        aten<"where.self">(interior, aten<"ones_like">(g), aten<"zeros_like">(g)), g));
  });
}

// Op.dropout in both passes: an entry is kept iff its uniform sample u is at least p (IEEE >=, so a
// NaN p keeps none); a kept entry is one IEEE division x / (1 - p), a dropped one x * 0 (NaN at an
// infinite or NaN x, -0 at a negative one), and every entry is +0 when p >= 1. The divisor is a
// device tensor, since ATen turns a CPU scalar divisor into a product with its rounded reciprocal.
// p is rounded to the dtype first; binary32 forms 1 - p in binary64 and rounds it once more, which
// is the binary32 difference (53 >= 2 * 24 + 2 bits).
extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_dropout(
    b_lean_obj_arg XObj, b_lean_obj_arg UObj, double probability) {
  return invoke([&]() {
    const auto x = flat(XObj);
    const auto u = flat(UObj);
    same_size(x, u, "torchlean_cuda_buffer_dropout: size mismatch");
    const auto dtype = x.scalar_type();
    const double p = rounded(probability, dtype);
    if (p >= 1.0) return box(aten<"zeros_like">(x));
    const auto divisor = scalar_tensor(rounded(1.0 - p, dtype), options(x));
    return box(aten<"where.self">(aten<"ge.Tensor">(u, scalar_for(u, p)),
                                  aten<"div.Tensor">(x, divisor),
                                  aten<"mul.Tensor">(x, scalar_for(x, 0.0))));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_max_bwd(
    b_lean_obj_arg AObj, b_lean_obj_arg BObj, b_lean_obj_arg GObj) {
  return invoke([&]() {
    const auto a = flat(AObj);
    const auto b = flat(BObj);
    const auto g = flat(GObj);
    same_size(a, b, "torchlean_cuda_buffer_max_bwd: size mismatch");
    same_size(a, g, "torchlean_cuda_buffer_max_bwd: size mismatch");
    return minmax_backward(a, b, g, true);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_min_bwd(
    b_lean_obj_arg AObj, b_lean_obj_arg BObj, b_lean_obj_arg GObj) {
  return invoke([&]() {
    const auto a = flat(AObj);
    const auto b = flat(BObj);
    const auto g = flat(GObj);
    same_size(a, b, "torchlean_cuda_buffer_min_bwd: size mismatch");
    same_size(a, g, "torchlean_cuda_buffer_min_bwd: size mismatch");
    return minmax_backward(a, b, g, false);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_copy_and_release(
    b_lean_obj_arg BObj) {
  return invoke([&]() {
    // Complete allocation and enqueue the copy before retiring the source
    // through the runtime's allocator and telemetry path.
    auto result = box(aten<"clone">(flat(BObj)));
    (void)torchlean_cuda_buffer_release(BObj);
    return result;
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_adam_step(
    b_lean_obj_arg ParametersObj,
    b_lean_obj_arg GradientObj,
    b_lean_obj_arg FirstMomentObj,
    b_lean_obj_arg SecondMomentObj,
    double beta1,
    double oneMinusBeta1,
    double beta2,
    double oneMinusBeta2,
    double firstMomentDenominator,
    double secondMomentDenominator,
    double epsilon,
    b_lean_obj_arg DecayObj,
    double updateScale) {
  return invoke([&]() {
    const auto parameters = flat(ParametersObj);
    const auto gradient = flat(GradientObj);
    const auto first_moment = flat(FirstMomentObj);
    const auto second_moment = flat(SecondMomentObj);
    same_size(parameters, gradient, "torchlean_cuda_buffer_adam_step: size mismatch");
    same_size(parameters, first_moment, "torchlean_cuda_buffer_adam_step: size mismatch");
    same_size(parameters, second_moment, "torchlean_cuda_buffer_adam_step: size mismatch");

    // Preserve independent coefficients and round each in the parameter dtype.
    const auto dtype = parameters.scalar_type();
    const double beta1_f = rounded(beta1, dtype);
    const double one_minus_beta1_f = rounded(oneMinusBeta1, dtype);
    const double beta2_f = rounded(beta2, dtype);
    const double one_minus_beta2_f = rounded(oneMinusBeta2, dtype);
    const double first_correction = rounded(1.0 / firstMomentDenominator, dtype);
    const double second_correction = rounded(1.0 / secondMomentDenominator, dtype);
    const double epsilon_f = rounded(epsilon, dtype);
    const double update_scale = rounded(updateScale, dtype);

    const auto m_scaled = aten<"mul.Tensor">(first_moment, scalar_for(first_moment, beta1_f));
    auto m = axpy(m_scaled, gradient, one_minus_beta1_f);
    const auto g2 = aten<"mul.Tensor">(gradient, gradient);
    const auto v_scaled = aten<"mul.Tensor">(second_moment, scalar_for(second_moment, beta2_f));
    const auto weighted_square = aten<"mul.Tensor">(
        aten<"mul.Tensor">(gradient, scalar_for(gradient, one_minus_beta2_f)), gradient);
    auto v = aten<"where.self">(
        aten<"lt.Tensor">(g2, scalar_for(g2, std::numeric_limits<double>::infinity())),
        axpy(v_scaled, g2, one_minus_beta2_f), add(v_scaled, weighted_square));
    // Retain the ordinary host-computed reciprocal, but do not require it to fit the dtype.
    const auto correct_moment = [&](const Tensor& moment, double inverse, double divisor) {
      if (std::isfinite(inverse)) return aten<"mul.Tensor">(moment, scalar_for(moment, inverse));
      return aten<"div.Tensor">(moment, full_like(moment, rounded(divisor, dtype)));
    };
    const auto m_hat = correct_moment(m, first_correction, firstMomentDenominator);
    const auto v_hat = correct_moment(v, second_correction, secondMomentDenominator);
    // The raw IEEE root of Buffer.sqrt, not Buffer.sqrtClampNegative: a negative v_hat stays
    // NaN, where the CPU's Optimizer.correctedRMS clamps it to zero.
    auto corrected_root = aten<"sqrt">(v_hat);
    const double second_divisor = rounded(secondMomentDenominator, dtype);
    if (second_divisor > 0) {
      const auto correction_root = aten<"sqrt">(full({}, second_divisor, options(v)));
      const auto scaled_root = aten<"div.Tensor">(aten<"sqrt">(v),
          correction_root);
      corrected_root = aten<"where.self">(finite(v_hat), corrected_root, scaled_root);
    }
    const auto denominator = add(corrected_root, scalar_for(corrected_root, epsilon_f));
    // Keep a tensor denominator: ATen's scalar-divisor optimization uses a
    // rounded reciprocal followed by multiplication, which can change bits.
    const auto normalized_update = aten<"div.Tensor">(m_hat, denominator);
    // Adam passes none: even a zero coefficient turns an infinite parameter into NaN.
    const auto decayed_parameters = lean_is_scalar(DecayObj) ? parameters
        : axpy(parameters, parameters,
               rounded(lean_unbox_float(lean_ctor_get(DecayObj, 0)), dtype));
    auto updated = axpy(decayed_parameters, normalized_update, update_scale);
    // A finite weighted update can survive an unrepresentable unscaled quotient.
    // Keep the ordinary FMA above; rescale exceptional entries by exact powers of two.
    if (std::isfinite(update_scale)) {
      const double min_normal = dtype == kFloat
          ? std::numeric_limits<float>::min() : std::numeric_limits<double>::min();
      const auto quotient_magnitude = aten<"abs">(normalized_update);
      const auto quotient_finite = aten<"lt.Tensor">(
          quotient_magnitude,
          scalar_for(quotient_magnitude, std::numeric_limits<double>::infinity()));
      const auto nonzero = [](const Tensor& t) {
        return aten<"ne.Tensor">(t, scalar_for(t, std::int64_t{0}));
      };
      const auto range_loss = aten<"logical_or">(aten<"logical_not">(quotient_finite),
          aten<"logical_and">(
              aten<"lt.Tensor">(quotient_magnitude, scalar_for(quotient_magnitude, min_normal)),
              nonzero(m_hat)));
      const auto largest_input = aten<"maximum">(aten<"maximum">(aten<"abs">(decayed_parameters),
          aten<"abs">(m_hat)), aten<"abs">(denominator));
      const auto finite_inputs = aten<"logical_and">(
          aten<"lt.Tensor">(largest_input,
                            scalar_for(largest_input, std::numeric_limits<double>::infinity())),
          nonzero(denominator));
      const auto repair = aten<"logical_and">(range_loss, finite_inputs);
      if (update_scale == 0) {
        // Multiplying finite inputs first also preserves the sign of a zero update.
        const auto zero_update = aten<"div.Tensor">(
            aten<"mul.Tensor">(m_hat, scalar_for(m_hat, update_scale)), denominator);
        updated = aten<"where.self">(repair, add(decayed_parameters, zero_update), updated);
      } else {
        const auto [m_fraction, m_exponent] = aten<"frexp.Tensor">(m_hat);
        const auto [d_fraction, d_exponent] = aten<"frexp.Tensor">(denominator);
        int scale_exponent = 0;
        const double scale_fraction = std::frexp(update_scale, &scale_exponent);
        const auto quotient = aten<"div.Tensor">(m_fraction, d_fraction);
        const auto fraction = aten<"mul.Tensor">(quotient, scalar_for(quotient, scale_fraction));
        const auto difference = sub(m_exponent, d_exponent);
        const auto exponent =
            add(difference, scalar_for(difference, std::int64_t{scale_exponent}));
        const auto scaled_parameters =
            aten<"ldexp.Tensor">(decayed_parameters, aten<"neg">(exponent));
        const auto rescaled = aten<"ldexp.Tensor">(add(scaled_parameters, fraction), exponent);
        // If this scaling overflows, the parameter dwarfs the update at its own precision.
        const auto recovered = aten<"where.self">(finite(scaled_parameters),
            rescaled, decayed_parameters);
        updated = aten<"where.self">(repair, recovered, updated);
      }
    }
    return triple(std::move(updated), std::move(m), std::move(v));
  });
}

}  // namespace torchlean::elementwise

// Tensor operations

#include <optional>

// These functions evaluate TorchLean's buffer operations and selected VJPs. The
// Lean tape owns differentiation; invoke() disables native graph recording.
namespace torchlean::operators {

namespace {

using namespace torchlean;

constexpr double kNegativeInfinity = -std::numeric_limits<double>::infinity();

int64_t element_count(ints shape) {
  for (int64_t dim : shape) require(dim >= 0, "LibTorch kernels: negative dimension");
  // A zero extent empties the shape whatever its other extents are.
  if (std::find(shape.begin(), shape.end(), 0) != shape.end()) return 0;
  int64_t count = 1;
  for (int64_t dim : shape) {
    require(count <= std::numeric_limits<int64_t>::max() / dim,
            "LibTorch kernels: shape size overflow");
    count *= dim;
  }
  return count;
}

void require_size(b_lean_obj_arg object, ints shape) {
  require(tensor(object).numel() == element_count(shape),
          "LibTorch kernels: buffer size does not match shape");
}

Tensor checked(b_lean_obj_arg object, ints shape) {
  require_size(object, shape);
  return shaped(object, shape);
}

// ATen's contiguous strides multiply the extents after the first, reading a zero extent as one,
// so they can overflow for a shape with a zero extent, which has no entries. A kernel returns the
// result of such a shape before it builds a tensor of that shape.
Tensor no_entries(b_lean_obj_arg like) {
  return aten<"empty.memory_format">({0}, options(tensor(like)));
}

std::vector<int64_t> shape_array(b_lean_obj_arg object) {
  require(lean_is_array(const_cast<lean_object*>(object)),
          "LibTorch kernels: expected Array Nat");
  auto shape = dimensions(object, "LibTorch kernels: dimension exceeds UInt32");
  element_count(shape);
  return shape;
}

std::vector<int64_t> index_array(b_lean_obj_arg object, uint32_t count) {
  require(lean_is_array(const_cast<lean_object*>(object)),
          "LibTorch kernels: expected Array Nat indices");
  require(lean_array_size(object) == count,
          "LibTorch kernels: indices.size mismatch");
  return dimensions(object, "LibTorch kernels: index exceeds UInt32");
}

Tensor device_indices(const std::vector<int64_t>& values, const Tensor& like) {
  // The transfer is blocking: no GPU operation may outlive this host vector.
  return aten<"to.dtype_layout">(
      host_view(const_cast<int64_t*>(values.data()), static_cast<int64_t>(values.size()), kLong),
      options(like).dtype(kLong), false, true);
}

// The sums of segments of the rows of a rank-two input, each column of a segment folded in order
// from `initial`: a segment of rows r, r + 1, ... gives (((initial + data[r, c]) + data[r + 1, c])
// + ...) in column c, one rounding per row. CUDA segment_reduce computes this "sum" of a rank-two
// input on axis 0 (segment_reduce_forward_kernel in aten/src/ATen/native/cuda/SegmentReduce.cu,
// with a CUB reduction at rank one), but its `str reduce` and `Scalar? initial` cannot
// pass the stable dispatcher, and ATen does not promise the order. A generated kernel therefore
// folds each column of a segment in one thread. TorchLeanTest/CUDA/Numerics.lean checks the order.
Tensor ordered_segments(const Tensor& data, const std::vector<int64_t>& lengths,
                        double initial = 0.0) {
  require(data.dim() == 2, "LibTorch kernels: ordered segments need rank two");
  const auto dtype = data.scalar_type();
  require(dtype == kFloat || dtype == kDouble,
          "LibTorch kernels: ordered segments need a binary32 or binary64 input");
  std::vector<int64_t> offsets{0};
  for (const int64_t length : lengths) {
    require(length >= 0 && length <= data.size(0) - offsets.back(),
            "LibTorch kernels: segment lengths must partition the rows");
    offsets.push_back(offsets.back() + length);
  }
  require(offsets.back() == data.size(0),
          "LibTorch kernels: segment lengths must partition the rows");
  const int64_t columns = data.size(1);
  const auto x = aten<"contiguous">(data);
  const auto starts = device_indices(offsets, data);
  const auto result = aten<"empty.memory_format">(
      {static_cast<int64_t>(lengths.size()), columns}, options(data));
  const std::string type = dtype == kFloat ? "float" : "double";
  const std::string source =
      "extern \"C\" __global__ void torchlean_kernel(const " + type +
      "* data, const long long* offsets, " + type +
      "* out, double initial, unsigned long long columns, unsigned long long n) {\n"
      "  unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n"
      "  if (index >= n) return;\n"
      "  unsigned long long segment = index / columns, column = index % columns;\n"
      "  " + type + " sum = (" + type + ")initial;\n"
      "  for (long long row = offsets[segment]; row < offsets[segment + 1]; ++row)\n"
      "    sum = sum + data[row * columns + column];\n"
      "  out[index] = sum;\n"
      "}\n";
  // The initial value travels as the bits of a binary64 argument.
  uint64_t bits = 0;
  std::memcpy(&bits, &initial, sizeof bits);
  const auto n = static_cast<uint64_t>(result.numel());
  std::vector<launch> launches{{n, {reinterpret_cast<uint64_t>(x.const_data_ptr()),
                                    reinterpret_cast<uint64_t>(starts.const_data_ptr()),
                                    reinterpret_cast<uint64_t>(result.mutable_data_ptr()),
                                    bits, static_cast<uint64_t>(columns), n}}};
  launch_rows(source, launches);
  return result;
}

Tensor ordered_columns(const Tensor& matrix) {
  return aten<"reshape">(ordered_segments(matrix, {matrix.size(0)}), {matrix.size(1)});
}

Tensor column_sum(const Tensor& matrix, bool ordered) {
  return ordered ? ordered_columns(matrix) : aten<"sum.dim_IntList">(matrix, {0});
}

Tensor max_ignoring_nan(const Tensor& input, int64_t axis, bool keepdim = false) {
  return aten<"amax">(aten<"masked_fill.Tensor">(input, aten<"isnan">(input),
                                                 scalar_for(input, kNegativeInfinity)),
                      {axis}, keepdim);
}

Tensor log_normalizer(const Tensor& shifted, const Tensor& pivot, int64_t axis) {
  // scatter.src with zeros writes what scatter.value with 0 writes, whose Scalar the stable
  // dispatcher cannot pass. Each row has one pivot.
  const auto others = aten<"scatter.src">(aten<"exp">(shifted), axis, pivot,
                                          aten<"zeros">(pivot.sizes(), options(shifted)));
  return aten<"log1p">(aten<"sum.dim_IntList">(others, {axis}, true));
}

Tensor cross_entropy_normalizer(const Tensor& x, const Tensor& shifted,
                                const Tensor& pivot, int64_t axis) {
  const auto zeros = aten<"zeros">(pivot.sizes(), options(x));
  const auto tail = aten<"sum.dim_IntList">(
      aten<"scatter.src">(aten<"exp">(shifted), axis, pivot, zeros), {axis}, true);
  const auto normalizer = aten<"log1p">(tail);
  if (x.size(axis) <= 1) return normalizer;

  // Individual tail exponentials may round to zero even when their combined mass is
  // representable. This uses the formula and the branch tests of CrossEntropy.twoLevelAt, not
  // its bits: ATen sums in its own order, half and bfloat16 in float32, and its exp and log1p are
  // not correctly rounded. Unlike CrossEntropy.log1pSum it does not rescale the inner sum, so a
  // row where log1p of that sum is not finite takes the one-level normalizer.
  const auto without_pivot = aten<"scatter.src">(
      x, axis, pivot, add(zeros, scalar_for(x, kNegativeInfinity)));
  const auto runner_up = aten<"argmax">(without_pivot, axis, true);
  const auto runner_value = aten<"gather">(x, axis, runner_up);
  const auto largest = aten<"gather">(x, axis, pivot);
  const auto inner = aten<"scatter.src">(
      aten<"scatter.src">(aten<"exp">(sub(x, runner_value)), axis, pivot, zeros),
      axis, runner_up, zeros);
  const auto others = aten<"exp">(add(sub(runner_value, largest),
      aten<"log1p">(aten<"sum.dim_IntList">(inner, {axis}, true))));
  const auto two_level = aten<"log1p">(others);
  const auto finite_result = aten<"logical_not">(aten<"any.dim">(
      aten<"logical_not">(aten<"isfinite">(sub(shifted, two_level))), axis, true));
  const auto use_two_level = aten<"logical_and">(
      aten<"ne.Tensor">(runner_up, pivot), aten<"logical_and">(
      aten<"logical_not">(aten<"isinf">(largest)), aten<"logical_and">(
      aten<"logical_not">(aten<"isinf">(runner_value)), aten<"logical_and">(
      aten<"eq.Tensor">(tail, zeros), aten<"logical_and">(
      aten<"isfinite">(others), finite_result)))));
  return aten<"where.self">(use_two_level, two_level, normalizer);
}

Tensor weighted_log_softmax(const Tensor& x, const Tensor& weights, int64_t axis) {
  const auto pivot = aten<"argmax">(x, axis, true);
  const auto largest = aten<"gather">(x, axis, pivot);
  const auto shifted = sub(x, largest);
  const auto normalizer = cross_entropy_normalizer(x, shifted, pivot, axis);
  const auto logp = sub(shifted, normalizer);
  const auto finite = aten<"isfinite">(logp);
  const auto direct = aten<"mul.Tensor">(
      weights, aten<"where.self">(finite, logp, aten<"zeros_like">(logp)));
  const auto difference = sub(largest, x);
  const auto finite_difference = aten<"isfinite">(difference);
  // Weight opposite-sign finite operands before an overflowing subtraction. This also avoids
  // halving a small derivative contribution. Nonfinite original inputs still propagate NaN.
  const auto gap = aten<"where.self">(finite_difference,
      aten<"mul.Tensor">(weights, aten<"where.self">(finite_difference, difference,
                                                     aten<"zeros_like">(difference))),
      sub(aten<"mul.Tensor">(weights, largest), aten<"mul.Tensor">(weights, x)));
  return aten<"where.self">(finite, direct,
                            aten<"neg">(add(aten<"mul.Tensor">(weights, normalizer), gap)));
}

struct BroadcastShape {
  std::vector<int64_t> input;
  std::vector<int64_t> output;
  std::vector<int64_t> map;
  std::vector<int64_t> input_permutation;
  std::vector<int64_t> expanded_input;
};

BroadcastShape broadcast_shape(b_lean_obj_arg input, b_lean_obj_arg output,
                               b_lean_obj_arg map) {
  BroadcastShape result;
  result.input = shape_array(input);
  result.output = shape_array(output);
  require(lean_is_array(const_cast<lean_object*>(map)),
          "LibTorch kernels: expected Array Nat axis map");
  result.map = dimensions(map, "LibTorch kernels: axis map exceeds UInt32");
  require(result.map.size() == result.output.size(),
          "LibTorch kernels: axis map rank mismatch");
  std::vector<bool> seen(result.input.size(), false);
  for (size_t axis = 0; axis < result.output.size(); ++axis) {
    const int64_t mapped = result.map[axis];
    require(mapped <= static_cast<int64_t>(result.input.size()),
            "LibTorch kernels: broadcast axis out of range");
    if (mapped == 0) {
      result.expanded_input.push_back(1);
      continue;
    }
    const int64_t source = mapped - 1;
    require(!seen[source], "LibTorch kernels: repeated broadcast input axis");
    seen[source] = true;
    const int64_t dim = result.input[source];
    require(dim == 1 || dim == result.output[axis],
            "LibTorch kernels: incompatible broadcast dimension");
    result.input_permutation.push_back(source);
    result.expanded_input.push_back(dim);
  }
  require(std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }),
          "LibTorch kernels: broadcast omits an input axis");
  return result;
}

Tensor gather_rows(const Tensor& input, const std::vector<int64_t>& indices) {
  const int64_t count = static_cast<int64_t>(indices.size());
  if (input.size(0) == 0 || input.size(1) == 0 || count == 0) {
    return aten<"zeros">({count, input.size(1)}, options(input));
  }
  const auto index = device_indices(indices, input);
  const auto valid = aten<"lt.Tensor">(index, scalar_for(index, input.size(0)));
  const auto gathered = aten<"index_select">(
      input, 0, aten<"minimum">(index, scalar_for(index, input.size(0) - 1)));
  // Multiplication by zero would leak NaNs from the clamped source row.
  return aten<"masked_fill.Tensor">(gathered, aten<"unsqueeze">(aten<"logical_not">(valid), 1),
                                    scalar_for(gathered, std::int64_t{0}));
}

Tensor scatter_rows(const Tensor& base, const Tensor& values,
                    const std::vector<int64_t>& indices) {
  auto out = aten<"clone">(base);
  if (base.numel() == 0 || indices.empty()) return out;
  std::vector<int64_t> source;
  source.reserve(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] < base.size(0)) source.push_back(static_cast<int64_t>(i));
  }
  if (source.empty()) return out;

  if (!unstable::deterministic()) {
    std::vector<int64_t> destination;
    destination.reserve(source.size());
    for (int64_t i : source) destination.push_back(indices[i]);
    // index_add_ with alpha 1, whose Scalar the stable dispatcher cannot pass, as scatter_add_ of
    // whole rows: both add each update into its row with atomic additions.
    const auto updates = aten<"index_select">(values, 0, device_indices(source, values));
    aten<"scatter_add_">(
        out, 0, aten<"expand_as">(aten<"unsqueeze">(device_indices(destination, base), 1), updates),
        updates);
    return out;
  }

  // The indices already live on the host. A stable sort groups updates while
  // preserving their original order; only touched rows enter the reduction.
  std::stable_sort(source.begin(), source.end(),
                   [&](int64_t a, int64_t b) { return indices[a] < indices[b]; });
  std::vector<int64_t> touched;
  std::vector<int64_t> lengths;
  for (int64_t i : source) {
    if (touched.empty() || touched.back() != indices[i]) {
      touched.push_back(indices[i]);
      lengths.push_back(1);  // Each segment begins with its base value.
    }
    ++lengths.back();
  }
  std::vector<int64_t> order;
  order.reserve(source.size() + touched.size());
  int64_t offset = 0;
  for (size_t group = 0; group < touched.size(); ++group) {
    order.push_back(static_cast<int64_t>(group));
    for (int64_t j = 1; j < lengths[group]; ++j) {
      order.push_back(static_cast<int64_t>(touched.size()) + offset++);
    }
  }
  auto destination = device_indices(touched, base);
  auto grouped = aten<"cat">(
      {aten<"index_select">(base, 0, destination),
       aten<"index_select">(values, 0, device_indices(source, values))}, 0);
  grouped = aten<"index_select">(grouped, 0, device_indices(order, base));
  // -0 + base preserves either sign of zero. Starting from +0 would change a
  // negative-zero base before the first update. Untouched rows remain cloned.
  auto reduced = ordered_segments(grouped, lengths, -0.0);
  aten<"index_copy_">(out, 0, destination, reduced);
  return out;
}

// Mutate only freshly allocated spectra, never a borrowed Lean buffer.
void real_endpoints(Tensor& spectrum, int64_t length, int64_t axis) {
  const auto imaginary = aten<"imag">(spectrum);
  aten<"zero_">(aten<"select.int">(imaginary, axis, 0));
  if (length % 2 == 0) aten<"zero_">(aten<"select.int">(imaginary, axis, length / 2));
}

Tensor inverse_real_fft(b_lean_obj_arg input, uint32_t batch, uint32_t n, bool normalized) {
  require(n > 0, "irfft1dPacked: length must be positive");
  auto packed = checked(input, {batch, n / 2 + 1, 2});
  element_count({batch, n});
  if (batch == 0) return aten<"empty.memory_format">({0}, options(packed));
  auto spectrum = aten<"clone">(aten<"view_as_complex">(aten<"contiguous">(packed)));
  real_endpoints(spectrum, n, 1);
  // Explicit n distinguishes odd/even lengths with the same half-spectrum shape.
  // The unnormalized transform avoids scaling large spectra up or tiny outputs down.
  // fft_irfft(spectrum, n, 1, norm) with norm "backward" or "forward", whose `str? norm` the
  // stable dispatcher cannot pass: on a spectrum of n / 2 + 1 entries it is _fft_c2r with
  // normalization 2 (divide by n) or 0 (none).
  return aten<"_fft_c2r">(spectrum, {1}, normalized ? 2 : 0, n);
}

// Evaluate h[t] = a[t] h[t-1] + b[t] left to right with one rounded product and one rounded sum
// per step, the order of TorchLean.StateSpace.scanStates and of its adjoint recurrence.
Tensor affine_scan(const Tensor& a, const Tensor& b, const Tensor& initial) {
  const int64_t steps = b.size(0);
  if (steps == 0 || b.size(1) == 0) return aten<"empty_like">(b);
  auto result = aten<"empty_like">(b);
  auto state = initial;
  for (int64_t t = 0; t < steps; ++t) {
    state = add(aten<"mul.Tensor">(aten<"select.int">(a, 0, t), state),
                aten<"select.int">(b, 0, t));
    aten<"copy_">(aten<"select.int">(result, 0, t), state);
  }
  return result;
}

struct ScanInputs {
  Tensor a;
  Tensor b;
  Tensor x;
  Tensor initial;
};

ScanInputs scan_inputs(b_lean_obj_arg a, b_lean_obj_arg b, b_lean_obj_arg x,
                       b_lean_obj_arg initial, uint32_t steps, uint32_t state,
                       bool variable) {
  auto coefficient_a = variable ? checked(a, {steps, state}) : checked(a, {state});
  auto coefficient_b = variable ? checked(b, {steps, state}) : checked(b, {state});
  if (!variable) {
    coefficient_a = aten<"expand">(aten<"unsqueeze">(coefficient_a, 0), {steps, state});
    coefficient_b = aten<"expand">(aten<"unsqueeze">(coefficient_b, 0), {steps, state});
  }
  return {coefficient_a, coefficient_b, checked(x, {steps, state}),
          checked(initial, {state})};
}

lean_obj_res scan_backward(const ScanInputs& inputs, const Tensor& output,
                           const Tensor& dy, bool variable) {
  const int64_t steps = inputs.x.size(0);
  const int64_t state = inputs.x.size(1);
  if (steps == 0 || state == 0) {
    auto parameter_shape = variable ? std::vector<int64_t>{steps, state}
                                    : std::vector<int64_t>{state};
    return quadruple(aten<"zeros">(parameter_shape, options(inputs.x)),
                     aten<"zeros">(parameter_shape, options(inputs.x)),
                     aten<"zeros_like">(inputs.x),
                     aten<"zeros_like">(inputs.initial));
  }
  // Reverse recurrence: g[t] = dy[t] + a[t+1] * g[t+1].
  // The final step adds (+0) * (-0) = -0, the identity of rounded addition, so g[T-1] = dy[T-1]
  // exactly as in StateSpace.scanAdjoints, without reading a coefficient beyond the sequence.
  auto shifted = aten<"cat">(
      {aten<"narrow">(inputs.a, 0, 1, steps - 1),
       aten<"zeros">({1, state}, options(inputs.a))}, 0);
  auto g = aten<"flip">(affine_scan(aten<"flip">(shifted, {0}), aten<"flip">(dy, {0}),
                                    full_like(inputs.initial, -0.0)), {0});
  auto previous = aten<"cat">(
      {aten<"unsqueeze">(inputs.initial, 0), aten<"narrow">(output, 0, 0, steps - 1)}, 0);
  auto da = aten<"mul.Tensor">(g, previous);
  auto db = aten<"mul.Tensor">(g, inputs.x);
  if (!variable) {
    // Shared coefficients add their cotangents from +0 in increasing t, the order of
    // TorchLean.Tensor.scanCoefficientGrad.
    da = ordered_columns(da);
    db = ordered_columns(db);
  }
  return quadruple(std::move(da), std::move(db), aten<"mul.Tensor">(g, inputs.b),
                   aten<"mul.Tensor">(aten<"select.int">(g, 0, 0),
                                      aten<"select.int">(inputs.a, 0, 0)));
}

}  // namespace

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_sum_by_row(
    b_lean_obj_arg input, uint32_t rows, uint32_t cols) {
  return invoke([&] { return box(aten<"sum.dim_IntList">(checked(input, {rows, cols}), {1})); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_max_by_row(
    b_lean_obj_arg input, uint32_t rows, uint32_t cols) {
  return invoke([&] {
    auto x = checked(input, {rows, cols});
    return box(rows == 0 || cols == 0 ? aten<"zeros">({rows}, options(x))
                                     : max_ignoring_nan(x, 1));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_argmax_mask_by_row(
    b_lean_obj_arg input, uint32_t rows, uint32_t cols) {
  return invoke([&] {
    const auto x = checked(input, {rows, cols});
    auto mask = aten<"zeros_like">(x);
    if (rows != 0 && cols != 0) {
      // scatter_.src with ones, as scatter_.value with 1 (see log_normalizer).
      const auto pivot = aten<"argmax">(x, 1, true);
      aten<"scatter_.src">(mask, 1, pivot, aten<"ones">(pivot.sizes(), options(x)));
    }
    return box(std::move(mask));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_distill(
    b_lean_obj_arg student, b_lean_obj_arg teacher, b_lean_obj_arg dimensions,
    uint32_t axis, uint8_t gradient) {
  return invoke([&] {
    const auto shape = shape_array(dimensions);
    require(axis < shape.size(), "distill: class axis out of range");
    require_size(student, shape);
    require_size(teacher, shape);
    require(tensor(student).scalar_type() == tensor(teacher).scalar_type(),
            "distill: student/teacher dtype mismatch");
    // The empty loss is the negated empty sum, `-0`, and the empty gradient is empty.
    if (element_count(shape) == 0) {
      return box(gradient ? no_entries(student)
                          : aten<"neg">(aten<"zeros">({}, options(tensor(student)))));
    }
    const auto x = shaped(student, shape);
    const auto t = shaped(teacher, shape);
    const auto samples = x.numel() / shape[axis];
    const auto p = aten<"softmax.int">(t, axis, t.scalar_type());
    if (!gradient) {
      const auto pivot = aten<"argmax">(x, axis, true);
      const auto shifted = sub(x, aten<"gather">(x, axis, pivot));
      const auto logq = sub(shifted, log_normalizer(shifted, pivot, axis));
      // A zero teacher weight must not multiply an infinite student log-probability.
      const auto safe = aten<"where.self">(aten<"eq.Tensor">(p, scalar_for(p, std::int64_t{0})),
                                           aten<"zeros_like">(logq), logq);
      return box(divide_by_count(aten<"neg">(aten<"sum">(aten<"mul.Tensor">(p, safe))), samples));
    }
    const auto q = aten<"softmax.int">(x, axis, x.scalar_type());
    const auto delta = sub(x, t);
    const auto centered =
        sub(delta, aten<"sum.dim_IntList">(aten<"mul.Tensor">(p, delta), {axis}, true));
    const auto ratio = sub(centered, aten<"log1p">(aten<"sum.dim_IntList">(
        aten<"mul.Tensor">(p, aten<"expm1">(centered)), {axis}, true)));
    const auto close = aten<"lt.Tensor">(aten<"abs">(ratio), scalar_for(ratio, std::int64_t{1}));
    // Mask before expm1 as well: the inactive near-field branch must not overflow.
    const auto safe = aten<"where.self">(close, ratio, aten<"zeros_like">(ratio));
    const auto difference = aten<"where.self">(
        close, aten<"mul.Tensor">(p, aten<"expm1">(safe)), sub(q, p));
    return box(divide_by_count(
        sub(difference,
            aten<"mul.Tensor">(q, aten<"sum.dim_IntList">(difference, {axis}, true))),
        samples));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_cross_entropy(
    b_lean_obj_arg logits, b_lean_obj_arg target, b_lean_obj_arg dimensions,
    uint32_t axis, uint8_t mean) {
  return invoke([&] {
    const auto shape = shape_array(dimensions);
    require(axis < shape.size(), "cross_entropy: class axis out of range");
    require_size(logits, shape);
    require_size(target, shape);
    require(tensor(logits).scalar_type() == tensor(target).scalar_type(),
            "cross_entropy: dtype mismatch");
    // The empty loss is the negated empty sum, `-0`.
    if (element_count(shape) == 0) {
      return box(aten<"neg">(aten<"zeros">({}, options(tensor(logits)))));
    }
    const auto x = shaped(logits, shape);
    const auto weights = shaped(target, shape);
    const auto total = aten<"neg">(aten<"sum">(weighted_log_softmax(x, weights, axis)));
    return box(mean ? divide_by_count(total, x.numel() / shape[axis]) : total);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_cross_entropy_bwd(
    b_lean_obj_arg logits, b_lean_obj_arg target, b_lean_obj_arg seed,
    b_lean_obj_arg dimensions, uint32_t axis, uint8_t mean) {
  return invoke([&] {
    const auto shape = shape_array(dimensions);
    require(axis < shape.size(), "cross_entropy: class axis out of range");
    require_size(logits, shape);
    require_size(target, shape);
    const auto g = checked(seed, {});
    require(tensor(logits).scalar_type() == tensor(target).scalar_type() &&
                tensor(logits).scalar_type() == g.scalar_type(),
            "cross_entropy: dtype mismatch");
    if (element_count(shape) == 0) return pair(no_entries(logits), no_entries(target));
    const auto x = shaped(logits, shape);
    const auto weights = shaped(target, shape);
    const auto pivot = aten<"argmax">(x, axis, true);
    const auto shifted = sub(x, aten<"gather">(x, axis, pivot));
    const auto logp = sub(shifted, cross_entropy_normalizer(x, shifted, pivot, axis));
    const auto probabilities = aten<"exp">(logp);
    const auto upstream = aten<"neg">(weights);
    // scatter.src with zeros, as scatter.value with 0 (see log_normalizer).
    const auto others = aten<"scatter.src">(upstream, axis, pivot,
                                            aten<"zeros">(pivot.sizes(), options(upstream)));
    const auto regular = sub(upstream, aten<"mul.Tensor">(
        probabilities, aten<"sum.dim_IntList">(upstream, {axis}, true)));
    const auto diagonal = sub(
        aten<"mul.Tensor">(aten<"neg">(aten<"expm1">(aten<"gather">(logp, axis, pivot))),
                           aten<"gather">(upstream, axis, pivot)),
        aten<"mul.Tensor">(aten<"gather">(probabilities, axis, pivot),
                           aten<"sum.dim_IntList">(others, {axis}, true)));
    // Each entry divided once by the number of samples when `mean`, as the spec's pullbacks.
    const auto scattered = aten<"scatter.src">(regular, axis, pivot, diagonal);
    const auto targets = aten<"neg">(weighted_log_softmax(x, aten<"expand_as">(g, x), axis));
    const auto samples = x.numel() / shape[axis];
    const auto dx =
        aten<"mul.Tensor">(mean ? divide_by_count(scattered, samples) : scattered, g);
    const auto dt = mean ? divide_by_count(targets, samples) : targets;
    return pair(dx, dt);
  });
}

// One for the entries of each row that fewer than k entries of the row beat, zero elsewhere
// (`MoE.kept`): entry j beats entry i when x_j > x_i, or when neither is larger and j < i.
// Without NaN that keeps min(k, cols) entries. Comparisons and integer counts only, so every
// device and run gives the same mask.
extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_top_k_mask_by_row(
    b_lean_obj_arg input, uint32_t rows, uint32_t cols, uint32_t k) {
  return invoke([&] {
    auto x = checked(input, {rows, cols});
    if (rows == 0 || cols == 0) return box(aten<"zeros_like">(x));
    // Entry [r, i, j] compares entry j of row r with entry i.
    const auto xi = aten<"unsqueeze">(x, 2);
    const auto xj = aten<"unsqueeze">(x, 1);
    const auto index = arange(static_cast<int64_t>(cols), options(x).dtype(kLong));
    const auto earlier =
        aten<"lt.Tensor">(aten<"unsqueeze">(index, 0), aten<"unsqueeze">(index, 1));
    const auto beats = aten<"logical_or">(
        aten<"gt.Tensor">(xj, xi),
        aten<"logical_and">(earlier, aten<"logical_not">(aten<"gt.Tensor">(xi, xj))));
    const auto rank = aten<"sum.dim_IntList">(beats, {2});
    return box(aten<"to.dtype">(
        aten<"lt.Tensor">(rank, scalar_for(rank, static_cast<int64_t>(k))), x.scalar_type()));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_hard_masked_softmax_by_row(
    b_lean_obj_arg scores, b_lean_obj_arg mask, uint32_t rows, uint32_t cols) {
  return invoke([&] {
    const auto x = checked(scores, {rows, cols});
    const auto m = checked(mask, {rows, cols});
    const auto allowed = aten<"ne.Tensor">(m, scalar_for(m, std::int64_t{0}));
    if (rows == 0 || cols == 0) return box(aten<"empty_like">(x));
    const auto masked = aten<"masked_fill.Tensor">(x, aten<"logical_not">(allowed),
                                                   scalar_for(x, kNegativeInfinity));
    // Only the mask determines whether a row is empty; allowed nonfinite scores
    // must retain their softmax arithmetic rather than being silently zeroed.
    const auto empty = aten<"logical_not">(aten<"any.dim">(allowed, 1, true));
    const auto probabilities = aten<"softmax.int">(
        aten<"masked_fill.Tensor">(masked, empty, scalar_for(masked, std::int64_t{0})), 1);
    return box(aten<"masked_fill.Tensor">(
        probabilities, aten<"logical_or">(aten<"logical_not">(allowed), empty),
        scalar_for(probabilities, std::int64_t{0})));
  });
}

// Whether two `FloatFormat` descriptors are equal. A descriptor holds `expWidth`, `fracWidth` and
// `exponentBias` as `Nat` objects and then its `encoding` byte. Its proofs are erased.
static bool same_format(b_lean_obj_arg x, b_lean_obj_arg y) {
  const unsigned fields = lean_ctor_num_objs(x);
  if (fields != lean_ctor_num_objs(y)) return false;
  for (unsigned i = 0; i < fields; ++i)
    if (!lean_nat_eq(lean_ctor_get(x, i), lean_ctor_get(y, i))) return false;
  const unsigned encoding = fields * sizeof(void*);
  return lean_ctor_get_uint8(x, encoding) == lean_ctor_get_uint8(y, encoding);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_concat1d(
    b_lean_obj_arg first, b_lean_obj_arg second, uint32_t n, uint32_t m) {
  return invoke([&] {
    auto* a = torchlean_cuda_buffer_unbox(first);
    auto* b = torchlean_cuda_buffer_unbox(second);
    if (a->format || b->format) {
      require(a->format && b->format && same_format(a->format, b->format) &&
                a->width == b->width && a->tensor && b->tensor && a->size == n && b->size == m,
              "concat: incompatible encoded buffers");
      return encoded(aten<"cat">({*a->tensor, *b->tensor}), a->format, a->width);
    }
    return box(aten<"cat">({checked(first, {n}), checked(second, {m})}));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_slice1d(
    b_lean_obj_arg input, uint32_t n, uint32_t start, uint32_t length) {
  return invoke([&] {
    auto* buffer = torchlean_cuda_buffer_unbox(input);
    if (buffer->format) {
      require(buffer->tensor && buffer->size == n && start <= n && length <= n - start,
                "slice: invalid encoded buffer or extent");
      return encoded(aten<"clone">(aten<"narrow">(*buffer->tensor, 0,
                                                  static_cast<int64_t>(start * buffer->width),
                                                  static_cast<int64_t>(length * buffer->width))),
                     buffer->format, buffer->width);
    }
    auto x = checked(input, {n});
    require(start <= n && length <= n - start, "slice1d: slice out of bounds");
    return box(aten<"clone">(aten<"narrow">(x, 0, start, length)));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_broadcast_vec_to_cols(
    b_lean_obj_arg input, uint32_t rows, uint32_t cols) {
  return invoke([&] {
    element_count({rows, cols});
    return box(aten<"clone">(
        aten<"expand">(aten<"unsqueeze">(checked(input, {rows}), 1), {rows, cols})));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_div_count(
    b_lean_obj_arg input, uint64_t count) {
  return invoke([&] { return box(divide_by_count(tensor(input), count)); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_layer_norm_fwd(
    b_lean_obj_arg input, b_lean_obj_arg gamma, b_lean_obj_arg beta,
    uint32_t rows, uint32_t cols, double epsilon) {
  return invoke([&] {
    require(rows > 0 && cols > 0, "layerNormFwd: dimensions must be positive");
    auto x = aten<"to.dtype">(checked(input, {rows, cols}), kDouble);
    auto weight = checked(gamma, {cols});
    auto bias = checked(beta, {cols});
    // Ordinary float32 native_layer_norm loses the double-centered intermediate
    // required by TorchLean's large-common-offset regression.
    // A device tensor divisor also avoids ATen's CPU-scalar division shortcut,
    // which multiplies by a rounded reciprocal instead of dividing the sum.
    auto divisor = full({1, 1}, static_cast<double>(cols), options(x));
    auto centered = sub(x, aten<"div.Tensor">(aten<"sum.dim_IntList">(x, {1}, true), divisor));
    // Normalization.normalizedRows: each centered row is divided by Tensor.normalizationScale,
    // the IEEE maximum of its largest magnitude and sqrt |epsilon|, before it is squared. The
    // variance of the scaled row centers it again, and its maximum with 0 keeps a NaN.
    auto regularizer = full({1, 1}, epsilon, options(x));
    auto scale = aten<"maximum">(aten<"amax">(aten<"abs">(centered), {1}, true),
                                 aten<"sqrt">(aten<"abs">(regularizer)));
    auto scaled = aten<"div.Tensor">(centered, scale);
    auto deviation =
        sub(scaled, aten<"div.Tensor">(aten<"sum.dim_IntList">(scaled, {1}, true), divisor));
    auto variance = add(
        aten<"div.Tensor">(aten<"sum.dim_IntList">(aten<"square">(deviation), {1}, true), divisor),
        aten<"div.Tensor">(aten<"div.Tensor">(regularizer, scale), scale));
    auto standard_deviation = aten<"sqrt">(
        aten<"maximum">(variance, scalar_for(variance, std::int64_t{0})));
    auto normalized =
        aten<"to.dtype">(aten<"div.Tensor">(scaled, standard_deviation), weight.scalar_type());
    auto inverse = aten<"to.dtype">(
        aten<"div.Tensor">(aten<"reciprocal">(standard_deviation), scale), weight.scalar_type());
    return triple(add(aten<"mul.Tensor">(normalized, weight), bias), normalized, inverse);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_layer_norm_bwd(
    b_lean_obj_arg dout, b_lean_obj_arg normalized, b_lean_obj_arg invstd,
    b_lean_obj_arg gamma, uint32_t rows, uint32_t cols) {
  return invoke([&] {
    require(rows > 0 && cols > 0, "layerNormBwd: dimensions must be positive");
    auto dy = checked(dout, {rows, cols});
    auto xhat = checked(normalized, {rows, cols});
    auto inverse = checked(invstd, {rows, 1});
    auto weight = checked(gamma, {cols});
    auto dxhat = aten<"mul.Tensor">(dy, weight);
    // LayerNorm.backward on the saved xhat and rstd (reconstructing native_layer_norm inputs
    // would change this VJP): rstd * ((dxhat - mean dxhat) - xhat * mean (dxhat * xhat)), each
    // mean a row sum divided once by the column count.
    auto mean = divide_by_count(aten<"sum.dim_IntList">(dxhat, {1}, true), cols);
    auto mean_xhat = divide_by_count(
        aten<"sum.dim_IntList">(aten<"mul.Tensor">(dxhat, xhat), {1}, true), cols);
    auto dx = aten<"mul.Tensor">(
        inverse, sub(sub(dxhat, mean), aten<"mul.Tensor">(xhat, mean_xhat)));
    return triple(std::move(dx), aten<"sum.dim_IntList">(aten<"mul.Tensor">(dy, xhat), {0}),
                  aten<"sum.dim_IntList">(dy, {0}));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_bmm_with_transpose(
    b_lean_obj_arg first, b_lean_obj_arg second, uint32_t batch,
    uint32_t m, uint32_t n, uint32_t p, uint32_t transposeA, uint32_t transposeB) {
  return invoke([&] {
    require(transposeA <= 1 && transposeB <= 1, "bmm: transpose flag must be zero or one");
    require_size(first, {batch, m, n});
    require_size(second, {batch, n, p});
    element_count({batch, m, p});
    if (batch == 0) return box(no_entries(first));
    auto a = transposeA ? aten<"transpose.int">(shaped(first, {batch, n, m}), 1, 2)
                        : shaped(first, {batch, m, n});
    auto b = transposeB ? aten<"transpose.int">(shaped(second, {batch, p, n}), 1, 2)
                        : shaped(second, {batch, n, p});
    return box(aten<"bmm">(a, b));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_scaled_bmm(
    b_lean_obj_arg first, b_lean_obj_arg second, uint32_t batch,
    uint32_t m, uint32_t n, uint32_t p, uint32_t transposeA, uint32_t transposeB,
    double scale) {
  return invoke([&] {
    require(transposeA <= 1 && transposeB <= 1, "scaledBmm: invalid transpose flag");
    require_size(first, {batch, m, n});
    require_size(second, {batch, n, p});
    element_count({batch, m, p});
    require(tensor(first).scalar_type() == tensor(second).scalar_type(),
            "scaledBmm: dtype mismatch");
    if (batch == 0) {
      // An empty batch has no product to replay.
      auto* result = lean_alloc_ctor(0, 2, 0);
      lean_ctor_set(result, 0, box(no_entries(first)));
      lean_ctor_set(result, 1, lean_box(false));
      return result;
    }
    auto a = transposeA ? aten<"transpose.int">(shaped(first, {batch, n, m}), 1, 2)
                        : shaped(first, {batch, m, n});
    auto b = transposeB ? aten<"transpose.int">(shaped(second, {batch, p, n}), 1, 2)
                        : shaped(second, {batch, n, p});
    const auto minimum = a.scalar_type() == kFloat
        ? static_cast<double>(std::numeric_limits<float>::min())
        : std::numeric_limits<double>::min();
    const auto maximum = a.scalar_type() == kFloat
        ? static_cast<double>(std::numeric_limits<float>::max())
        : std::numeric_limits<double>::max();
    const auto nonzero = [](const Tensor& t) {
      return aten<"ne.Tensor">(t, scalar_for(t, std::int64_t{0}));
    };
    // Whether |t| is below the least normal magnitude of the dtype.
    const auto tiny = [&](const Tensor& t) {
      return aten<"lt.Tensor">(aten<"abs">(t), scalar_for(t, minimum));
    };
    auto risky = aten<"zeros">({}, options(a).dtype(kBool));
    if (a.numel() != 0 && b.numel() != 0) {
      const auto abs_a = aten<"abs">(a);
      const auto abs_b = aten<"abs">(b);
      const auto least_a = aten<"amin">(aten<"where.self">(nonzero(abs_a), abs_a,
          full_like(abs_a, std::numeric_limits<double>::infinity())));
      const auto least_b = aten<"amin">(aten<"where.self">(nonzero(abs_b), abs_b,
          full_like(abs_b, std::numeric_limits<double>::infinity())));
      // Conservative bounds may request replay for products from different contraction rows.
      // They also catch products rounded to zero before a growing scale can recover them.
      const auto least = aten<"mul.Tensor">(least_a, least_b);
      const auto most = aten<"mul.Tensor">(aten<"amax">(abs_a), aten<"amax">(abs_b));
      risky = aten<"logical_or">(aten<"lt.Tensor">(least, scalar_for(least, minimum)),
                                 aten<"gt.Tensor">(most, scalar_for(most, maximum)));
    }
    // A zero original row or column has no nonzero finite product to recover. Padding must not
    // alone force a host replay of the matrix; nonzero orthogonal rows remain conservative.
    const auto contributes = aten<"logical_and">(aten<"any.dim">(nonzero(a), 2, true),
                                                 aten<"any.dim">(nonzero(b), 1, true));
    std::optional<Tensor> value;
    if (scale != 0.0 && std::abs(scale) < 1.0) {
      // A power of two near sqrt(|scale|) on the left and the remainder on the right: no rounded
      // square root, and no rounding for a power-of-two scale while the factors and the scaled
      // operands are representable in the dtype.
      int exponent = 0;
      std::frexp(scale, &exponent);
      const auto power = std::ldexp(1.0, exponent / 2);
      const auto left = aten<"mul.Tensor">(a, scalar_for(a, power));
      const auto right = aten<"mul.Tensor">(b, scalar_for(b, scale / power));
      // Zero is included: a nonzero original operand can disappear before the contraction.
      risky = aten<"logical_or">(risky, aten<"logical_or">(
          aten<"any">(aten<"logical_and">(nonzero(a), tiny(left))),
          aten<"any">(aten<"logical_and">(nonzero(b), tiny(right)))));
      value = aten<"bmm">(left, right);
    } else {
      const auto product = aten<"bmm">(a, b);
      // Cancellation can leave a tiny or zero sum even when every constituent product is normal.
      // A growing scale must see that original contraction, not its already rounded result.
      risky = aten<"logical_or">(risky, aten<"any">(aten<"logical_or">(
          aten<"logical_not">(aten<"isfinite">(product)),
          aten<"logical_and">(contributes, tiny(product)))));
      value = scale == 1.0 ? product : aten<"mul.Tensor">(product, scalar_for(product, scale));
    }
    risky = aten<"logical_or">(risky, aten<"any">(aten<"logical_or">(
        aten<"logical_not">(aten<"isfinite">(*value)),
        aten<"logical_and">(contributes, tiny(*value)))));
    // Keep the native buffer itself for nonfinite originals. Downloading a NaN through a
    // widened host FloatArray could change its payload even when no coefficient can be replayed.
    risky = aten<"logical_and">(risky, aten<"logical_and">(aten<"all">(aten<"isfinite">(a)),
                                                           aten<"all">(aten<"isfinite">(b))));
    // One scalar read gates host recovery. The conservative guard also includes exact zeros.
    const bool replay = std::isfinite(scale) && item<bool>(risky);
    auto* result = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(result, 0, box(std::move(*value)));
    lean_ctor_set(result, 1, lean_box(replay));
    return result;
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_rfft1d_packed(
    b_lean_obj_arg input, uint32_t batch, uint32_t n) {
  return invoke([&] {
    require(n > 0, "rfft1dPacked: length must be positive");
    auto x = checked(input, {batch, n});
    element_count({batch, n / 2 + 1, 2});
    if (batch == 0) return box(aten<"empty.memory_format">({0}, options(x)));
    // norm "backward" is fft_rfft's default None, the `str? norm` the stable dispatcher passes.
    return box(aten<"view_as_real">(aten<"fft_rfft">(x, n, 1)));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_irfft1d_packed(
    b_lean_obj_arg input, uint32_t batch, uint32_t n) {
  return invoke([&] { return box(inverse_real_fft(input, batch, n, true)); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_irfft1d_packed_unnormalized(
    b_lean_obj_arg input, uint32_t batch, uint32_t n) {
  return invoke([&] { return box(inverse_real_fft(input, batch, n, false)); });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_selective_scan_diag_fwd(
    b_lean_obj_arg a, b_lean_obj_arg b, b_lean_obj_arg x, b_lean_obj_arg initial,
    uint32_t steps, uint32_t state) {
  return invoke([&] {
    auto inputs = scan_inputs(a, b, x, initial, steps, state, false);
    return box(affine_scan(inputs.a, aten<"mul.Tensor">(inputs.b, inputs.x), inputs.initial));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_selective_scan_diag_bwd(
    b_lean_obj_arg a, b_lean_obj_arg b, b_lean_obj_arg x, b_lean_obj_arg initial,
    b_lean_obj_arg output, b_lean_obj_arg dout, uint32_t steps, uint32_t state) {
  return invoke([&] {
    auto inputs = scan_inputs(a, b, x, initial, steps, state, false);
    return scan_backward(inputs, checked(output, {steps, state}),
                         checked(dout, {steps, state}), false);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_selective_scan_diag_var_fwd(
    b_lean_obj_arg a, b_lean_obj_arg b, b_lean_obj_arg x, b_lean_obj_arg initial,
    uint32_t steps, uint32_t state) {
  return invoke([&] {
    auto inputs = scan_inputs(a, b, x, initial, steps, state, true);
    return box(affine_scan(inputs.a, aten<"mul.Tensor">(inputs.b, inputs.x), inputs.initial));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_selective_scan_diag_var_bwd(
    b_lean_obj_arg a, b_lean_obj_arg b, b_lean_obj_arg x, b_lean_obj_arg initial,
    b_lean_obj_arg output, b_lean_obj_arg dout, uint32_t steps, uint32_t state) {
  return invoke([&] {
    auto inputs = scan_inputs(a, b, x, initial, steps, state, true);
    return scan_backward(inputs, checked(output, {steps, state}),
                         checked(dout, {steps, state}), true);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_broadcast_to(
    b_lean_obj_arg input, b_lean_obj_arg input_dims,
    b_lean_obj_arg output_dims, b_lean_obj_arg axis_map) {
  return invoke([&] {
    auto shape = broadcast_shape(input_dims, output_dims, axis_map);
    require_size(input, shape.input);
    // An input with a zero extent broadcasts to an output with that extent.
    if (element_count(shape.output) == 0) return box(no_entries(input));
    auto x = shaped(input, shape.input);
    return box(aten<"clone">(aten<"expand">(
        aten<"reshape">(aten<"permute">(x, shape.input_permutation), shape.expanded_input),
        shape.output)));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_from_broadcast(
    b_lean_obj_arg dout, b_lean_obj_arg input_dims,
    b_lean_obj_arg output_dims, b_lean_obj_arg axis_map) {
  return invoke([&] {
    auto shape = broadcast_shape(input_dims, output_dims, axis_map);
    require_size(dout, shape.output);
    const int64_t input_size = element_count(shape.input);
    if (input_size == 0 || element_count(shape.output) == 0) {
      return box(aten<"zeros">({input_size}, options(tensor(dout))));
    }
    auto dy = shaped(dout, shape.output);
    std::vector<int64_t> permutation;
    std::vector<int64_t> kept(shape.input.size(), -1);
    for (size_t axis = 0; axis < shape.map.size(); ++axis) {
      const int64_t mapped = shape.map[axis];
      if (mapped == 0 || shape.input[mapped - 1] == 1) {
        permutation.push_back(static_cast<int64_t>(axis));
      } else {
        kept[mapped - 1] = static_cast<int64_t>(axis);
      }
    }
    for (int64_t axis : kept) {
      if (axis >= 0) permutation.push_back(axis);
    }
    // Reduced coordinates come first in their original row-major order;
    // surviving axes are then arranged in input order.
    auto matrix = aten<"reshape">(aten<"permute">(dy, permutation),
                                  {dy.numel() / input_size, input_size});
    return box(column_sum(matrix, unstable::deterministic()));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_swap_adjacent_at_depth(
    b_lean_obj_arg input, b_lean_obj_arg dims, uint32_t depth) {
  return invoke([&] {
    auto shape = shape_array(dims);
    require(static_cast<uint64_t>(depth) + 1 < shape.size(),
            "swapAdjacentAtDepth: invalid depth");
    require_size(input, shape);
    if (element_count(shape) == 0) return box(no_entries(input));
    return box(aten<"clone">(
        aten<"transpose.int">(shaped(input, shape), depth, static_cast<int64_t>(depth) + 1)));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_sum_axis(
    b_lean_obj_arg input, b_lean_obj_arg dims, uint32_t axis) {
  return invoke([&] {
    auto shape = shape_array(dims);
    require_size(input, shape);
    if (shape.empty()) return box(aten<"clone">(shaped(input, shape)));
    require(axis < shape.size(), "reduceSumAxis: invalid axis");
    std::vector<int64_t> permutation{axis};
    std::vector<int64_t> output_shape;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i != axis) {
        permutation.push_back(static_cast<int64_t>(i));
        output_shape.push_back(shape[i]);
      }
    }
    const int64_t output_size = element_count(output_shape);
    // The sums of an empty input are empty sums, `+0`.
    if (element_count(shape) == 0) {
      return box(aten<"zeros">({output_size}, options(tensor(input))));
    }
    auto x = shaped(input, shape);
    if (!unstable::deterministic()) return box(aten<"sum.dim_IntList">(x, {axis}));
    auto matrix = aten<"reshape">(aten<"permute">(x, permutation), {shape[axis], output_size});
    return box(ordered_columns(matrix));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_gather_rows(
    b_lean_obj_arg input, uint32_t rows, uint32_t cols,
    b_lean_obj_arg indices, uint32_t k) {
  return invoke([&] {
    element_count({k, cols});
    return box(gather_rows(checked(input, {rows, cols}), index_array(indices, k)));
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_scatter_add_rows(
    b_lean_obj_arg input, b_lean_obj_arg values, uint32_t rows, uint32_t cols,
    b_lean_obj_arg indices, uint32_t k) {
  return invoke([&] {
    return box(scatter_rows(checked(input, {rows, cols}), checked(values, {k, cols}),
                            index_array(indices, k)));
  });
}

}  // namespace torchlean::operators
