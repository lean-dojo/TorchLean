// Calls from the LibTorch bridge into torch through the LibTorch Stable ABI, torch 2.13 or later.
//
// Every ATen operator call has the form `aten<"name.overload">(arguments...)`. The key selects the
// operator's schema in the generated table `schemas.h`, the schema is parsed at compile time, and
// the call goes through torch's boxed dispatcher (`torch_call_dispatcher`). A key that is not in
// the table, a wrong number of arguments or an argument of the wrong type does not compile. The
// arguments a call leaves out get the schema's defaults, passed explicitly. Each parameter type
// of a schema has one C++ type:
//
//   Tensor (any alias annotation)   const Tensor&, and a scalar_operand if the operator is in
//                                   schema::no_device_check
//   Tensor?                         const std::optional<Tensor>&
//   Tensor[]                        HeaderOnlyArrayRef<Tensor>
//   int, SymInt                     integer: any integral type except bool
//   int?, SymInt?                   optional_integer: std::nullopt, a std::optional<std::int64_t>
//                                   or an integer as for int
//   int[], int[N], SymInt[]         ints
//   int[N]?, SymInt[]?              const optional_ints&
//   bool                            flag: bool only
//   bool[N]                         const std::array<bool, N>&
//   ScalarType, ScalarType?         ScalarType, std::optional<ScalarType>
//   Device                          Device
//   MemoryFormat, MemoryFormat?     MemoryFormat, std::optional<MemoryFormat>
//   ScalarType? dtype, Layout? layout, Device? device, bool? pin_memory
//                                   const tensor_options&: layout and pin_memory are always None,
//                                   so every tensor is strided and not pinned
//   str? (default None)             always None, and a call passes no parameter from it on
//
// Other types do not compile, among them `Scalar`, which the dispatcher does not convert, and
// `str`, which the torch 2.13 runtime leaks. The rest of this header is what the bridge needs
// besides operators: the CPU 0-dim tensors that stand for Scalar arguments, addition and
// subtraction (whose schemas have a Scalar `alpha`), zero padding (whose schema has a Scalar
// `value`), constants, items, host views, grad mode, and the CUDA device guard and stream. Errors
// throw without printing anything. An operator's error, and a failure to pass its arguments,
// carries torch's message without its backtrace. csrc/libtorch/README.md explains the design,
// and "LibTorch Stable ABI" in docs/TRUST_BOUNDARIES.md lists each schema and C shim the bridge
// uses.
#pragma once

#include "schemas.h"

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/generated/c_shim_aten.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/core/MemoryFormat.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/HeaderOnlyArrayRef.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef USE_CUDA
#error "stable.h needs USE_CUDA: the bridge uses torch's CUDA device guard and stream shims"
#endif
static_assert(TORCH_FEATURE_VERSION >= TORCH_VERSION_2_13_0,
              "the LibTorch bridge targets the Stable ABI of torch 2.13 or later");

// Everything here has hidden visibility, like torch's stable types. The bridge exports its
// `LEAN_EXPORT` C functions (the Lean entry points, and the counters and handle functions of
// torchlean_libtorch.h), `torchlean::initialize` and `device_index`, which the native tests call,
// and the weak copies of the standard library's templates and torch's inline functions that any
// C++ library exports.
namespace torchlean __attribute__((visibility("hidden"))) {

using torch::headeronly::HeaderOnlyArrayRef;
using torch::headeronly::kCPU;
using torch::headeronly::kCUDA;
using torch::headeronly::MemoryFormat;
using torch::headeronly::ScalarType;
using torch::stable::Device;
using torch::stable::Tensor;
using ints = HeaderOnlyArrayRef<std::int64_t>;

inline constexpr ScalarType kBool = ScalarType::Bool, kByte = ScalarType::Byte,
                            kInt = ScalarType::Int, kLong = ScalarType::Long,
                            kHalf = ScalarType::Half, kBFloat16 = ScalarType::BFloat16,
                            kFloat = ScalarType::Float, kDouble = ScalarType::Double;

// ---------------------------------------------------------------------------------------------
// Errors

/// An allocation failure. The IO form of the bridge's entry points reports it as Lean's
/// `resourceExhausted`, every other exception as `otherError`.
struct out_of_memory : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/// The message of an out-of-memory error: one that starts with `prefix` and contains `infix`.
struct memory_message {
  std::string_view prefix, infix;
};

/// The messages of every `c10::OutOfMemoryError` torch 2.13 to 2.15 throw, and of a host
/// `std::bad_alloc`. The C shims keep an error's message and drop its type, so the message is all
/// `raise_last` can tell an allocation failure by.
inline constexpr memory_message memory_messages[] = {
    // libc10_cuda: the native caching allocator, its memory fraction and its size limit
    {"CUDA out of memory.", ""},
    // libc10_cuda: the cudaMallocAsync allocator
    {"Allocation on device ", " would exceed allowed memory. (out of memory)"},
    // libtorch_cuda: the workspace of a cuDNN convolution
    {"Not enough memory for workspace!", ""},
    // the C++ runtime, and `std::bad_array_new_length`, a `std::bad_alloc`
    {"std::bad_alloc", ""},
    {"std::bad_array_new_length", ""},
};

/// Throws the error of the last failed torch C call on this thread, with torch's message without
/// its backtrace: `out_of_memory` if the message is one of `memory_messages`, and
/// `std::runtime_error` otherwise.
[[noreturn]] inline void raise_last() {
  const char* what = torch_exception_get_what_without_backtrace();
  std::string message = what != nullptr && *what != '\0' ? what : "LibTorch: unknown error";
  for (const auto& form : memory_messages) {
    if (message.starts_with(form.prefix) && message.find(form.infix) != std::string::npos) {
      throw out_of_memory(message);
    }
  }
  throw std::runtime_error(message);
}

/// Checks the status of a torch C call.
inline void check(AOTITorchError status) {
  if (status != TORCH_SUCCESS) [[unlikely]] raise_last();
}

/// Throws `std::runtime_error` whose message is `parts` written one after another.
template <typename... Parts>
[[noreturn]] void fail(const Parts&... parts) {
  std::ostringstream message;
  (message << ... << parts);
  throw std::runtime_error(message.str());
}

#define TORCHLEAN_CHECK(condition, ...)                            \
  do {                                                             \
    if (!(condition)) [[unlikely]] ::torchlean::fail(__VA_ARGS__); \
  } while (false)

// ---------------------------------------------------------------------------------------------
// Data types and tensor options

/// The code of `dtype` in torch's C shims.
inline std::int32_t code(ScalarType dtype) {
  return torch::stable::detail::to<std::int32_t>(torch::stable::detail::from(dtype));
}

/// The bytes of one element of `dtype`.
inline std::size_t element_size(ScalarType dtype) {
  return aoti_torch_dtype_element_size(code(dtype));
}

/// `c10::isIntegralType`: the integer types, and `Bool` if `with_bool`.
constexpr bool is_integral(ScalarType dtype, bool with_bool) {
  switch (dtype) {
    case ScalarType::Byte:
    case ScalarType::Char:
    case ScalarType::Short:
    case ScalarType::Int:
    case ScalarType::Long:
    case ScalarType::UInt16:
    case ScalarType::UInt32:
    case ScalarType::UInt64:
      return true;
    case ScalarType::Bool:
      return with_bool;
    default:
      return false;
  }
}

/// A CPU 0-dim tensor that stands for a Scalar argument, made by `cpu_scalar` or `scalar_for`.
/// ATen's CUDA kernel of an operator declared with a device check rejects it beside CUDA tensors,
/// so it is an argument only of the operators in `schema::no_device_check`. Passing it to any other
/// operator does not compile.
struct scalar_operand {
  Tensor tensor;
};

/// The options group of a factory schema: a dtype and a device, each None unless set.
struct tensor_options {
  std::optional<ScalarType> dtype_;
  std::optional<Device> device_;

  tensor_options dtype(ScalarType value) const {
    auto result = *this;
    result.dtype_ = value;
    return result;
  }
  tensor_options device(Device value) const {
    auto result = *this;
    result.device_ = value;
    return result;
  }
};

/// The dtype and device of `x`.
inline tensor_options options(const Tensor& x) { return {x.scalar_type(), x.device()}; }

/// A `bool` argument. Only `bool` converts, so an integer in a flag's place does not compile.
struct flag {
  bool value;
  template <std::same_as<bool> T>
  constexpr flag(T v) : value(v) {}
};

/// An `int` or `SymInt` argument: any integral type except `bool` converts.
struct integer {
  std::int64_t value;
  template <std::integral T>
    requires(!std::same_as<T, bool>)
  constexpr integer(T v) : value(static_cast<std::int64_t>(v)) {}
};

/// An `int?` or `SymInt?` argument: `std::nullopt`, a `std::optional<std::int64_t>` or an integer
/// as for `integer`, so a `double` or a `bool` in its place does not compile either.
struct optional_integer {
  std::optional<std::int64_t> value;

  constexpr optional_integer(std::nullopt_t) {}
  constexpr optional_integer(std::optional<std::int64_t> v) : value(v) {}
  template <std::integral T>
    requires(!std::same_as<T, bool>)
  constexpr optional_integer(T v) : value(static_cast<std::int64_t>(v)) {}
};

/// An `int[N]?` or `SymInt[]?` argument: `std::nullopt` or a list of integers.
struct optional_ints {
  std::optional<std::vector<std::int64_t>> value;

  optional_ints(std::nullopt_t) {}
  optional_ints(std::initializer_list<std::int64_t> v) : value(v) {}
  optional_ints(ints v) : value(std::vector<std::int64_t>(v.begin(), v.end())) {}
  optional_ints(const std::vector<std::int64_t>& v) : value(v) {}
};

// ---------------------------------------------------------------------------------------------
// The dispatcher call

namespace detail {

/// A string literal as a template argument.
template <std::size_t N>
struct fixed_string {
  char text[N]{};
  consteval fixed_string(const char (&value)[N]) { std::copy_n(value, N, text); }
  constexpr std::string_view view() const { return {text, N - 1}; }
};

enum class kind : unsigned char {
  tensor, optional_tensor, tensors, integer, optional_integer, integers, optional_integers,
  boolean, booleans, dtype, optional_dtype, device, memory_format, optional_memory_format,
  options, none
};

/// What a call that leaves a parameter out passes for it.
enum class fallback : unsigned char { required, null, value, list, contiguous };

struct parameter {
  kind type{};
  fallback missing = fallback::required;
  std::int64_t value = 0;  // the default of an int or bool, or each entry of a list default
  std::size_t count = 0;   // the entries of a list default
  std::size_t length = 0;  // N of `bool[N]` and `int[N]`
  std::size_t slots = 1;   // the stack entries it takes: 4 for the options group
};

struct signature {
  std::array<parameter, 16> parameters{};
  std::size_t size = 0;      // the parameters, the options group counted once
  std::size_t required = 0;  // the leading parameters every call passes
  std::size_t accepted = 0;  // the leading parameters a call may pass
  std::size_t slots = 0;     // the stack entries of all parameters
  std::size_t returns = 0;
};

/// Called only while parsing a schema the helper does not support. It is not `constexpr`, so the
/// compile-time evaluation stops here and the compiler names `reason`.
void unsupported(const char* reason);

consteval std::string_view trim(std::string_view text) {
  while (!text.empty() && text.front() == ' ') text.remove_prefix(1);
  while (!text.empty() && text.back() == ' ') text.remove_suffix(1);
  return text;
}

struct pieces {
  std::array<std::string_view, 24> items{};
  std::size_t size = 0;
};

/// The comma-separated items of `text` that are outside brackets and parentheses.
consteval pieces split(std::string_view text) {
  pieces result;
  std::size_t depth = 0, start = 0;
  for (std::size_t i = 0; i <= text.size(); ++i) {
    if (i == text.size() || (text[i] == ',' && depth == 0)) {
      if (result.size == result.items.size()) unsupported("too many items");
      result.items[result.size++] = trim(text.substr(start, i - start));
      start = i + 1;
    } else if (text[i] == '(' || text[i] == '[') {
      ++depth;
    } else if (text[i] == ')' || text[i] == ']') {
      --depth;
    }
  }
  return result;
}

/// `type` without an alias annotation such as `(a)` or `(a!)`.
consteval std::string_view unannotated(std::string_view type) {
  const auto open = type.find('(');
  if (open == std::string_view::npos) return type;
  const auto close = type.find(')', open);
  if (close == std::string_view::npos) unsupported("unclosed alias annotation");
  if (close + 1 != type.size()) unsupported("alias annotation before a type suffix");
  return type.substr(0, open);
}

consteval std::int64_t integer_literal(std::string_view text) {
  const bool negative = !text.empty() && text.front() == '-';
  if (negative) text.remove_prefix(1);
  if (text.empty()) unsupported("empty integer default");
  std::int64_t value = 0;
  for (const char digit : text) {
    if (digit < '0' || digit > '9') unsupported("unsupported default");
    value = 10 * value + (digit - '0');
  }
  return negative ? -value : value;
}

/// One declaration `type name` or `type name=default` of a schema.
struct declaration {
  std::string_view type, name, value;
  bool defaulted = false;
};

consteval declaration declare(std::string_view text) {
  const auto space = text.find(' ');
  if (space == std::string_view::npos) unsupported("parameter without a name");
  declaration result{text.substr(0, space), text.substr(space + 1), {}, false};
  const auto equals = result.name.find('=');
  if (equals != std::string_view::npos) {
    result.value = result.name.substr(equals + 1);
    result.name = result.name.substr(0, equals);
    result.defaulted = true;
  }
  return result;
}

/// Whether `items[at...]` is the options group `ScalarType? dtype=None, Layout? layout=None,
/// Device? device=None, bool? pin_memory=None`.
consteval bool options_group(const pieces& items, std::size_t at) {
  constexpr std::array<std::string_view, 4> group = {
      "ScalarType? dtype=None", "Layout? layout=None", "Device? device=None",
      "bool? pin_memory=None"};
  if (at + group.size() > items.size) return false;
  for (std::size_t i = 0; i < group.size(); ++i) {
    if (items.items[at + i] != group[i]) return false;
  }
  return true;
}

consteval parameter typed(const declaration& item) {
  std::string_view type = unannotated(item.type);
  const bool optional = type.ends_with('?');
  if (optional) type.remove_suffix(1);
  bool list = false;
  std::size_t length = 0;
  if (type.ends_with(']')) {
    const auto open = type.find('[');
    for (const char digit : type.substr(open + 1, type.size() - open - 2)) {
      if (digit < '0' || digit > '9') unsupported("unsupported list length");
      length = 10 * length + static_cast<std::size_t>(digit - '0');
    }
    type = type.substr(0, open);
    list = true;
  }
  parameter result;
  result.length = length;
  if (type == "Tensor" && !list) {
    result.type = optional ? kind::optional_tensor : kind::tensor;
  } else if (type == "Tensor" && length == 0 && !optional) {
    result.type = kind::tensors;
  } else if (type == "int" || type == "SymInt") {
    result.type = list ? (optional ? kind::optional_integers : kind::integers)
                       : (optional ? kind::optional_integer : kind::integer);
  } else if (type == "bool" && !optional) {
    if (list && length == 0) unsupported("bool[] without a length");
    result.type = list ? kind::booleans : kind::boolean;
  } else if (type == "ScalarType" && !list) {
    result.type = optional ? kind::optional_dtype : kind::dtype;
  } else if (type == "Device" && !list && !optional) {
    result.type = kind::device;
  } else if (type == "MemoryFormat" && !list) {
    result.type = optional ? kind::optional_memory_format : kind::memory_format;
  } else if (type == "str" && !list && optional) {
    result.type = kind::none;
  } else {
    unsupported("unsupported parameter type: Scalar, str, float, Layout, Generator or Tensor?[]");
  }
  if (!item.defaulted) {
    if (result.type == kind::none) unsupported("str? without the default None");
    return result;
  }
  const std::string_view value = item.value;
  const bool nullable = result.type == kind::optional_tensor ||
                        result.type == kind::optional_integer ||
                        result.type == kind::optional_integers ||
                        result.type == kind::optional_dtype ||
                        result.type == kind::optional_memory_format || result.type == kind::none;
  if (value == "None") {
    if (!nullable) unsupported("None default of a parameter that is not optional");
    result.missing = fallback::null;
  } else if (result.type == kind::none) {
    unsupported("str? with a default other than None");
  } else if (result.type == kind::boolean && (value == "True" || value == "False")) {
    result.missing = fallback::value;
    result.value = value == "True" ? 1 : 0;
  } else if (result.type == kind::integer) {
    result.missing = fallback::value;
    result.value = integer_literal(value);
  } else if (result.type == kind::integers && value == "[]") {
    result.missing = fallback::list;
  } else if (result.type == kind::integers && length > 0) {
    // A scalar default of `int[N]` stands for N copies of it.
    result.missing = fallback::list;
    result.value = integer_literal(value);
    result.count = length;
  } else if (result.type == kind::memory_format && value == "contiguous_format") {
    result.missing = fallback::contiguous;
  } else {
    unsupported("unsupported default");
  }
  return result;
}

/// The number of returns of `text` (what follows ` -> `). Every return must be a Tensor.
consteval std::size_t tensor_returns(std::string_view text) {
  if (text.starts_with('(')) {
    if (!text.ends_with(')')) unsupported("unclosed return list");
    text = text.substr(1, text.size() - 2);
  }
  const pieces items = split(text);
  for (std::size_t i = 0; i < items.size; ++i) {
    const auto item = items.items[i];
    if (unannotated(item.substr(0, item.find(' '))) != "Tensor") {
      unsupported("unsupported return type: every return must be a Tensor");
    }
  }
  return items.size;
}

consteval signature parse(std::string_view schema) {
  const auto open = schema.find('(');
  std::size_t depth = 0, close = open;
  for (; close < schema.size(); ++close) {
    if (schema[close] == '(') {
      ++depth;
    } else if (schema[close] == ')' && --depth == 0) {
      break;
    }
  }
  if (close == schema.size()) unsupported("unclosed parameter list");
  const std::string_view arrow = " -> ";
  if (schema.substr(close + 1, arrow.size()) != arrow) unsupported("no ` -> ` after parameters");

  signature result;
  result.returns = tensor_returns(schema.substr(close + 1 + arrow.size()));
  const pieces items = split(schema.substr(open + 1, close - open - 1));
  bool open_ended = false;
  for (std::size_t i = 0; i < items.size; ++i) {
    if (items.items[i] == "*") continue;
    parameter item;
    if (options_group(items, i)) {
      item.type = kind::options;
      item.missing = fallback::null;
      item.slots = 4;
      i += 3;
    } else {
      item = typed(declare(items.items[i]));
    }
    if (result.size == result.parameters.size()) unsupported("too many parameters");
    if (item.type == kind::none && !open_ended) {
      open_ended = true;
      result.accepted = result.size;
    }
    if (item.missing == fallback::required) {
      if (open_ended) unsupported("required parameter after a str? parameter");
      result.required = result.size + 1;
    }
    result.slots += item.slots;
    result.parameters[result.size++] = item;
  }
  if (!open_ended) result.accepted = result.size;
  return result;
}

inline constexpr std::size_t absent = static_cast<std::size_t>(-1);

consteval std::size_t find(std::string_view key) {
  for (std::size_t i = 0; i < std::size(schema::table); ++i) {
    const auto entry = schema::table[i];
    if (entry.substr(0, entry.find('(')) == key) return i;
  }
  return absent;
}

template <fixed_string Key>
consteval signature schema_signature() {
  constexpr std::size_t index = find(Key.view());
  static_assert(index != absent,
                "aten: the operator is not in schemas.h; run "
                "scripts/checks/libtorch_abi.py --write --torch <torch package>");
  return parse(schema::table[index == absent ? 0 : index]);
}

template <fixed_string Key>
inline constexpr signature schema_of = schema_signature<Key>();

/// Whether operator `key` is in `schema::no_device_check`.
consteval bool no_device_check(std::string_view key) {
  return std::ranges::find(schema::no_device_check, key) != std::end(schema::no_device_check);
}

/// The operator name `aten::name` and the overload name of `Key`, as C strings in one array.
template <fixed_string Key>
struct names {
  static constexpr std::string_view key = Key.view();
  static constexpr std::size_t dot = std::min(key.find('.'), key.size());
  static constexpr std::array<char, 6 + key.size() + 2> text = [] {
    std::array<char, 6 + key.size() + 2> result{};
    std::copy_n("aten::", 6, result.begin());
    std::copy(key.begin(), key.end(), result.begin() + 6);
    if (dot < key.size()) result[6 + dot] = '\0';
    return result;
  }();
  static constexpr const char* op = text.data();
  static constexpr const char* overload = text.data() + 6 + dot + 1;
};

/// A `Tensor` argument: a tensor, or a `scalar_operand` if `Unchecked`, which is whether the
/// operator is in `schema::no_device_check`.
template <bool Unchecked>
struct tensor_argument {
  const Tensor& value;
  tensor_argument(const Tensor& v) : value(v) {}
  tensor_argument(const scalar_operand& v)
    requires Unchecked
      : value(v.tensor) {}
};

/// The C++ type of a parameter of kind `K` other than `Tensor`, with `N` from `bool[N]`.
template <kind K, std::size_t N>
struct argument;
template <std::size_t N>
struct argument<kind::optional_tensor, N> { using type = const std::optional<Tensor>&; };
template <std::size_t N>
struct argument<kind::tensors, N> { using type = HeaderOnlyArrayRef<Tensor>; };
template <std::size_t N> struct argument<kind::integer, N> { using type = integer; };
template <std::size_t N>
struct argument<kind::optional_integer, N> { using type = optional_integer; };
template <std::size_t N> struct argument<kind::integers, N> { using type = ints; };
template <std::size_t N>
struct argument<kind::optional_integers, N> { using type = const optional_ints&; };
template <std::size_t N> struct argument<kind::boolean, N> { using type = flag; };
template <std::size_t N>
struct argument<kind::booleans, N> { using type = const std::array<bool, N>&; };
template <std::size_t N> struct argument<kind::dtype, N> { using type = ScalarType; };
template <std::size_t N>
struct argument<kind::optional_dtype, N> { using type = std::optional<ScalarType>; };
template <std::size_t N> struct argument<kind::device, N> { using type = Device; };
template <std::size_t N> struct argument<kind::memory_format, N> { using type = MemoryFormat; };
template <std::size_t N>
struct argument<kind::optional_memory_format, N> { using type = std::optional<MemoryFormat>; };
template <std::size_t N> struct argument<kind::options, N> { using type = const tensor_options&; };

/// The C++ type of parameter `I` of `Key`.
template <fixed_string Key, std::size_t I, kind K = schema_of<Key>.parameters[I].type>
struct argument_of : argument<K, schema_of<Key>.parameters[I].length> {};
template <fixed_string Key, std::size_t I>
struct argument_of<Key, I, kind::tensor> {
  using type = tensor_argument<no_device_check(Key.view())>;
};

template <fixed_string Key, std::size_t I>
using argument_t = typename argument_of<Key, I>::type;

/// Frees what a StableIValue owns, or null for a value that owns nothing.
using dropper = void (*)(StableIValue) noexcept;

/// How a `T` goes on the dispatcher's stack: `box(value)` is its StableIValue and, if `owns`,
/// `drop` frees what that owns. A tensor, a list and a present optional own a new torch object
/// each, made with `check`, so a failure to make one reports torch's error as an operator's error
/// does. A value of any other type is its bits and owns nothing. The dispatcher frees what it
/// takes.
template <typename T>
struct boxing {
  static constexpr bool owns = false;
  static StableIValue box(const T& value) { return torch::stable::detail::from(value); }
  static constexpr dropper drop = nullptr;
};

/// A boxed value that `drop` frees unless `release` hands it over first.
class held {
 public:
  held(StableIValue value, dropper drop) : value_(value), drop_(drop) {}
  held(const held&) = delete;
  held& operator=(const held&) = delete;
  ~held() {
    if (drop_ != nullptr) drop_(value_);
  }
  StableIValue get() const { return value_; }
  StableIValue release() {
    drop_ = nullptr;
    return value_;
  }

 private:
  StableIValue value_;
  dropper drop_;
};

/// A new reference to the tensor.
template <>
struct boxing<Tensor> {
  static constexpr bool owns = true;
  static StableIValue box(const Tensor& value) {
    AtenTensorHandle result = nullptr;
    check(aoti_torch_new_tensor_handle(value.get(), &result));
    return torch::stable::detail::from(result);
  }
  static constexpr dropper drop = [](StableIValue value) noexcept {
    (void)aoti_torch_delete_tensor_object(torch::stable::detail::to<AtenTensorHandle>(value));
  };
};

/// Null, or a new box holding the boxed value.
template <typename T>
struct boxing<std::optional<T>> {
  static constexpr bool owns = true;
  static StableIValue box(const std::optional<T>& value) {
    if (!value) return torch::stable::detail::from(std::nullopt);
    held item(boxing<T>::box(*value), boxing<T>::drop);
    StableIValue* result = nullptr;
    check(torch_new_stable_ivalue(&result));
    *result = item.release();
    return torch::stable::detail::from(result);
  }
  static constexpr dropper drop = [](StableIValue value) noexcept {
    auto* const item = torch::stable::detail::to<StableIValue*>(value);
    if (item == nullptr) return;
    if constexpr (boxing<T>::owns) boxing<T>::drop(*item);
    (void)torch_delete_stable_ivalue(item);
  };
};

/// A new list of the boxed entries.
template <typename T>
struct list_boxing {
  static constexpr bool owns = true;
  static StableIValue box(HeaderOnlyArrayRef<T> values) {
    StableListHandle list = nullptr;
    check(torch_new_list_reserve_size(values.size(), &list));
    held result(torch::stable::detail::from(list), drop);
    for (const T& value : values) {
      held item(boxing<T>::box(value), boxing<T>::drop);
      check(torch_list_push_back(list, item.get()));
      (void)item.release();
    }
    return result.release();
  }
  static constexpr dropper drop = [](StableIValue value) noexcept {
    const auto list = torch::stable::detail::to<StableListHandle>(value);
    if constexpr (boxing<T>::owns) {
      std::size_t size = 0;
      (void)torch_list_size(list, &size);
      for (std::size_t i = 0; i < size; ++i) {
        StableIValue item{};
        if (torch_list_get_item(list, i, &item) == TORCH_SUCCESS) boxing<T>::drop(item);
      }
    }
    (void)torch_delete_list(list);
  };
};
template <typename T>
struct boxing<HeaderOnlyArrayRef<T>> : list_boxing<T> {};
template <typename T>
struct boxing<std::vector<T>> : list_boxing<T> {};

/// The stack of one dispatcher call. It owns the arguments pushed on it until `hand_over` gives
/// them to the dispatcher, which takes them whether or not the call succeeds, so an argument that
/// fails to box frees those before it. After the call it owns the returns until `take`.
template <std::size_t N>
class stack {
 public:
  stack() = default;
  stack(const stack&) = delete;
  stack& operator=(const stack&) = delete;
  ~stack() {
    for (std::size_t i = 0; i < size_; ++i) {
      if (drops_[i] != nullptr) drops_[i](values_[i]);
    }
  }

  template <typename T>
  void push(const T& value) {
    values_[size_] = boxing<T>::box(value);
    drops_[size_++] = boxing<T>::drop;
  }
  template <bool Unchecked>
  void push(const tensor_argument<Unchecked>& value) {
    push(value.value);
  }
  void push(flag value) { push(value.value); }
  void push(integer value) { push(value.value); }
  void push(const optional_integer& value) { push(value.value); }
  void push(const optional_ints& value) { push(value.value); }
  template <std::size_t M>
  void push(const std::array<bool, M>& value) {
    push(HeaderOnlyArrayRef<bool>(value.data(), M));
  }
  void push(const tensor_options& value) {
    push(value.dtype_);
    push(std::nullopt);  // layout: strided
    push(value.device_);
    push(std::nullopt);  // pin_memory
  }

  StableIValue* hand_over() {
    size_ = 0;
    return values_.data();
  }
  /// After the call succeeded: the stack owns its first `count` entries, the returned tensors.
  void returned(std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) drops_[i] = boxing<Tensor>::drop;
    size_ = count;
  }
  /// Return `i` as a Tensor, which owns it from here on.
  Tensor take(std::size_t i) {
    drops_[i] = nullptr;
    return Tensor(torch::stable::detail::to<AtenTensorHandle>(values_[i]));
  }

 private:
  std::array<StableIValue, N> values_{};
  std::array<dropper, N> drops_{};
  std::size_t size_ = 0;
};

/// Pushes the default of parameter `I` of `Key`.
template <fixed_string Key, std::size_t I, std::size_t N>
void push_default(stack<N>& top) {
  constexpr parameter item = schema_of<Key>.parameters[I];
  static_assert(item.missing != fallback::required);
  if constexpr (item.missing == fallback::null) {
    for (std::size_t slot = 0; slot < item.slots; ++slot) top.push(std::nullopt);
  } else if constexpr (item.missing == fallback::value && item.type == kind::boolean) {
    top.push(item.value != 0);
  } else if constexpr (item.missing == fallback::value) {
    top.push(item.value);
  } else if constexpr (item.missing == fallback::list) {
    top.push(std::vector<std::int64_t>(item.count, item.value));
  } else {
    top.push(MemoryFormat::Contiguous);
  }
}

/// The one call of the dispatcher.
template <fixed_string Key, typename... Arguments>
auto call(const Arguments&... arguments) {
  constexpr std::size_t returns = schema_of<Key>.returns;
  stack<std::max(schema_of<Key>.slots, returns)> top;
  (top.push(arguments), ...);
  [&]<std::size_t... D>(std::index_sequence<D...>) {
    (push_default<Key, sizeof...(Arguments) + D>(top), ...);
  }(std::make_index_sequence<schema_of<Key>.size - sizeof...(Arguments)>{});
  check(torch_call_dispatcher(names<Key>::op, names<Key>::overload, top.hand_over(),
                              TORCH_ABI_VERSION));
  top.returned(returns);
  if constexpr (returns == 1) {
    return top.take(0);
  } else {
    return [&]<std::size_t... R>(std::index_sequence<R...>) {
      return std::array<Tensor, sizeof...(R)>{top.take(R)...};
    }(std::make_index_sequence<returns>{});
  }
}

template <fixed_string Key, typename Indices>
struct overload;

template <fixed_string Key, std::size_t... I>
struct overload<Key, std::index_sequence<I...>> {
  auto operator()(argument_t<Key, I>... arguments) const { return call<Key>(arguments...); }
};

template <fixed_string Key, typename Extra>
struct overloads;

/// One call operator for each number of arguments from `required` to `accepted`.
template <fixed_string Key, std::size_t... E>
struct overloads<Key, std::index_sequence<E...>>
    : overload<Key, std::make_index_sequence<schema_of<Key>.required + E>>... {
  using overload<Key, std::make_index_sequence<schema_of<Key>.required + E>>::operator()...;
};

template <fixed_string Key>
using operators =
    overloads<Key, std::make_index_sequence<schema_of<Key>.accepted - schema_of<Key>.required + 1>>;

}  // namespace detail

/// The ATen operator `Key` ("name" or "name.overload"), called through the stable dispatcher.
/// It returns a `Tensor`, or a `std::array<Tensor, N>` for an operator with N returns.
template <detail::fixed_string Key>
inline constexpr detail::operators<Key> aten{};

// ---------------------------------------------------------------------------------------------
// Scalars, addition, padding and constants

namespace detail {

inline Tensor cpu_tensor(std::int64_t value) {
  AtenTensorHandle result = nullptr;
  check(aoti_torch_scalar_to_tensor_int64(value, &result));
  return Tensor(result);
}
inline Tensor cpu_tensor(double value) {
  AtenTensorHandle result = nullptr;
  check(aoti_torch_scalar_to_tensor_float64(value, &result));
  return Tensor(result);
}

}  // namespace detail

/// `value` as a CPU 0-dim tensor of its own type, Long or Double. A `bool`, an `int` or another
/// integer argument is ambiguous on purpose, so every call names its type.
inline scalar_operand cpu_scalar(std::int64_t value) { return {detail::cpu_tensor(value)}; }
inline scalar_operand cpu_scalar(double value) { return {detail::cpu_tensor(value)}; }

/// The operand that makes the `.Tensor` overload of an operator compute with `x` exactly what its
/// `.Scalar` overload computes with `value`: a CPU 0-dim tensor, which ATen passes to the kernel
/// as a scalar like the Scalar overload's wrapped number. Its dtype makes type promotion agree.
/// An integer takes the dtype of an integer `x` and is Long otherwise. A floating value takes the
/// dtype of a Float or Double `x` and is Float for an integer or Bool `x` and a Half or BFloat16
/// `x` of rank one or more. Promotion differs for a 0-dim Half or BFloat16 `x` and for a complex
/// `x`, which therefore fail.
inline scalar_operand scalar_for(const Tensor& x, std::int64_t value) {
  const auto dtype = x.scalar_type();
  const auto result = detail::cpu_tensor(value);
  if (!is_integral(dtype, false) || dtype == kLong) return {result};
  return {aten<"to.dtype">(result, dtype, false, false)};
}
inline scalar_operand scalar_for(const Tensor& x, double value) {
  const auto dtype = x.scalar_type();
  const auto result = detail::cpu_tensor(value);
  if (dtype == kDouble) return {result};
  if (dtype == kFloat || is_integral(dtype, true) ||
      ((dtype == kHalf || dtype == kBFloat16) && x.dim() > 0)) {
    return {aten<"to.dtype">(result, kFloat, false, false)};
  }
  fail("LibTorch: no Scalar operand for a tensor of dtype ", dtype, " and rank ", x.dim());
}

namespace detail {

/// The broadcast shape of `a` and `b`, with ATen's error (for the last dimension that fails, as
/// ATen reports it) if they do not broadcast.
inline std::vector<std::int64_t> broadcast(ints a, ints b) {
  const std::size_t rank = std::max(a.size(), b.size());
  std::vector<std::int64_t> shape(rank);
  for (std::size_t i = rank; i-- > 0;) {
    const std::int64_t x = i + a.size() < rank ? 1 : a[i + a.size() - rank];
    const std::int64_t y = i + b.size() < rank ? 1 : b[i + b.size() - rank];
    TORCHLEAN_CHECK(x == y || x == 1 || y == 1, "The size of tensor a (", x,
                    ") must match the size of tensor b (", y, ") at non-singleton dimension ", i);
    shape[i] = x == 1 ? y : x;
  }
  return shape;
}

/// `a + sign * b` for integer or Bool tensors of one dtype: the sum, over a new leading dimension,
/// of both operands broadcast. The CUDA sum of N-bit integers wraps modulo 2^N like `add`.
inline Tensor integer_sum(const Tensor& a, const Tensor& b, bool negate) {
  const auto dtype = a.scalar_type();
  TORCHLEAN_CHECK(b.scalar_type() == dtype, "LibTorch: integer addition needs one dtype, not ",
                  dtype, " and ", b.scalar_type());
  // A CPU 0-dim operand is filled into a tensor on `a`'s device first.
  static_assert(no_device_check("fill_.Tensor"));
  const auto other = b.is_cpu() && b.dim() == 0 && !a.is_cpu()
                         ? aten<"fill_.Tensor">(aten<"empty_like">(a), b)
                         : b;
  const auto shape = broadcast(a.sizes(), other.sizes());
  const auto addend = negate ? aten<"neg">(other) : other;
  return aten<"sum.dim_IntList">(
      aten<"stack">({aten<"expand">(a, shape), aten<"expand">(addend, shape)}), {0}, false, dtype);
}

}  // namespace detail

/// `at::add(a, b)` and `at::sub(a, b)` (alpha 1), for a tensor or a `scalar_operand` `b`.
/// Floating operands go through the C shim `subtract.Tensor` with alpha -1 or 1, which runs ATen's
/// add kernel with alpha 1 or -1 as `add` and `sub` do. Integer and Bool operands go through
/// `detail::integer_sum`. A CPU 0-dim `b` beside a CUDA `a` reaches `sub.Tensor`, which
/// `subtract.Tensor` calls, or `fill_.Tensor`, and neither checks devices.
static_assert(detail::no_device_check("sub.Tensor"));
inline Tensor add(const Tensor& a, const Tensor& b) {
  if (is_integral(a.scalar_type(), true) && is_integral(b.scalar_type(), true)) {
    return detail::integer_sum(a, b, false);
  }
  AtenTensorHandle result = nullptr;
  check(aoti_torch_aten_subtract_Tensor(a.get(), b.get(), -1.0, &result));
  return Tensor(result);
}
inline Tensor sub(const Tensor& a, const Tensor& b) {
  if (is_integral(a.scalar_type(), true) && is_integral(b.scalar_type(), true)) {
    return detail::integer_sum(a, b, true);
  }
  AtenTensorHandle result = nullptr;
  check(aoti_torch_aten_subtract_Tensor(a.get(), b.get(), 1.0, &result));
  return Tensor(result);
}
inline Tensor add(const Tensor& a, const scalar_operand& b) { return add(a, b.tensor); }
inline Tensor sub(const Tensor& a, const scalar_operand& b) { return sub(a, b.tensor); }

/// `self.add_(b)`: the sum, copied into `self`.
inline const Tensor& add_(const Tensor& self, const Tensor& b) {
  aten<"copy_">(self, add(self, b));
  return self;
}

/// `at::constant_pad_nd(x, pad, 0)`: the C shim `pad` in mode "constant" with value 0, which runs
/// `constant_pad_nd` with the value 0.0, the same fill.
inline Tensor zero_pad(const Tensor& x, ints pad) {
  double value = 0.0;
  AtenTensorHandle result = nullptr;
  check(aoti_torch_aten_pad(x.get(), pad.data(), static_cast<std::int64_t>(pad.size()), "constant",
                            &value, &result));
  return Tensor(result);
}

/// `self.fill_(value)`: ATen's `fill_.Tensor` with a CPU 0-dim value on another device fills with
/// `value.item()`, the Scalar the Scalar overload gets.
template <typename T>
const Tensor& fill(const Tensor& self, T value) {
  aten<"fill_.Tensor">(self, cpu_scalar(value));
  return self;
}

/// `at::full(size, value, options)`, `at::full_like(x, value)` and `at::scalar_tensor(value,
/// options)`: an empty tensor, filled. The options must name the dtype, which ATen would otherwise
/// infer from the value.
template <typename T>
Tensor full(ints size, T value, const tensor_options& options) {
  TORCHLEAN_CHECK(options.dtype_.has_value(), "LibTorch: full needs a dtype");
  const auto result = aten<"empty.memory_format">(size, options);
  fill(result, value);
  return result;
}
template <typename T>
Tensor full_like(const Tensor& x, T value) {
  const auto result = aten<"empty_like">(x);
  fill(result, value);
  return result;
}
template <typename T>
Tensor scalar_tensor(T value, const tensor_options& options) {
  return full({}, value, options);
}

/// `at::arange(n, options)` for Long options: ones with a zero first entry, summed cumulatively.
inline Tensor arange(std::int64_t n, const tensor_options& options) {
  TORCHLEAN_CHECK(options.dtype_ == kLong, "LibTorch: arange builds Long ranges only");
  const auto result = aten<"ones">({n}, options);
  if (n > 0) fill(aten<"select.int">(result, 0, 0), std::int64_t{0});
  return aten<"cumsum">(result, 0);
}

// ---------------------------------------------------------------------------------------------
// Items, host views, grad mode, device and stream

/// `x.item<T>()` for T = bool or std::int64_t: the value of a one-element tensor.
template <typename T>
T item(const Tensor& x);
template <>
inline bool item<bool>(const Tensor& x) {
  bool result = false;
  check(aoti_torch_item_bool(x.get(), &result));
  return result;
}
template <>
inline std::int64_t item<std::int64_t>(const Tensor& x) {
  std::int64_t result = 0;
  check(aoti_torch_item_int64(x.get(), &result));
  return result;
}

/// `at::from_blob(data, {size}, dtype)`: a CPU vector over `data`, which it does not own.
inline Tensor host_view(void* data, std::int64_t size, ScalarType dtype) {
  const std::int64_t stride = 1;
  AtenTensorHandle result = nullptr;
  check(aoti_torch_create_tensor_from_blob(data, 1, &size, &stride, 0, code(dtype),
                                           aoti_torch_device_type_cpu(), -1, &result));
  return Tensor(result);
}

/// `at::NoGradGuard`: grad mode is off while it lives and is restored when it is destroyed.
class no_grad {
 public:
  no_grad() : previous_(aoti_torch_grad_mode_is_enabled()) {
    aoti_torch_grad_mode_set_enabled(false);
  }
  ~no_grad() { aoti_torch_grad_mode_set_enabled(previous_); }
  no_grad(const no_grad&) = delete;
  no_grad& operator=(const no_grad&) = delete;

 private:
  bool previous_;
};

/// `c10::cuda::CUDAGuard`: CUDA device `index` is current while it lives, then the previous one.
class cuda_guard {
 public:
  explicit cuda_guard(std::int32_t index) { check(aoti_torch_create_cuda_guard(index, &guard_)); }
  ~cuda_guard() { aoti_torch_delete_cuda_guard(guard_); }
  cuda_guard(const cuda_guard&) = delete;
  cuda_guard& operator=(const cuda_guard&) = delete;

 private:
  CUDAGuardHandle guard_ = nullptr;
};

/// The current CUDA stream of device `index` (a `cudaStream_t`).
inline void* current_stream(std::int32_t index) {
  void* stream = nullptr;
  check(aoti_torch_get_current_cuda_stream(index, &stream));
  return stream;
}

}  // namespace torchlean
