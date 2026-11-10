// Native side of `TorchLean.Kernel.Arena.Device`. Lean packs the columns of an arena into two
// blocks of little-endian words and proves that reading the blocks back gives the columns. Here we
// keep the words block as int64 columns (the tag column, then each index column) and the scalar
// block as float, double or byte columns in device memory, and return the same bytes on download.
// A handle remembers the scalar storage, width and precision descriptor it was uploaded with and
// refuses to be read with any other. A fold runs one generated row kernel per level, from the
// deepest level up, each launch reading the results of the level below. An expansion builds the
// levels of the calls of a hylomorphism from the roots down, with one generated row kernel for
// the numbers of calls, a prefix sum, and one generated row kernel per slot of the calls. With
// node kernels it builds the arena of the values of an unfold instead, one generated row kernel
// per column of the nodes of a level. LibTorch
// and CUDA failures, device out-of-memory included, come back as IO errors. Lean objects are
// allocated by the Lean runtime, which aborts when the host is out of memory, as it does
// everywhere else.
#include "torchlean_libtorch.h"

#include <cstring>
#include <string>

namespace {

struct tree {
  // (1 + index_slots) * count words.
  at::Tensor words;
  // scalar_slots * count scalars: kFloat, kDouble, or kByte words of `width` bytes.
  at::Tensor scalars;
  uint64_t count;
  uint64_t index_slots;
  uint64_t scalar_slots;
  uint32_t format;
  uint64_t width;
  // `Precision.descriptor` of the scalars.
  std::string descriptor;
};

void finalize(void* pointer) { delete static_cast<tree*>(pointer); }

void foreach_reference(void*, b_lean_obj_arg) {}

lean_external_class* tree_class() {
  static lean_external_class* result = lean_register_external_class(finalize, foreach_reference);
  return result;
}

at::ScalarType scalar_type(uint32_t format) {
  return format == 0 ? at::kFloat : format == 1 ? at::kDouble : at::kByte;
}

void check_layout(uint32_t format, uint64_t width) {
  TORCH_CHECK(format <= 2 && width > 0, "tree: invalid scalar layout");
  TORCH_CHECK(format == 2 || width == c10::elementSize(scalar_type(format)),
              "tree: native scalar width mismatch");
}

uint64_t times(uint64_t a, uint64_t b) {
  uint64_t result;
  TORCH_CHECK(!__builtin_mul_overflow(a, b, &result), "tree: block size overflow");
  return result;
}

std::string string_of(b_lean_obj_arg text) {
  return std::string(lean_string_cstr(text), lean_string_size(text) - 1);
}

// The handle, after checking that it is read with the scalar layout it was uploaded with.
const tree& unbox(b_lean_obj_arg object, uint32_t format, b_lean_obj_arg descriptor,
                  uint64_t width) {
  TORCH_CHECK(!lean_is_scalar(object) && lean_is_external(object) &&
                  lean_get_external_class(object) == tree_class(),
              "tree: invalid handle");
  const auto& result = *static_cast<const tree*>(lean_get_external_data(object));
  TORCH_CHECK(result.format == format && result.width == width &&
                  result.descriptor == string_of(descriptor),
              "tree: the arena's scalars are in another layout");
  return result;
}

// A block of little-endian words as a device tensor whose elements are those words.
at::Tensor upload_block(b_lean_obj_arg bytes, at::ScalarType dtype) {
  const size_t size = c10::elementSize(dtype);
  const size_t total = lean_sarray_size(bytes);
  TORCH_CHECK(total % size == 0 && total / size <= INT64_MAX, "tree: truncated block");
  const size_t n = total / size;
  auto host = at::empty({static_cast<int64_t>(n)}, at::TensorOptions().dtype(dtype));
  const auto* source = reinterpret_cast<const uint8_t*>(lean_sarray_cptr(bytes));
  auto* values = static_cast<uint8_t*>(host.data_ptr());
  if (size == 1) {
    if (total) std::memcpy(values, source, total);
  } else for (size_t i = 0; i != n; ++i) {
    uint64_t bits = 0;
    for (size_t j = 0; j != size; ++j) bits |= uint64_t(source[size * i + j]) << (8 * j);
    if (size == 4) {
      const uint32_t word = static_cast<uint32_t>(bits);
      std::memcpy(values + size * i, &word, size);
    } else std::memcpy(values + size * i, &bits, size);
  }
  return host.to(torchlean::options().dtype(dtype), false, true);
}

// The elements of a device tensor as a block of little-endian words.
lean_obj_res download_block(const at::Tensor& block) {
  const size_t size = c10::elementSize(block.scalar_type());
  const size_t n = static_cast<size_t>(block.numel());
  const size_t total = times(n, size);
  const auto host = block.to(at::kCPU).contiguous();
  lean_object* out = lean_alloc_sarray(1, total, total);
  const auto* values = static_cast<const uint8_t*>(host.const_data_ptr());
  auto* destination = lean_sarray_cptr(out);
  if (size == 1) {
    if (total) std::memcpy(destination, values, total);
  } else for (size_t i = 0; i != n; ++i) {
    uint64_t bits = 0;
    if (size == 4) {
      uint32_t word;
      std::memcpy(&word, values + size * i, size);
      bits = word;
    } else std::memcpy(&bits, values + size * i, size);
    for (size_t j = 0; j != size; ++j)
      destination[size * i + j] = static_cast<uint8_t>(bits >> (8 * j));
  }
  return out;
}

// The device address of byte `offset` of a block. Offsets stay within the blocks, whose byte
// sizes are checked when they are made.
uint64_t address(const at::Tensor& block, uint64_t offset) {
  return reinterpret_cast<uint64_t>(static_cast<const uint8_t*>(block.const_data_ptr()) + offset);
}

// The input blocks of a generated kernel as device tensors of scalars, with their numbers of
// scalars. A configured scalar takes `stride` byte elements, a native one a single element.
struct inputs_on_device {
  std::vector<at::Tensor> blocks;
  std::vector<uint64_t> lengths;

  inputs_on_device(b_lean_obj_arg inputs, at::ScalarType dtype, uint64_t stride) {
    for (size_t i = 0; i != lean_array_size(inputs); ++i) {
      blocks.push_back(upload_block(lean_array_get_core(inputs, i), dtype));
      TORCH_CHECK(blocks.back().numel() % stride == 0, "tree: truncated input scalars");
      lengths.push_back(static_cast<uint64_t>(blocks.back().numel()) / stride);
    }
  }

  // The inputs as kernel arguments: a pointer and a number of scalars each.
  void push(torchlean::launch& entry) const {
    for (size_t i = 0; i != blocks.size(); ++i) {
      entry.arguments.push_back(address(blocks[i], 0));
      entry.arguments.push_back(lengths[i]);
    }
  }
};

}  // namespace

extern "C" LEAN_EXPORT lean_obj_res torchlean_tree_upload(
    uint32_t format, b_lean_obj_arg descriptor, uint64_t width, uint64_t count,
    uint64_t index_slots, uint64_t scalar_slots, b_lean_obj_arg words, b_lean_obj_arg scalars) {
  return torchlean::io([&] {
    check_layout(format, width);
    TORCH_CHECK(index_slots < UINT64_MAX, "tree: block size overflow");
    TORCH_CHECK(lean_sarray_size(words) == times(times(8, index_slots + 1), count),
                "tree: the words block does not match the descriptor");
    TORCH_CHECK(lean_sarray_size(scalars) == times(times(width, scalar_slots), count),
                "tree: the scalar block does not match the descriptor");
    auto result = std::make_unique<tree>(tree{upload_block(words, at::kLong),
        upload_block(scalars, scalar_type(format)), count, index_slots, scalar_slots, format,
        width, string_of(descriptor)});
    return lean_alloc_external(tree_class(), result.release());
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_tree_words(
    b_lean_obj_arg handle, uint32_t format, b_lean_obj_arg descriptor, uint64_t width) {
  return torchlean::io([&] {
    return download_block(unbox(handle, format, descriptor, width).words);
  });
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_tree_scalars(
    b_lean_obj_arg handle, uint32_t format, b_lean_obj_arg descriptor, uint64_t width) {
  return torchlean::io([&] {
    return download_block(unbox(handle, format, descriptor, width).scalars);
  });
}

// Fold the levels of a handle from the deepest up with the generated row kernel `source`. Level r
// is one launch over its rows. Kernel column p is column p of the level (the tag, the index
// columns, then the scalar columns), input i < inputs.size() is block i of `inputs`, the last
// input holds the results of level r + 1, and the output takes the results of level r. Returns
// the results of level zero as little-endian scalar words, or none when a read failed.
extern "C" LEAN_EXPORT lean_obj_res torchlean_tree_fold(
    b_lean_obj_arg handle, uint32_t format, b_lean_obj_arg descriptor, uint64_t width,
    uint64_t index_slots, uint64_t scalar_slots, b_lean_obj_arg source, b_lean_obj_arg sizes,
    b_lean_obj_arg inputs) {
  return torchlean::io([&]() -> lean_obj_res {
    const tree& arena = unbox(handle, format, descriptor, width);
    TORCH_CHECK(arena.index_slots == index_slots && arena.scalar_slots == scalar_slots,
                "tree fold: the arena has other slots than the fold");
    TORCH_CHECK(arena.words.device() == torchlean::device() &&
                    arena.scalars.device() == torchlean::device(),
                "tree fold: the arena is on another device");
    const size_t levels = lean_array_size(sizes);
    std::vector<uint64_t> size(levels), base(levels + 1, 0);
    for (size_t r = 0; r != levels; ++r) {
      size[r] = lean_unbox_uint64(lean_array_get_core(sizes, r));
      TORCH_CHECK(!__builtin_add_overflow(base[r], size[r], &base[r + 1]),
                  "tree fold: level sizes overflow");
    }
    TORCH_CHECK(base[levels] == arena.count,
                "tree fold: the level sizes do not add up to the rows");
    const auto dtype = scalar_type(format);
    // A configured scalar takes `width` byte elements, a native one a single element.
    const uint64_t stride = format == 2 ? width : 1;
    const inputs_on_device blocks(inputs, dtype, stride);
    // Zeros, so that a failed row, which writes no result, leaves a defined word.
    auto results = at::zeros({static_cast<int64_t>(times(arena.count, stride))},
                             torchlean::options().dtype(dtype));
    auto error = at::zeros({4}, torchlean::options().dtype(at::kLong));
    std::vector<torchlean::launch> launches;
    for (size_t r = levels; r-- != 0;) {
      torchlean::launch entry{size[r], {}};
      for (uint64_t p = 0; p <= index_slots; ++p)
        entry.arguments.push_back(address(arena.words, 8 * (p * arena.count + base[r])));
      for (uint64_t j = 0; j != scalar_slots; ++j)
        entry.arguments.push_back(address(arena.scalars, width * (j * arena.count + base[r])));
      blocks.push(entry);
      entry.arguments.push_back(address(results, width * base[r + 1]));
      entry.arguments.push_back(r + 1 < levels ? size[r + 1] : 0);
      entry.arguments.push_back(address(results, width * base[r]));
      entry.arguments.push_back(address(error, 0));
      entry.arguments.push_back(size[r]);
      launches.push_back(std::move(entry));
    }
    torchlean::launch_rows(string_of(source), launches);
    const auto record = error.to(at::kCPU);
    if (record.const_data_ptr<int64_t>()[0] != 0) return lean_box(0);
    const uint64_t roots = levels ? size[0] : 0;
    // The download may throw, so the option cell is allocated after it.
    lean_object* bytes =
        download_block(results.narrow(0, 0, static_cast<int64_t>(times(roots, stride))));
    lean_object* some = lean_alloc_ctor(1, 1, 0);
    lean_ctor_set(some, 0, bytes);
    return some;
  });
}

// Unfold the calls of a hylomorphism from `count` roots, one level per depth of the recursion.
// The roots are the columns of the blocks `words`, whose tag column is not read, and `scalars`.
// For each level, kernel `calls` reads the slots of every row (the index columns, then the scalar
// columns) and writes its number of calls. An exclusive prefix sum of the numbers gives the first
// call of every row, which becomes its tag, and kernel p of `arguments` reads the number of every
// call within its row, then the slots of the row, and writes slot p of the call. Block i of
// `inputs` is input i of every kernel. Without node kernels, returns the handle of all levels, root
// level first, with their sizes. With node kernels, kernel p of `nodes` reads the first call of
// every row, then the slots of the row, and writes column p of the node of the row (its tag, its
// `node_index_slots` index slots, then its `node_scalar_slots` scalar slots), and the handle holds
// these columns instead. Returns none when a read failed or the levels would hold more than
// `capacity` rows.
extern "C" LEAN_EXPORT lean_obj_res torchlean_tree_expand(
    uint32_t format, b_lean_obj_arg descriptor, uint64_t width, uint64_t index_slots,
    uint64_t scalar_slots, uint64_t count, b_lean_obj_arg words, b_lean_obj_arg scalars,
    b_lean_obj_arg calls, b_lean_obj_arg arguments, b_lean_obj_arg nodes,
    uint64_t node_index_slots, uint64_t node_scalar_slots, b_lean_obj_arg inputs,
    uint64_t capacity) {
  return torchlean::io([&]() -> lean_obj_res {
    check_layout(format, width);
    TORCH_CHECK(capacity < (uint64_t(1) << 31), "tree expand: the capacity is too large");
    TORCH_CHECK(index_slots < UINT64_MAX, "tree: block size overflow");
    TORCH_CHECK(lean_sarray_size(words) == times(times(8, index_slots + 1), count),
                "tree expand: the words block does not match the roots");
    TORCH_CHECK(lean_sarray_size(scalars) == times(times(width, scalar_slots), count),
                "tree expand: the scalar block does not match the roots");
    uint64_t slots;
    TORCH_CHECK(!__builtin_add_overflow(index_slots, scalar_slots, &slots) &&
                    lean_array_size(arguments) == slots,
                "tree expand: there is not one kernel per slot");
    const bool build = lean_array_size(nodes) != 0;
    uint64_t columns = 0;
    TORCH_CHECK(!build || (!__builtin_add_overflow(node_index_slots, node_scalar_slots, &columns) &&
                           columns < UINT64_MAX && lean_array_size(nodes) == columns + 1),
                "tree expand: there is not one node kernel per column");
    // The slots of the rows of the handle: those of the nodes, or those of the calls.
    const uint64_t out_index = build ? node_index_slots : index_slots;
    const uint64_t out_scalar = build ? node_scalar_slots : scalar_slots;
    if (count > capacity) return lean_box(0);
    const auto dtype = scalar_type(format);
    const uint64_t stride = format == 2 ? width : 1;
    const auto index_options = torchlean::options().dtype(at::kLong);
    const auto scalar_options = torchlean::options().dtype(dtype);
    const inputs_on_device blocks(inputs, dtype, stride);
    const std::string calls_source = string_of(calls);
    std::vector<std::string> sources;
    for (uint64_t p = 0; p != slots; ++p)
      sources.push_back(string_of(lean_array_get_core(arguments, p)));
    std::vector<std::string> node_sources;
    for (size_t p = 0; p != lean_array_size(nodes); ++p)
      node_sources.push_back(string_of(lean_array_get_core(nodes, p)));
    // Each kernel is compiled at its first launch and held for the later levels, so that the
    // module cache does not compile the kernels of one expansion again from level to level.
    torchlean::rows_kernel calls_kernel;
    std::vector<torchlean::rows_kernel> kernels(sources.size()), node_kernels(node_sources.size());
    const auto compiled = [](torchlean::rows_kernel& kernel,
                             const std::string& source) -> const torchlean::rows_kernel& {
      if (!kernel) kernel = torchlean::compile_rows(source);
      return kernel;
    };
    // The slot columns of the frontier, at first the roots.
    const auto root_words = upload_block(words, at::kLong);
    const auto root_scalars = upload_block(scalars, dtype);
    const auto rows = static_cast<int64_t>(count);
    std::vector<at::Tensor> index, scalar;
    for (uint64_t i = 0; i != index_slots; ++i)
      index.push_back(root_words.narrow(0, static_cast<int64_t>(i + 1) * rows, rows));
    for (uint64_t j = 0; j != scalar_slots; ++j)
      scalar.push_back(root_scalars.narrow(0, static_cast<int64_t>(j * stride) * rows,
                                           rows * static_cast<int64_t>(stride)));
    // The columns of every level, and their sizes.
    std::vector<at::Tensor> tags;
    std::vector<std::vector<at::Tensor>> index_levels(out_index), scalar_levels(out_scalar);
    std::vector<uint64_t> sizes;
    auto error = at::zeros({4}, index_options);
    uint64_t budget = capacity - count, n = count, total_rows = 0;
    while (n != 0) {
      const auto frontier = static_cast<int64_t>(n);
      // Zeroed, since a failed read stops the threads before they write their numbers.
      auto counts = at::zeros({frontier}, index_options);
      torchlean::launch entry{n, {}};
      for (const auto& column : index) entry.arguments.push_back(address(column, 0));
      for (const auto& column : scalar) entry.arguments.push_back(address(column, 0));
      blocks.push(entry);
      entry.arguments.push_back(address(counts, 0));
      entry.arguments.push_back(address(error, 0));
      entry.arguments.push_back(n);
      std::vector<torchlean::launch> launches{std::move(entry)};
      torchlean::launch_rows(compiled(calls_kernel, calls_source), launches);
      // One copy to the host per level: the error record, which also records a failed read of
      // the slot kernels of the level above, whether a number of calls is out of range (as a
      // signed word, a number of 2^63 or more is negative), and the total.
      const auto limit = static_cast<int64_t>(budget);
      const auto summary =
          at::stack({error[0], counts.lt(0).logical_or(counts.gt(limit)).any().to(at::kLong),
                     counts.sum()})
              .to(at::kCPU);
      const int64_t* record = summary.const_data_ptr<int64_t>();
      if (record[0] != 0 || record[1] != 0 || record[2] > limit) return lean_box(0);
      const uint64_t total = static_cast<uint64_t>(record[2]);
      const auto firsts = counts.cumsum(0).sub_(counts);
      if (build) {
        // Column p of the nodes: the tag for p = 0, then the index columns, then the scalar
        // columns.
        for (size_t p = 0; p != node_sources.size(); ++p) {
          const bool word = p <= out_index;
          auto output = word ? at::empty({frontier}, index_options)
                             : at::empty({frontier * static_cast<int64_t>(stride)}, scalar_options);
          torchlean::launch node{n, {}};
          node.arguments.push_back(address(firsts, 0));
          for (const auto& column : index) node.arguments.push_back(address(column, 0));
          for (const auto& column : scalar) node.arguments.push_back(address(column, 0));
          blocks.push(node);
          node.arguments.push_back(address(output, 0));
          node.arguments.push_back(address(error, 0));
          node.arguments.push_back(n);
          std::vector<torchlean::launch> node_launches{std::move(node)};
          torchlean::launch_rows(compiled(node_kernels[p], node_sources[p]), node_launches);
          if (p == 0) tags.push_back(std::move(output));
          else if (word) index_levels[p - 1].push_back(std::move(output));
          else scalar_levels[p - 1 - out_index].push_back(std::move(output));
        }
      } else {
        tags.push_back(firsts);
        for (uint64_t i = 0; i != index_slots; ++i) index_levels[i].push_back(index[i]);
        for (uint64_t j = 0; j != scalar_slots; ++j) scalar_levels[j].push_back(scalar[j]);
      }
      sizes.push_back(n);
      total_rows += n;
      if (total == 0) break;
      // Call c of the level is call `call[c]` of row `parent[c]`.
      const auto calls_count = static_cast<int64_t>(total);
      const auto parent = at::repeat_interleave(counts, std::optional<int64_t>(calls_count));
      const auto call = at::arange(calls_count, index_options).sub_(firsts.index_select(0, parent));
      std::vector<at::Tensor> parent_index, parent_scalar;
      for (const auto& column : index) parent_index.push_back(column.index_select(0, parent));
      for (const auto& column : scalar)
        parent_scalar.push_back(stride == 1 ? column.index_select(0, parent)
                                            : column.view({frontier, static_cast<int64_t>(stride)})
                                                  .index_select(0, parent)
                                                  .reshape({-1}));
      std::vector<at::Tensor> next_index, next_scalar;
      for (uint64_t p = 0; p != slots; ++p) {
        auto output = p < index_slots
                          ? at::empty({calls_count}, index_options)
                          : at::empty({calls_count * static_cast<int64_t>(stride)}, scalar_options);
        torchlean::launch slot{total, {}};
        slot.arguments.push_back(address(call, 0));
        for (const auto& column : parent_index) slot.arguments.push_back(address(column, 0));
        for (const auto& column : parent_scalar) slot.arguments.push_back(address(column, 0));
        blocks.push(slot);
        slot.arguments.push_back(address(output, 0));
        slot.arguments.push_back(address(error, 0));
        slot.arguments.push_back(total);
        std::vector<torchlean::launch> slot_launches{std::move(slot)};
        torchlean::launch_rows(compiled(kernels[p], sources[p]), slot_launches);
        (p < index_slots ? next_index : next_scalar).push_back(std::move(output));
      }
      budget -= total;
      n = total;
      index = std::move(next_index);
      scalar = std::move(next_scalar);
    }
    // A failed read of the node kernels of the last level is recorded after the last copy.
    if (build && error[0].item<int64_t>() != 0) return lean_box(0);
    // The blocks of all levels: the tags, then each index column, and each scalar column.
    auto all_words = at::empty({0}, index_options);
    auto all_scalars = at::empty({0}, scalar_options);
    if (!sizes.empty()) {
      std::vector<at::Tensor> parts{at::cat(tags)};
      for (const auto& column : index_levels) parts.push_back(at::cat(column));
      all_words = at::cat(parts);
      std::vector<at::Tensor> pieces;
      for (const auto& column : scalar_levels) pieces.push_back(at::cat(column));
      if (!pieces.empty()) all_scalars = at::cat(pieces);
    }
    auto result = std::make_unique<tree>(tree{std::move(all_words), std::move(all_scalars),
        total_rows, out_index, out_scalar, format, width, string_of(descriptor)});
    // The Lean objects come last, after every call that may throw.
    lean_object* level_sizes = lean_alloc_array(sizes.size(), sizes.size());
    for (size_t r = 0; r != sizes.size(); ++r)
      lean_array_set_core(level_sizes, r, lean_box_uint64(sizes[r]));
    lean_object* handle = lean_alloc_external(tree_class(), result.release());
    lean_object* some = lean_alloc_ctor(1, 1, 0);
    lean_ctor_set(some, 0, torchlean::cons(handle, level_sizes));
    return some;
  });
}
