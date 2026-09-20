#include <lean/lean.h>

#include <stdint.h>
#include <string.h>

#if defined(__GNUC__) || defined(__clang__)
#define TORCHLEAN_WEAK __attribute__((weak))
#else
#define TORCHLEAN_WEAK
#endif

static size_t torchlean_clamped_nat(b_lean_obj_arg value, size_t upper) {
  if (!lean_is_scalar(value)) {
    return upper;
  }
  size_t result = lean_unbox(value);
  return result < upper ? result : upper;
}

static size_t torchlean_grown_capacity(size_t current, size_t needed) {
  size_t capacity = current < 4 ? 4 : current;
  while (capacity < needed) {
    if (capacity > SIZE_MAX / 2) {
      return needed;
    }
    capacity *= 2;
  }
  return capacity;
}

/*
 * Append source[start:stop] to output.
 *
 * The Lean declaration keeps `source`, `start`, and `stop` borrowed and owns
 * `output`. Bounds follow `Array.extract`: oversized natural numbers clamp to
 * the source length, including heap-allocated big naturals.
 */
LEAN_EXPORT lean_obj_res torchlean_float_array_append_slice(
    b_lean_obj_arg source, b_lean_obj_arg start_obj,
    b_lean_obj_arg stop_obj, lean_obj_arg output) {
  size_t source_size = lean_sarray_size(source);
  size_t start = torchlean_clamped_nat(start_obj, source_size);
  size_t stop = torchlean_clamped_nat(stop_obj, source_size);
  if (stop <= start) {
    return output;
  }

  size_t length = stop - start;
  size_t output_size = lean_sarray_size(output);
  if (output_size > SIZE_MAX - length) {
    lean_internal_panic_out_of_memory();
  }
  size_t needed = output_size + length;

  if (lean_is_exclusive(output) &&
      lean_sarray_capacity(output) >= needed) {
    double *output_data = lean_float_array_cptr(output);
    const double *source_data = lean_float_array_cptr(source);
    memmove(output_data + output_size, source_data + start,
            length * sizeof(double));
    lean_sarray_set_size(output, needed);
    return output;
  }

  size_t capacity =
      torchlean_grown_capacity(lean_sarray_capacity(output), needed);
  if (lean_alloc_sarray_would_overflow(sizeof(double), capacity)) {
    lean_internal_panic_out_of_memory();
  }
  lean_object *result =
      lean_alloc_sarray(sizeof(double), needed, capacity);
  double *result_data = lean_float_array_cptr(result);
  const double *output_data = lean_float_array_cptr(output);
  const double *source_data = lean_float_array_cptr(source);
  memcpy(result_data, output_data, output_size * sizeof(double));
  memcpy(result_data + output_size, source_data + start,
         length * sizeof(double));
  lean_dec(output);
  return result;
}

/*
 * Transpose one row-major [rows, columns] FloatArray into
 * [columns, rows].
 *
 * Blocking keeps both the source reads and output writes within a small
 * working set. The Lean caller proves that rows * columns is exactly the
 * source length, so the checks below only defend the FFI boundary.
 */
LEAN_EXPORT lean_obj_res torchlean_float_array_transpose2d(
    b_lean_obj_arg source, b_lean_obj_arg rows_obj,
    b_lean_obj_arg columns_obj) {
  size_t rows = lean_usize_of_nat(rows_obj);
  size_t columns = lean_usize_of_nat(columns_obj);
  size_t source_size = lean_sarray_size(source);

  if (rows != 0 && columns > SIZE_MAX / rows) {
    lean_internal_panic("torchlean FloatArray transpose dimensions overflow");
  }
  size_t output_size = rows * columns;
  if (output_size != source_size) {
    lean_internal_panic("torchlean FloatArray transpose size mismatch");
  }
  if (lean_alloc_sarray_would_overflow(sizeof(double), output_size)) {
    lean_internal_panic_out_of_memory();
  }

  lean_object *result =
      lean_alloc_sarray(sizeof(double), output_size, output_size);
  /* A zero column count permits arbitrarily large row dimensions. */
  if (output_size == 0) {
    return result;
  }
  const double *source_data = lean_float_array_cptr(source);
  double *result_data = lean_float_array_cptr(result);
  /*
   * A 32-wide tile thrashes L1 cache sets at common power-of-two strides.
   * Sixteen is best for small matrices; eight keeps the larger-stride case
   * below the cache associativity limit.
   */
  const size_t tile = (rows >= 256 || columns >= 256) ? 8 : 16;

  for (size_t row_block = 0; row_block < rows; row_block += tile) {
    size_t row_stop = row_block + tile;
    if (row_stop > rows) {
      row_stop = rows;
    }
    for (size_t column_block = 0; column_block < columns;
         column_block += tile) {
      size_t column_stop = column_block + tile;
      if (column_stop > columns) {
        column_stop = columns;
      }
      for (size_t row = row_block; row < row_stop; ++row) {
        for (size_t column = column_block; column < column_stop; ++column) {
          result_data[column * rows + row] =
              source_data[row * columns + column];
        }
      }
    }
  }
  return result;
}

/*
 * Transpose one row-major ordinary Lean Array into [columns, rows].
 *
 * Generic tensor storage is an Array of retained Lean objects. Moving the
 * pointers directly avoids scalar unboxing/reboxing and preserves exact
 * values such as rationals and bit-level IEEE records. A uniquely owned square
 * array is transposed in place. Other uniquely owned arrays transfer their
 * element ownership to the output; shared arrays retain each copied element.
 */
LEAN_EXPORT lean_obj_res torchlean_array_transpose2d(
    lean_obj_arg source, b_lean_obj_arg rows_obj,
    b_lean_obj_arg columns_obj) {
  size_t rows = lean_usize_of_nat(rows_obj);
  size_t columns = lean_usize_of_nat(columns_obj);
  size_t source_size = lean_array_size(source);

  if (rows != 0 && columns > SIZE_MAX / rows) {
    lean_internal_panic("torchlean Array transpose dimensions overflow");
  }
  size_t output_size = rows * columns;
  if (output_size != source_size) {
    lean_internal_panic("torchlean Array transpose size mismatch");
  }

  /* Empty rectangles need no traversal, regardless of either dimension. */
  if (output_size == 0) {
    return source;
  }

  const size_t tile = (rows >= 256 || columns >= 256) ? 8 : 16;
  if (rows == columns && lean_is_exclusive(source)) {
    lean_object **data = lean_array_cptr(source);
    for (size_t row_block = 0; row_block < rows; row_block += tile) {
      size_t row_stop = row_block + tile;
      if (row_stop > rows) {
        row_stop = rows;
      }
      for (size_t column_block = row_block; column_block < columns;
           column_block += tile) {
        size_t column_stop = column_block + tile;
        if (column_stop > columns) {
          column_stop = columns;
        }
        for (size_t row = row_block; row < row_stop; ++row) {
          size_t column_start = column_block;
          if (column_block == row_block && column_start <= row) {
            column_start = row + 1;
          }
          for (size_t column = column_start; column < column_stop; ++column) {
            lean_object *value = data[row * columns + column];
            data[row * columns + column] = data[column * rows + row];
            data[column * rows + row] = value;
          }
        }
      }
    }
    return source;
  }

  bool move_elements = lean_is_exclusive(source);
  lean_object *result = lean_alloc_array(output_size, output_size);
  lean_object **source_data = lean_array_cptr(source);
  lean_object **result_data = lean_array_cptr(result);

  if (move_elements) {
    for (size_t row_block = 0; row_block < rows; row_block += tile) {
      size_t row_stop = row_block + tile;
      if (row_stop > rows) {
        row_stop = rows;
      }
      for (size_t column_block = 0; column_block < columns;
           column_block += tile) {
        size_t column_stop = column_block + tile;
        if (column_stop > columns) {
          column_stop = columns;
        }
        for (size_t row = row_block; row < row_stop; ++row) {
          for (size_t column = column_block; column < column_stop; ++column) {
            result_data[column * rows + row] =
                source_data[row * columns + column];
          }
        }
      }
    }
    lean_free_object(source);
  } else {
    for (size_t row_block = 0; row_block < rows; row_block += tile) {
      size_t row_stop = row_block + tile;
      if (row_stop > rows) {
        row_stop = rows;
      }
      for (size_t column_block = 0; column_block < columns;
           column_block += tile) {
        size_t column_stop = column_block + tile;
        if (column_stop > columns) {
          column_stop = columns;
        }
        for (size_t row = row_block; row < row_stop; ++row) {
          for (size_t column = column_block; column < column_stop; ++column) {
            lean_object *value = source_data[row * columns + column];
            lean_inc(value);
            result_data[column * rows + row] = value;
          }
        }
      }
    }
    lean_dec(source);
  }
  return result;
}

/* Allocate the output of Array.zipWith, which truncates to the shorter input.
 * Tensor callers have matching shapes, but the safe Lean buffer primitives
 * also accept unequal lengths and their native bodies must preserve that model.
 */
static lean_obj_res torchlean_float_zip_output(
    b_lean_obj_arg left, b_lean_obj_arg right) {
  size_t left_size = lean_sarray_size(left);
  size_t right_size = lean_sarray_size(right);
  size_t size = left_size < right_size ? left_size : right_size;
  if (lean_alloc_sarray_would_overflow(sizeof(double), size)) {
    lean_internal_panic_out_of_memory();
  }
  return lean_alloc_sarray(sizeof(double), size, size);
}

/* Independent IEEE operations retain scalar order and permit vectorization. */
LEAN_EXPORT lean_obj_res torchlean_float_array_add(
    b_lean_obj_arg left, b_lean_obj_arg right) {
  lean_object *result = torchlean_float_zip_output(left, right);
  size_t output_size = lean_sarray_size(result);
  const double *left_data = lean_float_array_cptr(left);
  const double *right_data = lean_float_array_cptr(right);
  double *result_data = lean_float_array_cptr(result);
  for (size_t index = 0; index < output_size; ++index) {
    result_data[index] = left_data[index] + right_data[index];
  }
  return result;
}

LEAN_EXPORT lean_obj_res torchlean_float_array_sub(
    b_lean_obj_arg left, b_lean_obj_arg right) {
  lean_object *result = torchlean_float_zip_output(left, right);
  size_t output_size = lean_sarray_size(result);
  const double *left_data = lean_float_array_cptr(left);
  const double *right_data = lean_float_array_cptr(right);
  double *result_data = lean_float_array_cptr(result);
  for (size_t index = 0; index < output_size; ++index) {
    result_data[index] = left_data[index] - right_data[index];
  }
  return result;
}

LEAN_EXPORT lean_obj_res torchlean_float_array_mul(
    b_lean_obj_arg left, b_lean_obj_arg right) {
  lean_object *result = torchlean_float_zip_output(left, right);
  size_t output_size = lean_sarray_size(result);
  const double *left_data = lean_float_array_cptr(left);
  const double *right_data = lean_float_array_cptr(right);
  double *result_data = lean_float_array_cptr(result);
  for (size_t index = 0; index < output_size; ++index) {
    result_data[index] = left_data[index] * right_data[index];
  }
  return result;
}

LEAN_EXPORT lean_obj_res torchlean_float_array_div(
    b_lean_obj_arg left, b_lean_obj_arg right) {
  lean_object *result = torchlean_float_zip_output(left, right);
  size_t output_size = lean_sarray_size(result);
  const double *left_data = lean_float_array_cptr(left);
  const double *right_data = lean_float_array_cptr(right);
  double *result_data = lean_float_array_cptr(result);
  for (size_t index = 0; index < output_size; ++index) {
    result_data[index] = left_data[index] / right_data[index];
  }
  return result;
}

/*
 * Promote packed bytes and add them to packed doubles in one output loop.
 *
 * UInt8 values convert exactly to IEEE binary64. Keeping separate functions
 * for the two operand orders preserves the scalar Float expression, including
 * implementation-defined NaN payload selection.
 */
LEAN_EXPORT lean_obj_res torchlean_byte_float_array_add(
    b_lean_obj_arg left, b_lean_obj_arg right) {
  lean_object *result = torchlean_float_zip_output(left, right);
  size_t output_size = lean_sarray_size(result);
  const uint8_t *left_data = lean_sarray_cptr(left);
  const double *right_data = lean_float_array_cptr(right);
  double *result_data = lean_float_array_cptr(result);
  for (size_t index = 0; index < output_size; ++index) {
    result_data[index] = (double)left_data[index] + right_data[index];
  }
  return result;
}

LEAN_EXPORT lean_obj_res torchlean_float_byte_array_add(
    b_lean_obj_arg left, b_lean_obj_arg right) {
  lean_object *result = torchlean_float_zip_output(left, right);
  size_t output_size = lean_sarray_size(result);
  const double *left_data = lean_float_array_cptr(left);
  const uint8_t *right_data = lean_sarray_cptr(right);
  double *result_data = lean_float_array_cptr(result);
  for (size_t index = 0; index < output_size; ++index) {
    result_data[index] = left_data[index] + (double)right_data[index];
  }
  return result;
}

/*
 * Lean's evaluator resolves an external declaration through the generated
 * boxed wrapper name, while compiled modules generate these wrappers
 * themselves. Weak definitions make the wrappers available from the package
 * dynamic library without conflicting with the generated strong definitions
 * in ordinary executables.
 */
LEAN_EXPORT TORCHLEAN_WEAK lean_obj_res
lp_TorchLean_TorchLean_Storage_Internal_floatBufferAppendSliceNative___boxed(
    lean_obj_arg source, lean_obj_arg start, lean_obj_arg stop,
    lean_obj_arg output) {
  lean_object *result =
      torchlean_float_array_append_slice(source, start, stop, output);
  lean_dec(stop);
  lean_dec(start);
  lean_dec_ref(source);
  return result;
}

LEAN_EXPORT TORCHLEAN_WEAK lean_obj_res
lp_TorchLean_TorchLean_Tensor_Internal_Elab_Impl_floatBufferTranspose2DNative___boxed(
    lean_obj_arg source, lean_obj_arg rows, lean_obj_arg columns,
    lean_obj_arg h_size) {
  (void)h_size;
  lean_object *result =
      torchlean_float_array_transpose2d(source, rows, columns);
  lean_dec(columns);
  lean_dec(rows);
  lean_dec_ref(source);
  return result;
}

LEAN_EXPORT TORCHLEAN_WEAK lean_obj_res
lp_TorchLean_TorchLean_Tensor_Internal_Elab_Impl_arrayBufferTranspose2DNative___boxed(
    lean_obj_arg type, lean_obj_arg source, lean_obj_arg rows,
    lean_obj_arg columns, lean_obj_arg h_size) {
  (void)type;
  (void)h_size;
  lean_object *result = torchlean_array_transpose2d(source, rows, columns);
  lean_dec(columns);
  lean_dec(rows);
  return result;
}

#define TORCHLEAN_BINARY_EVAL_WRAPPER(wrapper, kernel)                    \
  LEAN_EXPORT TORCHLEAN_WEAK lean_obj_res wrapper(                        \
      lean_obj_arg left, lean_obj_arg right) {                            \
    lean_object *result = kernel(left, right);                            \
    lean_dec_ref(right);                                                  \
    lean_dec_ref(left);                                                   \
    return result;                                                        \
  }

TORCHLEAN_BINARY_EVAL_WRAPPER(
    lp_TorchLean_TorchLean_Tensor_Internal_Elab_Impl_floatBufferAddNative___boxed,
    torchlean_float_array_add)
TORCHLEAN_BINARY_EVAL_WRAPPER(
    lp_TorchLean_TorchLean_Tensor_Internal_Elab_Impl_floatBufferSubNative___boxed,
    torchlean_float_array_sub)
TORCHLEAN_BINARY_EVAL_WRAPPER(
    lp_TorchLean_TorchLean_Tensor_Internal_Elab_Impl_floatBufferMulNative___boxed,
    torchlean_float_array_mul)
TORCHLEAN_BINARY_EVAL_WRAPPER(
    lp_TorchLean_TorchLean_Tensor_Internal_Elab_Impl_floatBufferDivNative___boxed,
    torchlean_float_array_div)
TORCHLEAN_BINARY_EVAL_WRAPPER(
    lp_TorchLean_TorchLean_Tensor_Internal_Elab_Impl_byteFloatBufferAddNative___boxed,
    torchlean_byte_float_array_add)
TORCHLEAN_BINARY_EVAL_WRAPPER(
    lp_TorchLean_TorchLean_Tensor_Internal_Elab_Impl_floatByteBufferAddNative___boxed,
    torchlean_float_byte_array_add)

#undef TORCHLEAN_BINARY_EVAL_WRAPPER
#undef TORCHLEAN_WEAK
