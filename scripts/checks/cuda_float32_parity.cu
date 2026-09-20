// Device-side parity check for the five fields of `NativePrimitiveAgreement`.
//
// TorchLean proves half-ULP error bounds about `IEEE32Exec`, its executable binary32 reference
// model, and then assumes that the machine which actually ran a kernel returns the same bits. That
// assumption is `Runtime.Autograd.Cuda.Float32Contract.NativePrimitiveAgreement`, and its five
// fields are `add_bits`, `mul_bits`, `div_bits`, `fma_bits` and `sqrt_bits`. Lean cannot prove any
// of them.
// This program checks them on one GPU, one driver and one compiler, which is the strongest thing a
// test can say about an assumption of this shape.
//
// It reads cases on stdin, one per line, in the format written by
//
//   lake exe native_float32_parity --emit-cases
//
//   add  0xAAAAAAAA 0xBBBBBBBB 0xEEEEEEEE
//   mul  0xAAAAAAAA 0xBBBBBBBB 0xEEEEEEEE
//   div  0xAAAAAAAA 0xBBBBBBBB 0xEEEEEEEE
//   sqrt 0xAAAAAAAA 0xEEEEEEEE
//   fma  0xAAAAAAAA 0xBBBBBBBB 0xCCCCCCCC 0xEEEEEEEE
//
// where the last field on each line is the bit pattern `IEEE32Exec` returns. Every operand and
// result travels as a bit pattern rather than as a decimal literal, so nothing in the comparison
// depends on how a printf or a strtof rounds.
//
// The device side uses the `__*_rn` intrinsics on purpose. Writing `x + y` would let the compiler
// contract a multiply and an add into a single fused instruction, which is exactly one of the ways
// the contract can silently fail; asking for round-to-nearest explicitly is what a kernel author
// who cares about the assumption should do.
//
// Two verdicts come out of every run, and the difference between them is the whole point.
//
//   strict    the native bits equal the reference bits, with no exceptions
//   contract  the native bits equal the reference bits, or both patterns are quiet `NaN`s
//
// The second is what `NativePrimitiveAgreement` now asks for, via `AgreeUpToNaN`, and the reason is
// visible in this program's own output: an invalid operation such as `(-0)/(+0)` returns
// `0x7fc00000` in `IEEE32Exec`, `0xffc00000` on this x86-64 host and `0x7fffffff` on an A100. All
// three are quiet `NaN`s. IEEE 754-2019 clause 6.2 leaves the sign and the payload of a `NaN`
// produced by an invalid operation to the implementation, so strict equality is a contract that no
// provider on this machine satisfies, while up to `NaN` is one that all three do. Only the contract
// verdict decides the exit status; the strict count is printed because it is the interesting number
// when it is not equal to the total.
//
// Build and run through scripts/checks/cuda_float32_parity.sh.

#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

enum OpKind { kAdd = 0, kMul = 1, kDiv = 2, kSqrt = 3, kFma = 4 };

constexpr int kOpCount = 5;

const char *kOpNames[kOpCount] = {"add", "mul", "div", "sqrt", "fma"};

struct ParityCase {
  int op;
  uint32_t x;
  uint32_t y;
  uint32_t z;
  uint32_t expected;
};

// Host bit casts. `memcpy` rather than a union or a pointer cast, so that strict aliasing stays
// satisfied and the compiler is free to keep the value in a register.
float bits_to_float(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof value);
  return value;
}

uint32_t float_to_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof bits);
  return bits;
}

// The host reference, using the C library on this machine.
uint32_t host_apply(const ParityCase &c) {
  const float x = bits_to_float(c.x);
  const float y = bits_to_float(c.y);
  const float z = bits_to_float(c.z);
  switch (c.op) {
    case kAdd:
      return float_to_bits(x + y);
    case kMul:
      return float_to_bits(x * y);
    case kDiv:
      return float_to_bits(x / y);
    case kSqrt:
      return float_to_bits(sqrtf(x));
    case kFma:
      return float_to_bits(fmaf(x, y, z));
    default:
      return 0;
  }
}

__global__ void device_apply(const ParityCase *cases, uint32_t *results, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const float x = __uint_as_float(cases[i].x);
  const float y = __uint_as_float(cases[i].y);
  const float z = __uint_as_float(cases[i].z);
  float result = 0.0f;
  switch (cases[i].op) {
    case kAdd:
      result = __fadd_rn(x, y);
      break;
    case kMul:
      result = __fmul_rn(x, y);
      break;
    case kDiv:
      result = __fdiv_rn(x, y);
      break;
    case kSqrt:
      result = __fsqrt_rn(x);
      break;
    case kFma:
      result = __fmaf_rn(x, y, z);
      break;
    default:
      break;
  }
  results[i] = __float_as_uint(result);
}

// A quiet or signaling `NaN`: exponent all ones and a nonzero significand.
bool is_nan_bits(uint32_t bits) {
  return (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u;
}

// The contract of `AgreeUpToNaN`: equal bits, or two patterns that both mean "not a number".
bool agrees_up_to_nan(uint32_t native, uint32_t reference) {
  return native == reference || (is_nan_bits(native) && is_nan_bits(reference));
}

struct Tally {
  int checked = 0;
  int host_strict_failed = 0;
  int device_strict_failed = 0;
  int host_contract_failed = 0;
  int device_contract_failed = 0;
};

// The distinct `NaN` encodings a provider produced, in first-seen order. Kept so the run can say
// what the payloads actually were instead of only that they differed.
struct NaNEncodings {
  std::vector<uint32_t> seen;

  void note(uint32_t bits) {
    if (!is_nan_bits(bits)) {
      return;
    }
    for (uint32_t already : seen) {
      if (already == bits) {
        return;
      }
    }
    if (seen.size() < 8) {
      seen.push_back(bits);
    }
  }
};

void print_encodings(const char *who, const NaNEncodings &e) {
  std::printf("  %-7s", who);
  if (e.seen.empty()) {
    std::printf(" none\n");
    return;
  }
  for (uint32_t bits : e.seen) {
    std::printf(" 0x%08x", bits);
  }
  std::printf("\n");
}

bool parse_hex(const std::string &token, uint32_t *out) {
  // Case files contain exactly one unsigned binary32 bit pattern per token.
  const size_t start =
      (token.compare(0, 2, "0x") == 0 || token.compare(0, 2, "0X") == 0) ? 2 : 0;
  if (token.size() <= start || token.size() - start > 8 ||
      token.find_first_not_of("0123456789abcdefABCDEF", start) != std::string::npos) {
    return false;
  }
  *out = static_cast<uint32_t>(std::stoul(token, nullptr, 16));
  return true;
}

// One line of stdin. Returns false on a line this program does not understand, which is a hard
// error rather than a skip: a silently ignored case is a parity check that passes for the wrong
// reason.
bool parse_line(char *line, ParityCase *out) {
  std::vector<std::string> tokens;
  for (char *token = std::strtok(line, " \t\r\n"); token != nullptr;
       token = std::strtok(nullptr, " \t\r\n")) {
    tokens.emplace_back(token);
  }
  if (tokens.empty()) {
    return false;
  }
  int op = -1;
  for (int i = 0; i < kOpCount; ++i) {
    if (tokens[0] == kOpNames[i]) {
      op = i;
    }
  }
  if (op < 0) {
    return false;
  }
  const size_t expected_tokens = (op == kSqrt) ? 3 : (op == kFma ? 5 : 4);
  if (tokens.size() != expected_tokens) {
    return false;
  }
  *out = ParityCase{op, 0, 0, 0, 0};
  if (!parse_hex(tokens[1], &out->x)) {
    return false;
  }
  if (op == kSqrt) {
    return parse_hex(tokens[2], &out->expected);
  }
  if (!parse_hex(tokens[2], &out->y)) {
    return false;
  }
  if (op == kFma) {
    return parse_hex(tokens[3], &out->z) && parse_hex(tokens[4], &out->expected);
  }
  return parse_hex(tokens[3], &out->expected);
}

void describe(const ParityCase &c, char *buffer, size_t size) {
  switch (c.op) {
    case kSqrt:
      snprintf(buffer, size, "sqrt(0x%08x)", c.x);
      break;
    case kFma:
      snprintf(buffer, size, "fma(0x%08x, 0x%08x, 0x%08x)", c.x, c.y, c.z);
      break;
    default:
      snprintf(buffer, size, "%s(0x%08x, 0x%08x)", kOpNames[c.op], c.x, c.y);
      break;
  }
}

}  // namespace

int main() {
  std::vector<ParityCase> cases;
  char line[512];
  int line_number = 0;
  while (std::fgets(line, sizeof line, stdin) != nullptr) {
    ++line_number;
    std::string original(line);
    ParityCase parsed{};
    if (!parse_line(line, &parsed)) {
      std::fprintf(stderr, "cuda_float32_parity: cannot read line %d: %s", line_number,
                   original.c_str());
      return 2;
    }
    cases.push_back(parsed);
  }
  if (cases.empty()) {
    std::fprintf(stderr, "cuda_float32_parity: no cases on stdin\n");
    return 2;
  }

  if (std::ferror(stdin) || cases.size() > static_cast<size_t>(INT_MAX)) {
    std::fprintf(stderr, "cuda_float32_parity: input read failed or case count exceeds kernel range\n");
    return 2;
  }
  for (int op = 0; op < kOpCount; ++op) {
    bool present = false;
    for (const auto &c : cases) {
      present = present || c.op == op;
    }
    if (!present) {
      std::fprintf(stderr, "cuda_float32_parity: no cases for %s\n", kOpNames[op]);
      return 2;
    }
  }

  int device = 0;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
    std::fprintf(stderr, "cuda_float32_parity: no usable CUDA device\n");
    return 2;
  }

  ParityCase *device_cases = nullptr;
  uint32_t *device_results = nullptr;
  const size_t case_bytes = cases.size() * sizeof(ParityCase);
  const size_t result_bytes = cases.size() * sizeof(uint32_t);
  if (cudaMalloc(&device_cases, case_bytes) != cudaSuccess ||
      cudaMalloc(&device_results, result_bytes) != cudaSuccess) {
    std::fprintf(stderr, "cuda_float32_parity: device allocation failed\n");
    return 2;
  }
  if (cudaMemcpy(device_cases, cases.data(), case_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    std::fprintf(stderr, "cuda_float32_parity: input transfer failed\n");
    return 2;
  }
  const int block = 128;
  const int grid = static_cast<int>((cases.size() + block - 1) / block);
  device_apply<<<grid, block>>>(device_cases, device_results, static_cast<int>(cases.size()));
  const cudaError_t launch = cudaDeviceSynchronize();
  if (launch != cudaSuccess) {
    std::fprintf(stderr, "cuda_float32_parity: kernel failed: %s\n", cudaGetErrorString(launch));
    return 2;
  }
  std::vector<uint32_t> device_bits(cases.size(), 0u);
  if (cudaMemcpy(device_bits.data(), device_results, result_bytes, cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    std::fprintf(stderr, "cuda_float32_parity: result transfer failed\n");
    return 2;
  }
  cudaFree(device_cases);
  cudaFree(device_results);

  std::printf("== CUDA binary32 parity against IEEE32Exec ==\n");
  std::printf("device: %s, compute capability %d.%d\n", properties.name, properties.major,
              properties.minor);
  std::printf("cases: %zu\n", cases.size());

  Tally tallies[kOpCount];
  NaNEncodings model_nans;
  NaNEncodings host_nans;
  NaNEncodings device_nans;
  int reported = 0;
  int strict_only = 0;
  for (size_t i = 0; i < cases.size(); ++i) {
    const ParityCase &c = cases[i];
    Tally &t = tallies[c.op];
    ++t.checked;
    const uint32_t host_bits = host_apply(c);
    model_nans.note(c.expected);
    host_nans.note(host_bits);
    device_nans.note(device_bits[i]);
    const bool host_strict = host_bits == c.expected;
    const bool device_strict = device_bits[i] == c.expected;
    const bool host_ok = agrees_up_to_nan(host_bits, c.expected);
    const bool device_ok = agrees_up_to_nan(device_bits[i], c.expected);
    if (!host_strict) {
      ++t.host_strict_failed;
    }
    if (!device_strict) {
      ++t.device_strict_failed;
    }
    if (!host_ok) {
      ++t.host_contract_failed;
    }
    if (!device_ok) {
      ++t.device_contract_failed;
    }
    // Print the cases that break the contract first; if there are none, spend the same budget on
    // the ones that only differ in a `NaN` payload, because those are the evidence for weakening
    // the assumption in the first place.
    const bool contract_break = !host_ok || !device_ok;
    const bool payload_only = !contract_break && (!host_strict || !device_strict);
    if (contract_break && reported < 8) {
      ++reported;
      char what[128];
      describe(c, what, sizeof what);
      std::printf("  broken  %s: model 0x%08x host 0x%08x device 0x%08x\n", what, c.expected,
                  host_bits, device_bits[i]);
    } else if (payload_only) {
      ++strict_only;
      if (strict_only <= 4) {
        char what[128];
        describe(c, what, sizeof what);
        std::printf("  payload %s: model 0x%08x host 0x%08x device 0x%08x\n", what, c.expected,
                    host_bits, device_bits[i]);
      }
    }
  }

  int contract_failed = 0;
  int strict_failed = 0;
  for (int op = 0; op < kOpCount; ++op) {
    const Tally &t = tallies[op];
    if (t.checked == 0) {
      continue;
    }
    contract_failed += t.host_contract_failed + t.device_contract_failed;
    strict_failed += t.host_strict_failed + t.device_strict_failed;
    std::printf("  %-4s contract host %d/%d device %d/%d  strict host %d/%d device %d/%d\n",
                kOpNames[op], t.checked - t.host_contract_failed, t.checked,
                t.checked - t.device_contract_failed, t.checked, t.checked - t.host_strict_failed,
                t.checked, t.checked - t.device_strict_failed, t.checked);
  }

  std::printf("distinct NaN encodings produced:\n");
  print_encodings("model", model_nans);
  print_encodings("host", host_nans);
  print_encodings("device", device_nans);

  if (contract_failed == 0) {
    std::printf("all five contract fields hold on this device and this compiler\n");
    if (strict_failed != 0) {
      std::printf("%d of them only up to the NaN payload, which is what AgreeUpToNaN allows\n",
                  strict_failed);
    }
    return 0;
  }
  std::printf("%d comparisons disagree: NativePrimitiveAgreement is false here\n", contract_failed);
  return 1;
}
