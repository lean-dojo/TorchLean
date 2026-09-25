// Actual ATen/C ABI parity against the Lean binary32 reference stream.
// Compile through cuda_float32_parity.sh on a CUDA host. All GPU arithmetic comes from
// the selected LibTorch SDK; there are no local CUDA kernels or intrinsics.
//
// Input (stdin, or an optional file argument):
//   add  X Y EXPECTED
//   mul  X Y EXPECTED
//   div  X Y EXPECTED
//   sqrt X EXPECTED
//   fma  X Y Z EXPECTED
// Words are hexadecimal binary32 encodings emitted by native_float32_parity --emit-cases.
//
// The implementation checks exact finite bits/AgreeUpToNaN, production add/mul/div/AXPY,
// batched upstream tensor FMA and raw IEEE sqrt, plus Buffer.sqrt's separately selected
// nonpositive branch. It also runs the FMA, staged Adam and noAutograd discriminators.
#include <iostream>

int torchlean_elementwise_regressions(const char* reference_path);

int main(int argc, char** argv) {
  if (argc > 2) {
    std::cerr << "usage: torchlean_elementwise_regression [LEAN_CASES.txt]\n";
    return 2;
  }
  return torchlean_elementwise_regressions(argc == 2 ? argv[1] : "/dev/stdin");
}
