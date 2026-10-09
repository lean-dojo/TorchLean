// Copyright (c) 2026 TorchLean contributors. MIT license.
#pragma once

// NVRTC receives this bundled header in memory; execution never reads source files from disk.
namespace torchlean {
inline constexpr char binary_cuda[] = R"CUDA(
// Copyright (c) 2026 TorchLean contributors. MIT license.
// Device-side configured binary arithmetic for custom computations. This is a foreign-code
// implementation, not a replacement for FloatLib's proofs. Words are little-endian bytes;
// compatible IEEE operations use native instructions; other formats use integer limbs.
#pragma once

enum class TLOperation { add, mul, div };

template<int Bits> struct TLInteger {
  static constexpr int limbs = (Bits + 31) / 32;
  unsigned int words[limbs]{};
  __device__ bool bit(long long i) const {
    return i >= 0 && i < limbs * 32 && ((words[i / 32] >> (i % 32)) & 1U);
  }
  __device__ void set(long long i, bool b = true) {
    if (i >= 0 && i < limbs * 32) {
      const auto mask = 1U << (i % 32);
      words[i / 32] = b ? words[i / 32] | mask : words[i / 32] & ~mask;
    }
  }
  __device__ int length() const {
    for (int i = limbs - 1; i >= 0; --i)
      if (words[i]) return i * 32 + 32 - __clz(words[i]);
    return 0;
  }
  __device__ int compare(const TLInteger& b) const {
    for (int i = limbs - 1; i >= 0; --i)
      if (words[i] != b.words[i]) return words[i] > b.words[i] ? 1 : -1;
    return 0;
  }
  __device__ bool below(long long n) const {
    if (n <= 0) return false;
    for (int i = 0; i < limbs; ++i) {
      const long long k = n - i * 32;
      if (k >= 32 && words[i]) return true;
      if (k > 0 && k < 32 && (words[i] & ((1U << k) - 1U))) return true;
    }
    return false;
  }
  __device__ TLInteger shift(long long n) const {
    TLInteger r;
    if (n >= limbs * 32 || n <= -limbs * 32) return r;
    for (int i = 0; i < limbs * 32; ++i) if (bit(i)) r.set(i + n);
    return r;
  }
  __device__ TLInteger jam(long long n) const {
    if (n <= 0) return shift(-n);
    TLInteger r = shift(-n);
    if (below(n)) r.set(0);
    return r;
  }
  __device__ TLInteger add(const TLInteger& b) const {
    TLInteger r;
    unsigned long long carry = 0;
    for (int i = 0; i < limbs; ++i) {
      carry += (unsigned long long)words[i] + b.words[i];
      r.words[i] = (unsigned int)carry;
      carry >>= 32;
    }
    return r;
  }
  __device__ TLInteger sub(const TLInteger& b) const {
    TLInteger r;
    unsigned long long borrow = 0;
    for (int i = 0; i < limbs; ++i) {
      const unsigned long long y = (unsigned long long)b.words[i] + borrow;
      r.words[i] = (unsigned int)((unsigned long long)words[i] - y);
      borrow = (unsigned long long)words[i] < y;
    }
    return r;
  }
  __device__ TLInteger mul(const TLInteger& b) const {
    TLInteger r;
    for (int i = 0; i < limbs; ++i) {
      unsigned long long carry = 0;
      for (int j = 0; i + j < limbs; ++j) {
        const auto value = (unsigned long long)words[i] * b.words[j] +
                           r.words[i + j] + carry;
        r.words[i + j] = (unsigned int)value;
        carry = value >> 32;
      }
    }
    return r;
  }
  __device__ TLInteger quotient(const TLInteger& b, TLInteger& rem) const {
    TLInteger q;
    rem = {};
    for (int i = length() - 1; i >= 0; --i) {
      rem = rem.shift(1);
      rem.set(0, bit(i));
      if (rem.compare(b) >= 0) { rem = rem.sub(b); q.set(i); }
    }
    return q;
  }
};

template<int E, int F, long long Bias, int Encoding> struct TLBinary {
  static constexpr int bytes = (1 + E + F + 7) / 8;
  static constexpr unsigned long long all = (1ULL << E) - 1;
  using Integer = TLInteger<2 * F + 16>;
  unsigned char data[bytes]{};
  __device__ bool bit(int i) const { return (data[i / 8] >> (i % 8)) & 1; }
  __device__ void set(int i, bool value = true) {
    const auto mask = (unsigned char)(1U << (i % 8));
    data[i / 8] = value ? data[i / 8] | mask : data[i / 8] & ~mask;
  }
  __device__ bool sign() const { return bit(E + F); }
  __device__ unsigned long long exponent() const {
    unsigned long long e = 0;
    for (int i = 0; i < E; ++i) if (bit(F + i)) e |= 1ULL << i;
    return e;
  }
  __device__ Integer fraction() const {
    Integer r;
    for (int i = 0; i < F; ++i) r.set(i, bit(i));
    return r;
  }
  __device__ Integer significand() const {
    auto r = fraction();
    if (exponent() != 0) r.set(F);
    return r;
  }
  __device__ long long scale() const {
    return (long long)(exponent() ? exponent() : 1) - Bias - F;
  }
  __device__ bool fractionAll() const {
    for (int i = 0; i < F; ++i) if (!bit(i)) return false;
    return true;
  }
  __device__ bool nan() const {
    if constexpr (Encoding == 0) return exponent() == all && fraction().length() != 0;
    if constexpr (Encoding == 1) return exponent() == all && fractionAll();
    if constexpr (Encoding == 2) return sign() && exponent() == 0 && !fraction().length();
    return false;
  }
  __device__ bool signaling() const { return Encoding == 0 && nan() && !bit(F - 1); }
  __device__ bool inf() const {
    return Encoding == 0 && exponent() == all && !fraction().length();
  }
  __device__ bool zero() const { return exponent() == 0 && !fraction().length() && !nan(); }
  __device__ TLBinary quiet() const {
    auto r = *this;
    if constexpr (Encoding == 0) r.set(F - 1);
    return r;
  }
  __device__ static TLBinary fields(bool negative, unsigned long long e, Integer f = {}) {
    TLBinary r;
    r.set(E + F, negative);
    for (int i = 0; i < E; ++i) r.set(F + i, (e >> i) & 1);
    for (int i = 0; i < F; ++i) r.set(i, f.bit(i));
    return r;
  }
  __device__ static TLBinary invalid() {
    if constexpr (Encoding == 0) {
      Integer f; f.set(F - 1); return fields(false, all, f);
    }
    if constexpr (Encoding == 1) return overflow(false);
    if constexpr (Encoding == 2) return fields(true, 0);
    return fields(false, 0);
  }
  __device__ static TLBinary zeroValue(bool negative) {
    return fields(Encoding == 2 ? false : negative, 0);
  }
  __device__ static TLBinary overflow(bool negative) {
    if constexpr (Encoding == 0) return fields(negative, all);
    if constexpr (Encoding == 2) return fields(true, 0);
    Integer f;
    for (int i = 0; i < F; ++i) f.set(i);
    return fields(negative, all, f);
  }
  __device__ static bool choose(TLBinary a, TLBinary b, TLBinary& result) {
    if (a.signaling()) result = a.quiet();
    else if (b.signaling()) result = b.quiet();
    else if (a.nan()) result = a.quiet();
    else if (b.nan()) result = b.quiet();
    else return false;
    return true;
  }
  // Match the complete descriptor, not just its storage width. In particular, a custom bias
  // or a finite-only encoding must never inherit IEEE hardware overflow/NaN behavior.
  // NaNs and infinities are handled before this call, preserving FloatLib's payload policy.
  // Half/bfloat16 division stays in software: widening division and then narrowing can round
  // twice. Explicit rounding instructions also prevent accidental mul-add contraction.
  template<TLOperation Op>
  __device__ bool hardware(TLBinary b, TLBinary& result) const {
#ifndef TORCHLEAN_BINARY_SOFTWARE
    constexpr bool half = E == 5 && F == 10 && Bias == 15 && Encoding == 0;
    constexpr bool bfloat = E == 8 && F == 7 && Bias == 127 && Encoding == 0;
    constexpr bool single = E == 8 && F == 23 && Bias == 127 && Encoding == 0;
    constexpr bool wide = E == 11 && F == 52 && Bias == 1023 && Encoding == 0;
    if constexpr (half || bfloat || single || wide) {
      unsigned long long aWord = 0, bWord = 0, rWord = 0;
      for (int i = 0; i < bytes; ++i) {
        aWord |= (unsigned long long)data[i] << (8 * i);
        bWord |= (unsigned long long)b.data[i] << (8 * i);
      }
      if constexpr (single) {
        const float a = __uint_as_float((unsigned int)aWord);
        const float y = __uint_as_float((unsigned int)bWord);
        float r;
        if constexpr (Op == TLOperation::add) r = __fadd_rn(a, y);
        if constexpr (Op == TLOperation::mul) r = __fmul_rn(a, y);
        if constexpr (Op == TLOperation::div) r = __fdiv_rn(a, y);
        rWord = __float_as_uint(r);
      } else if constexpr (wide) {
        const double a = __longlong_as_double((long long)aWord);
        const double y = __longlong_as_double((long long)bWord);
        double r;
        if constexpr (Op == TLOperation::add) r = __dadd_rn(a, y);
        if constexpr (Op == TLOperation::mul) r = __dmul_rn(a, y);
        if constexpr (Op == TLOperation::div) r = __ddiv_rn(a, y);
        rWord = (unsigned long long)__double_as_longlong(r);
      } else {
        unsigned short r, a = (unsigned short)aWord, y = (unsigned short)bWord;
        if constexpr (Op == TLOperation::div) return false;
        else if constexpr (half) {
#if __CUDA_ARCH__ >= 530
          if constexpr (Op == TLOperation::add)
            asm("add.rn.f16 %0, %1, %2;" : "=h"(r) : "h"(a), "h"(y));
          if constexpr (Op == TLOperation::mul)
            asm("mul.rn.f16 %0, %1, %2;" : "=h"(r) : "h"(a), "h"(y));
#else
          return false;
#endif
        } else {
#if __CUDA_ARCH__ >= 900
          if constexpr (Op == TLOperation::add)
            asm("add.rn.bf16 %0, %1, %2;" : "=h"(r) : "h"(a), "h"(y));
          if constexpr (Op == TLOperation::mul)
            asm("mul.rn.bf16 %0, %1, %2;" : "=h"(r) : "h"(a), "h"(y));
#elif __CUDA_ARCH__ >= 800
          // CUDA's bfloat16 intrinsics use these identities on Ampere. The negative-zero
          // addend preserves a negative-zero product. This is still one rounding per op,
          // not contraction of two operations from the user's program.
          if constexpr (Op == TLOperation::add)
            asm("fma.rn.bf16 %0, %1, %2, %3;"
                : "=h"(r) : "h"(a), "h"((unsigned short)0x3f80), "h"(y));
          if constexpr (Op == TLOperation::mul)
            asm("fma.rn.bf16 %0, %1, %2, %3;"
                : "=h"(r) : "h"(a), "h"(y), "h"((unsigned short)0x8000));
#else
          return false;
#endif
        }
        rWord = r;
      }
      for (int i = 0; i < bytes; ++i) result.data[i] = (unsigned char)(rWord >> (8 * i));
      return true;
    }
#endif
    return false;
  }
  // Round an integer times 2^scale, with a sticky remainder below its low bit.
  __device__ static TLBinary round(bool negative, Integer m, long long scale,
                                   bool sticky = false) {
    if (!m.length()) return zeroValue(negative);
    long long shift = m.length() - 1 - F;
    const long long minimum = 1 - Bias - F;
    if (scale + shift < minimum) shift = minimum - scale;
    auto q = m.shift(-shift);
    if (shift > 0 && m.bit(shift - 1) &&
        (m.below(shift - 1) || sticky || q.bit(0))) {
      Integer one; one.set(0); q = q.add(one);
    }
    scale += shift;
    if (q.length() > F + 1) { q = q.shift(-1); ++scale; }
    if (!q.length()) return zeroValue(negative);
    const long long e = scale + F + Bias;
    if (e > (long long)all || (Encoding == 0 && e == (long long)all))
      return overflow(negative);
    if constexpr (Encoding == 1) {
      if (e == (long long)all) {
        bool maximum = true;
        for (int i = 0; i < F; ++i) maximum = maximum && q.bit(i);
        if (maximum) return overflow(negative);
      }
    }
    return fields(negative, q.bit(F) ? (unsigned long long)e : 0, q);
  }
  __device__ TLBinary operator-() const {
    if constexpr (Encoding == 2) if (nan() || zero()) return *this;
    auto r = *this; r.set(E + F, !sign()); return r;
  }
  __device__ TLBinary operator+(TLBinary b) const {
    TLBinary selected;
    if (choose(*this, b, selected)) return selected;
    if (inf()) return b.inf() && sign() != b.sign() ? invalid() : *this;
    if (b.inf()) return b;
    if (hardware<TLOperation::add>(b, selected)) return selected;
    auto x = significand(), y = b.significand();
    const auto sx = scale(), sy = b.scale();
    const auto common = sx > sy ? sx : sy;
    x = x.shift(3).jam(common - sx);
    y = y.shift(3).jam(common - sy);
    if (sign() == b.sign()) return round(sign(), x.add(y), common - 3);
    const auto order = x.compare(y);
    if (order == 0) return zeroValue(false);
    return order > 0 ? round(sign(), x.sub(y), common - 3)
                     : round(b.sign(), y.sub(x), common - 3);
  }
  __device__ TLBinary operator-(TLBinary b) const { return *this + (-b); }
  __device__ TLBinary operator*(TLBinary b) const {
    TLBinary selected;
    if (choose(*this, b, selected)) return selected;
    const bool negative = sign() != b.sign();
    if (inf() || b.inf()) return zero() || b.zero() ? invalid() : fields(negative, all);
    if (hardware<TLOperation::mul>(b, selected)) return selected;
    return round(negative, significand().mul(b.significand()), scale() + b.scale());
  }
  __device__ TLBinary operator/(TLBinary b) const {
    TLBinary selected;
    if (choose(*this, b, selected)) return selected;
    const bool negative = sign() != b.sign();
    if (inf()) return b.inf() ? invalid() : fields(negative, all);
    if (b.inf()) return zeroValue(negative);
    if (b.zero()) return zero() ? invalid() : overflow(negative);
    if (zero()) return zeroValue(negative);
    if (hardware<TLOperation::div>(b, selected)) return selected;
    const auto x = significand(), y = b.significand();
    const int extra = F + 4 + y.length() - x.length();
    Integer remainder;
    const auto q = x.shift(extra).quotient(y, remainder);
    return round(negative, q, scale() - b.scale() - extra, remainder.length() != 0);
  }
  __device__ bool operator==(TLBinary b) const {
    if (nan() || b.nan()) return false;
    if (zero() && b.zero()) return true;
    for (int i = 0; i < bytes; ++i) if (data[i] != b.data[i]) return false;
    return true;
  }
  __device__ bool operator<(TLBinary b) const {
    if (nan() || b.nan() || (*this == b)) return false;
    if (sign() != b.sign()) return sign();
    for (int i = E + F - 1; i >= 0; --i)
      if (bit(i) != b.bit(i)) return sign() ? bit(i) : !bit(i);
    return false;
  }
  __device__ bool operator<=(TLBinary b) const { return (*this < b) || (*this == b); }
};
)CUDA";
}  // namespace torchlean
