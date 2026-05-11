# TorchLean Julia producer: piecewise-linear “spline” certificate.
#
# This script is dependency-free (Julia Base only). It “fits” a piecewise-linear
# interpolant to a small in-script dataset, then prints a JSON certificate to stdout.
#
# The corresponding Lean checker lives at:
#   `NN/Verification/Splines/PiecewisePolyCert.lean`
#
# Run from the TorchLean repo root:
#   julia --color=no --startup-file=no scripts/verification/splines/fit_piecewise_linear.jl
#
# The Lean checker calls this script via `IO.Process` when invoked with `--regen`:
#   `lake exe verify -- spline-cert --regen`

function rat_str(q::Rational{T}) where {T}
  n = numerator(q)
  d = denominator(q)
  if d == 1
    return string(n)
  else
    return string(n) * "/" * string(d)
  end
end

function json_string_array(ss::Vector{String})
  return "[" * join(["\"" * s * "\"" for s in ss], ", ") * "]"
end

function main()
  # Small exact-rational dataset used for a deterministic certificate example.
  xs = Rational{BigInt}[0//1, 1//1, 2//1, 3//1]
  ys = Rational{BigInt}[0//1, 1//1, 0//1, 1//1]
  degree = 1

  pieces = String[]
  for i in 1:(length(xs) - 1)
    lo = xs[i]
    hi = xs[i + 1]
    ylo = ys[i]
    yhi = ys[i + 1]
    m = (yhi - ylo) / (hi - lo)  # slope
    coeffs = [rat_str(ylo), rat_str(m)]  # p(t) = ylo + m * (x - lo)
    push!(
      pieces,
      "{ " *
      "\"lo\": \"" * rat_str(lo) * "\", " *
      "\"hi\": \"" * rat_str(hi) * "\", " *
      "\"coeffs\": " * json_string_array(coeffs) *
      " }",
    )
  end

  println("{")
  println("  \"format\": \"piecewise_poly_v0\",")
  println("  \"degree\": $degree,")
  println("  \"xs\": " * json_string_array([rat_str(x) for x in xs]) * ",")
  println("  \"ys\": " * json_string_array([rat_str(y) for y in ys]) * ",")
  println("  \"pieces\": [")
  for (k, p) in enumerate(pieces)
    comma = k < length(pieces) ? "," : ""
    println("    " * p * comma)
  end
  println("  ]")
  println("}")
end

main()
