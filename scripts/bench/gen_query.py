#!/usr/bin/env python3
"""Write a crown_query_v1 document for `verify crown-query`: a ReLU chain of the given widths.

The weights and biases are multiples of 1/8 from -1 to 1, drawn from a fixed linear congruential
generator, so every run writes the same file. Each ReLU layer proposes the CROWN slope 1/2, the
input box is [-1, 1], and each query row asks that one output stay below 10^9, so the checker
accepts the query.
"""

import argparse
import json


def lcg(seed):
    """The upper 31 bits of a 64-bit linear congruential generator started at `seed`."""
    x = seed
    while True:
        x = (6364136223846793005 * x + 1442695040888963407) % (1 << 64)
        yield x >> 33


def query(dims):
    """The query document of a chain with widths `dims` (input first)."""
    g = lcg(20261108)

    def w():
        return f"{(next(g) % 17) - 8}/8"

    layers = []
    for i, (n, m) in enumerate(zip(dims, dims[1:])):
        layers.append({"kind": "linear",
                       "weights": [[w() for _ in range(n)] for _ in range(m)],
                       "bias": [w() for _ in range(m)]})
        if i + 2 < len(dims):
            layers.append({"kind": "relu", "alpha": ["1/2"] * m})
    k = dims[-1]
    return {"format": "crown_query_v1", "input_dim": dims[0],
            "input": {"lo": ["-1"] * dims[0], "hi": ["1"] * dims[0]},
            "layers": layers,
            "query": {"weights": [["1" if j == i else "0" for j in range(k)] for i in range(k)],
                      "bias": ["-1000000000"] * k, "strict": True}}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", metavar="OUT.json", help="the file to write")
    parser.add_argument("dims", metavar="DIM", type=int, nargs="+",
                        help="the widths of the chain, input first")
    args = parser.parse_args()
    if len(args.dims) < 2:
        parser.error("give at least two widths")
    with open(args.out, "w") as f:
        json.dump(query(args.dims), f, separators=(",", ":"))
        f.write("\n")


if __name__ == "__main__":
    main()
