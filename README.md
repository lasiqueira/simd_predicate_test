# simd_predicate_test

A small experiment in branchless conditionals and SIMD: how a plain `a < b ? x : y` conditional
can be rewritten with no branch at all (compute both results, select with a bitmask), and how that
trick scales across register widths (XMM/YMM/ZMM) and data types (`uint8_t` through `double`).
Includes a repetition-tester-based benchmark comparing genuinely-scalar, AVX2, and AVX-512
implementations, both in a realistic large-buffer scenario and in an L1-cache-resident,
compute-isolated one.

The full write-up (branchless conditionals, register width history, saturating arithmetic, and the
benchmark results/analysis below) is in [docs/index.html](docs/index.html).

## Prerequisites

- Windows
- Visual Studio 2022+ with the "Desktop development with C++" workload (provides `cl.exe`, `vcvarsall.bat`,
  and the Windows SDK)
- Optionally, the "C++ Clang tools for Windows" VS component, if you want to build with `clang-cl`
  instead of `cl`
- A CPU with AVX-512 support (the code is compiled with `/arch:AVX512` and uses AVX-512 intrinsics
  directly)
- To exercise the large-page code path, the process needs the "Lock pages in memory" user right
  (`SeLockMemoryPrivilege`) assigned to your account via Local Security Policy (`secpol.msc`); without
  it, the benchmark still runs correctly, just falling back to normal `VirtualAlloc` pages

## Building

```
build.bat          # MSVC (cl), default
build.bat msvc     # MSVC (cl), explicit
build.bat clang    # clang-cl
```

Builds to `build\simd_predicate_test.exe`. Every invocation recompiles all sources from scratch, so
there's nothing to clean between builds.

## Results

Measured on an AMD Ryzen 7 7800X3D, MSVC build. Both benchmarks run each implementation repeatedly
for a few seconds and report the true minimum time (via an RDTSC-based repetition tester), so the
numbers reflect real, repeated computation rather than a single noisy sample - see the "are we
optimizing anything away" discussion in the article for how that was verified.

### Large buffer (10,000,000 elements, memory-bandwidth-bound for the wider types)

| Type | Scalar (bitmask) | AVX2 | AVX-512 |
|---|---|---|---|
| u8  | 0.79 GOp/s | 58.6 GOp/s | 58.5 GOp/s |
| u16 | 1.94 GOp/s | 29.1 GOp/s | 29.0 GOp/s |
| u32 | 14.4 GOp/s | 14.6 GOp/s | 13.8 GOp/s |
| u64 | 3.78 GOp/s | 3.78 GOp/s | 3.93 GOp/s |
| f32 | 1.56 GOp/s | 14.4 GOp/s | 13.9 GOp/s |
| f64 | 1.56 GOp/s | 3.67 GOp/s | 3.84 GOp/s |

### L1-resident (4KB buffer, looped 4096 times per sample - compute-bound, no memory bandwidth)

| Type | Scalar (bitmask)| AVX2 | AVX-512 | AVX-512 vs AVX2 |
|---|---|---|---|---|
| u8  | 0.80 GOp/s | 77.8 GOp/s | 80.7 GOp/s | +4% |
| u16 | 1.98 GOp/s | 38.9 GOp/s | 58.0 GOp/s | +49% |
| u32 | 17.6 GOp/s | 19.2 GOp/s | 29.3 GOp/s | +52% |
| u64 | 8.86 GOp/s | 9.71 GOp/s | 14.1 GOp/s | +46% |
| f32 | 1.60 GOp/s | 18.9 GOp/s | 34.0 GOp/s | +79% |
| f64 | 1.61 GOp/s | 9.41 GOp/s | 17.3 GOp/s | +84% |

The short version: once memory bandwidth is taken out of the picture, AVX-512 has a real,
consistent edge over AVX2 for every type - the near-tie in the large-buffer table was memory
bandwidth masking that advantage, not Zen 4's "double-pumped" AVX-512 erasing it outright. See
[docs/index.html](docs/index.html) for the full explanation.

## License

[MIT](LICENSE)
