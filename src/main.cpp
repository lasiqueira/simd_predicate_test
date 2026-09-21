#include <vector>
#include <stdio.h>
#include <cstring>
#include <immintrin.h>
#include <cstdint>
#include <random>
#include <limits>
#include "repetition_tester.hpp"
#include "platform_metrics.hpp"
#include "large_page_buffer.hpp"

typedef double f64;
typedef float f32;
typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;

// saturating add/sub so the false/true branches can never over/underflow
inline u8 sat_add_u8(u8 a, u8 b)
{
    unsigned sum = (unsigned)a + b;
    return sum > 255 ? (u8)255 : (u8)sum;
}

inline u8 sat_sub_u8(u8 a, u8 b)
{
    return a > b ? (u8)(a - b) : (u8)0;
}

inline u8 condScalar(u8 a, u8 b)
{
    return a < b
        ? sat_add_u8(a, b)
        : sat_sub_u8(sat_sub_u8(a, b), 2);
}


inline u8 condScalarBin(u8 a, u8 b)
{
    u8 mask = (u8)0 - (u8)(a < b);
    u8 trueRes = sat_add_u8(a, b);
    u8 falseRes = sat_sub_u8(sat_sub_u8(a, b), 2);
    return (u8)((mask & trueRes) | (~mask & falseRes));
}

inline __m256i cond256(__m256i a, __m256i b)
{
    __mmask32 mask = _mm256_cmp_epu8_mask(a, b, _MM_CMPINT_LT);

    __m256i trueRes = _mm256_adds_epu8(a, b);
    __m256i falseRes = _mm256_subs_epu8(
        _mm256_subs_epu8(a, b),
        _mm256_set1_epi8(2)
    );

    return _mm256_mask_blend_epi8(mask, falseRes, trueRes);
}

inline __m512i cond512(__m512i a, __m512i b)
{
    __mmask64 mask = _mm512_cmp_epu8_mask(a, b, _MM_CMPINT_LT);

    __m512i falseRes = _mm512_subs_epu8(
        _mm512_subs_epu8(a, b),
        _mm512_set1_epi8(2)
    );

    return _mm512_mask_adds_epu8(falseRes, mask, a, b);
}

// writes into a caller-provided buffer - no allocation, so the timed loop only measures the compute
inline void lpScalar(const u8* __restrict a, size_t n, u8 b, u8* __restrict res)
{
    for (size_t i = 0; i < n; ++i)
        res[i] = condScalar(a[i], b);
}

inline void lpScalarBin(const u8* __restrict a, size_t n, u8 b, u8* __restrict res)
{
    for (size_t i = 0; i < n; ++i)
        res[i] = condScalarBin(a[i], b);
}

inline void lp256(const u8* __restrict a, size_t n, u8 b, u8* __restrict res)
{
    __m256i wide_b = _mm256_set1_epi8((char)b);

    for (size_t i = 0; i < n; i += 32)
    {
        __m256i packed = _mm256_loadu_si256((const __m256i*)&a[i]);
        __m256i condRes = cond256(packed, wide_b);
        _mm256_storeu_si256((__m256i*)&res[i], condRes);
    }
}

inline void lp512(const u8* __restrict a, size_t n, u8 b, u8* __restrict res)
{
    __m512i wide_b = _mm512_set1_epi8((char)b);

    for (size_t i = 0; i < n; i += 64)
    {
        __m512i packed = _mm512_loadu_si512(&a[i]);
        __m512i condRes = cond512(packed, wide_b);
        _mm512_storeu_si512(&res[i], condRes);
    }
}

// ---- wider integer types: instead of saturating, bound the random inputs so a+b and a*b-2
// can never over/underflow in the first place - this lets us use real (non-saturating) SIMD
// add/multiply, which u8's tiny range is too small to make worthwhile.
template <typename T>
inline T condScalar(T a, T b)
{
    if(a < b)
        return (T)(a + b);
    else
        return (T)(a * b - 2);
}

template <typename T>
inline T condScalarBin(T a, T b)
{
    T mask = (T)0 - (T)(a < b);
    T trueRes = (T)(a + b);
    T falseRes = (T)(a * b - 2);
    return (T)((mask & trueRes) | (T)(~mask & falseRes));
}

// floats can't use bitwise ops directly (not defined by the language for float/double), so these
// reinterpret the bits through a union instead - same mask/select shape as the integer template.
inline f64 condScalarBin(f64 a, f64 b)
{
    union { f64 f; u64 u; } M, T, F, Result;
    M.u = (u64)0 - (u64)(a < b);
    T.f = a + b;
    F.f = a * b - 2;
    Result.u = (M.u & T.u) | (~M.u & F.u);
    return Result.f;
}

inline f32 condScalarBin(f32 a, f32 b)
{
    union { f32 f; u32 u; } M, T, F, Result;
    M.u = (u32)0 - (u32)(a < b);
    T.f = a + b;
    F.f = a * b - 2;
    Result.u = (M.u & T.u) | (~M.u & F.u);
    return Result.f;
}

inline __m256d cond256(__m256d a, __m256d b)
{
    __m256d mask = _mm256_cmp_pd(a, b, _CMP_LT_OQ);
    __m256d t = _mm256_add_pd(a, b);
    __m256d f = _mm256_sub_pd(_mm256_mul_pd(a, b), _mm256_set1_pd(2.0));
    return _mm256_blendv_pd(f, t, mask);
}

inline __m512d cond512(__m512d a, __m512d b)
{
    __mmask8 mask = _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ);
    __m512d result = _mm512_sub_pd(_mm512_mul_pd(a, b), _mm512_set1_pd(2.0));
    return _mm512_mask_add_pd(result, mask, a, b);
}

inline __m256 cond256(__m256 a, __m256 b)
{
    __m256 mask = _mm256_cmp_ps(a, b, _CMP_LT_OQ);
    __m256 t = _mm256_add_ps(a, b);
    __m256 f = _mm256_sub_ps(_mm256_mul_ps(a, b), _mm256_set1_ps(2.0f));
    return _mm256_blendv_ps(f, t, mask);
}

inline __m512 cond512(__m512 a, __m512 b)
{
    __mmask16 mask = _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ);
    __m512 result = _mm512_sub_ps(_mm512_mul_ps(a, b), _mm512_set1_ps(2.0f));
    return _mm512_mask_add_ps(result, mask, a, b);
}

inline void lp256(const f64* __restrict a, size_t n, f64 b, f64* __restrict res)
{
    __m256d wide_b = _mm256_set1_pd(b);

    for (size_t i = 0; i < n; i += 4)
    {
        __m256d packed = _mm256_loadu_pd(&a[i]);
        __m256d condRes = cond256(packed, wide_b);
        _mm256_storeu_pd(&res[i], condRes);
    }
}

inline void lp512(const f64* __restrict a, size_t n, f64 b, f64* __restrict res)
{
    __m512d wide_b = _mm512_set1_pd(b);

    for (size_t i = 0; i < n; i += 8)
    {
        __m512d packed = _mm512_loadu_pd(&a[i]);
        __m512d condRes = cond512(packed, wide_b);
        _mm512_storeu_pd(&res[i], condRes);
    }
}

inline void lp256(const f32* __restrict a, size_t n, f32 b, f32* __restrict res)
{
    __m256 wide_b = _mm256_set1_ps(b);

    for (size_t i = 0; i < n; i += 8)
    {
        __m256 packed = _mm256_loadu_ps(&a[i]);
        __m256 condRes = cond256(packed, wide_b);
        _mm256_storeu_ps(&res[i], condRes);
    }
}

inline void lp512(const f32* __restrict a, size_t n, f32 b, f32* __restrict res)
{
    __m512 wide_b = _mm512_set1_ps(b);

    for (size_t i = 0; i < n; i += 16)
    {
        __m512 packed = _mm512_loadu_ps(&a[i]);
        __m512 condRes = cond512(packed, wide_b);
        _mm512_storeu_ps(&res[i], condRes);
    }
}

template <typename T>
inline void lp(const T* __restrict a, size_t n, T b, T* __restrict res)
{
    for (size_t i = 0; i < n; ++i)
        res[i] = condScalar(a[i], b);
}

template <typename T>
inline void lp2(const T* __restrict a, size_t n, T b, T* __restrict res)
{
    for (size_t i = 0; i < n; ++i)
        res[i] = condScalarBin(a[i], b);
}

inline __m256i cond256_u16(__m256i a, __m256i b)
{
    __mmask16 mask = _mm256_cmp_epu16_mask(a, b, _MM_CMPINT_LT);
    __m256i trueRes = _mm256_add_epi16(a, b);
    __m256i falseRes = _mm256_sub_epi16(_mm256_mullo_epi16(a, b), _mm256_set1_epi16(2));
    return _mm256_mask_blend_epi16(mask, falseRes, trueRes);
}

inline __m512i cond512_u16(__m512i a, __m512i b)
{
    __mmask32 mask = _mm512_cmp_epu16_mask(a, b, _MM_CMPINT_LT);
    __m512i falseRes = _mm512_sub_epi16(_mm512_mullo_epi16(a, b), _mm512_set1_epi16(2));
    return _mm512_mask_add_epi16(falseRes, mask, a, b);
}

inline void lp256_u16(const u16* __restrict a, size_t n, u16 b, u16* __restrict res)
{
    __m256i wide_b = _mm256_set1_epi16((short)b);

    for (size_t i = 0; i < n; i += 16)
    {
        __m256i packed = _mm256_loadu_si256((const __m256i*)&a[i]);
        __m256i condRes = cond256_u16(packed, wide_b);
        _mm256_storeu_si256((__m256i*)&res[i], condRes);
    }
}

inline void lp512_u16(const u16* __restrict a, size_t n, u16 b, u16* __restrict res)
{
    __m512i wide_b = _mm512_set1_epi16((short)b);

    for (size_t i = 0; i < n; i += 32)
    {
        __m512i packed = _mm512_loadu_si512(&a[i]);
        __m512i condRes = cond512_u16(packed, wide_b);
        _mm512_storeu_si512(&res[i], condRes);
    }
}

inline __m256i cond256_u32(__m256i a, __m256i b)
{
    __mmask8 mask = _mm256_cmp_epu32_mask(a, b, _MM_CMPINT_LT);
    __m256i trueRes = _mm256_add_epi32(a, b);
    __m256i falseRes = _mm256_sub_epi32(_mm256_mullo_epi32(a, b), _mm256_set1_epi32(2));
    return _mm256_mask_blend_epi32(mask, falseRes, trueRes);
}

inline __m512i cond512_u32(__m512i a, __m512i b)
{
    __mmask16 mask = _mm512_cmp_epu32_mask(a, b, _MM_CMPINT_LT);
    __m512i falseRes = _mm512_sub_epi32(_mm512_mullo_epi32(a, b), _mm512_set1_epi32(2));
    return _mm512_mask_add_epi32(falseRes, mask, a, b);
}

inline void lp256_u32(const u32* __restrict a, size_t n, u32 b, u32* __restrict res)
{
    __m256i wide_b = _mm256_set1_epi32((int)b);

    for (size_t i = 0; i < n; i += 8)
    {
        __m256i packed = _mm256_loadu_si256((const __m256i*)&a[i]);
        __m256i condRes = cond256_u32(packed, wide_b);
        _mm256_storeu_si256((__m256i*)&res[i], condRes);
    }
}

inline void lp512_u32(const u32* __restrict a, size_t n, u32 b, u32* __restrict res)
{
    __m512i wide_b = _mm512_set1_epi32((int)b);

    for (size_t i = 0; i < n; i += 16)
    {
        __m512i packed = _mm512_loadu_si512(&a[i]);
        __m512i condRes = cond512_u32(packed, wide_b);
        _mm512_storeu_si512(&res[i], condRes);
    }
}

inline __m256i cond256_u64(__m256i a, __m256i b)
{
    __mmask8 mask = _mm256_cmp_epu64_mask(a, b, _MM_CMPINT_LT);
    __m256i trueRes = _mm256_add_epi64(a, b);
    __m256i falseRes = _mm256_sub_epi64(_mm256_mullo_epi64(a, b), _mm256_set1_epi64x(2));
    return _mm256_mask_blend_epi64(mask, falseRes, trueRes);
}

inline __m512i cond512_u64(__m512i a, __m512i b)
{
    __mmask8 mask = _mm512_cmp_epu64_mask(a, b, _MM_CMPINT_LT);
    __m512i falseRes = _mm512_sub_epi64(_mm512_mullo_epi64(a, b), _mm512_set1_epi64(2));
    return _mm512_mask_add_epi64(falseRes, mask, a, b);
}

inline void lp256_u64(const u64* __restrict a, size_t n, u64 b, u64* __restrict res)
{
    __m256i wide_b = _mm256_set1_epi64x((long long)b);

    for (size_t i = 0; i < n; i += 4)
    {
        __m256i packed = _mm256_loadu_si256((const __m256i*)&a[i]);
        __m256i condRes = cond256_u64(packed, wide_b);
        _mm256_storeu_si256((__m256i*)&res[i], condRes);
    }
}

inline void lp512_u64(const u64* __restrict a, size_t n, u64 b, u64* __restrict res)
{
    __m512i wide_b = _mm512_set1_epi64((long long)b);

    for (size_t i = 0; i < n; i += 8)
    {
        __m512i packed = _mm512_loadu_si512(&a[i]);
        __m512i condRes = cond512_u64(packed, wide_b);
        _mm512_storeu_si512(&res[i], condRes);
    }
}

// runs the scalar/scalarbin/AVX2/AVX512 waves for one element type and reports whether all three agreed
template <typename T>
void RunTypeBenchmark(
    const char* typeName,
    size_t elementCount,
    T b,
    T minA,
    T maxA,
    void (*fnScalar)(const T* __restrict, size_t, T, T* __restrict),
    void (*fnScalarBin)(const T* __restrict, size_t, T, T* __restrict),
    void (*fn256)(const T* __restrict, size_t, T, T* __restrict),
    void (*fn512)(const T* __restrict, size_t, T, T* __restrict),
    uint64_t cpu_freq,
    uint32_t seconds_per_wave,
    size_t repeatsPerSample = 1)
{
    size_t bytes = elementCount * sizeof(T);
    size_t sampleBytes = bytes * repeatsPerSample;

    LargePageBuffer inputBuf(bytes);
    LargePageBuffer res1Buf(bytes);
    LargePageBuffer res2Buf(bytes);
    LargePageBuffer res3Buf(bytes);
    LargePageBuffer res4Buf(bytes);

    T* v = (T*)inputBuf.data();
    T* res1 = (T*)res1Buf.data();
    T* res2 = (T*)res2Buf.data();
    T* res3 = (T*)res3Buf.data();
    T* res4 = (T*)res4Buf.data();

    std::default_random_engine rnd{std::random_device{}()};
    std::uniform_int_distribution<uint64_t> dist((uint64_t)minA, (uint64_t)maxA);

    for (size_t i = 0; i < elementCount; ++i)
        v[i] = (T)dist(rnd);

    memset(res1, 0, bytes);
    memset(res2, 0, bytes);
    memset(res3, 0, bytes);
    memset(res4, 0, bytes);

    RepetitionTester scalarTester = {};
    RepetitionTester scalarBinTester = {};
    RepetitionTester avx2Tester = {};
    RepetitionTester avx512Tester = {};

    printf("--- %s scalar ---\n", typeName);
    NewTestWave(scalarTester, sampleBytes, cpu_freq, seconds_per_wave);
    while (IsTesting(scalarTester))
    {
        BeginTime(scalarTester);
        for (size_t r = 0; r < repeatsPerSample; ++r)
            fnScalar(v, elementCount, b, res1);
        EndTime(scalarTester);
        CountBytes(scalarTester, sampleBytes);
        CountOps(scalarTester, elementCount * repeatsPerSample);
    }

    printf("--- %s scalar bin ---\n", typeName);
    NewTestWave(scalarBinTester, sampleBytes, cpu_freq, seconds_per_wave);
    while (IsTesting(scalarBinTester))
    {
        BeginTime(scalarBinTester);
        for (size_t r = 0; r < repeatsPerSample; ++r)
            fnScalarBin(v, elementCount, b, res2);
        EndTime(scalarBinTester);
        CountBytes(scalarBinTester, sampleBytes);
        CountOps(scalarBinTester, elementCount * repeatsPerSample);
    }

    printf("--- %s AVX2 (256) ---\n", typeName);
    NewTestWave(avx2Tester, sampleBytes, cpu_freq, seconds_per_wave);
    while (IsTesting(avx2Tester))
    {
        BeginTime(avx2Tester);
        for (size_t r = 0; r < repeatsPerSample; ++r)
            fn256(v, elementCount, b, res3);
        EndTime(avx2Tester);
        CountBytes(avx2Tester, sampleBytes);
        CountOps(avx2Tester, elementCount * repeatsPerSample);
    }

    printf("--- %s AVX512 ---\n", typeName);
    NewTestWave(avx512Tester, sampleBytes, cpu_freq, seconds_per_wave);
    while (IsTesting(avx512Tester))
    {
        BeginTime(avx512Tester);
        for (size_t r = 0; r < repeatsPerSample; ++r)
            fn512(v, elementCount, b, res4);
        EndTime(avx512Tester);
        CountBytes(avx512Tester, sampleBytes);
        CountOps(avx512Tester, elementCount * repeatsPerSample);
    }

    bool allMatch = true;
    for (size_t i = 0; i < elementCount; ++i)
    {
        if (res1[i] != res2[i] || res1[i] != res3[i] || res1[i] != res4[i])
        {
            allMatch = false;
            printf(
                "%zu: scalar = %llu, scalar bin = %llu, AVX2 = %llu, AVX512 = %llu\n",
                i, (unsigned long long)res1[i], (unsigned long long)res2[i], (unsigned long long)res3[i], (unsigned long long)res4[i]
            );
        }
    }

    printf(allMatch ? "%s: all match\n\n" : "\n", typeName);
}

int main()
{
    InitializeOSMetrics();
    uint64_t cpu_freq = GetCPUFreqEstimate();

    size_t elementCount = 10000000;
    uint32_t seconds_per_wave = 3;

    // u8: range is too small for a safe (non-overflowing) bound to leave any interesting variety,
    // so it keeps the saturating-arithmetic approach instead and uses the full 0-255 range.
    RunTypeBenchmark<u8>("u8", elementCount, (u8)100, (u8)0, (u8)255, lpScalar, lpScalarBin, lp256, lp512, cpu_freq, seconds_per_wave);

    // u16/u32/u64: bound 'a' so a*b (the largest term) can never overflow, avoiding the need
    // for saturating instructions - minimum of 1 avoids a*b-2 underflowing when a*b < 2.
    RunTypeBenchmark<u16>("u16", elementCount, (u16)100, (u16)1, (u16)((std::numeric_limits<u16>::max)() / 100), lp<u16>, lp2<u16>, lp256_u16, lp512_u16, cpu_freq, seconds_per_wave);
    RunTypeBenchmark<u32>("u32", elementCount, (u32)100, (u32)1, (u32)((std::numeric_limits<u32>::max)() / 100), lp<u32>, lp2<u32>, lp256_u32, lp512_u32, cpu_freq, seconds_per_wave);
    RunTypeBenchmark<u64>("u64", elementCount, (u64)100, (u64)1, (u64)((std::numeric_limits<u64>::max)() / 100), lp<u64>, lp2<u64>, lp256_u64, lp512_u64, cpu_freq, seconds_per_wave);

    // f32/f64: huge dynamic range makes overflow a non-issue at these magnitudes, so a modest
    // fixed range is enough (no need to compute a max(T)/b style bound like the integer types).
    RunTypeBenchmark<f32>("f32", elementCount, (f32)100, (f32)1, (f32)10000, lp<f32>, lp2<f32>, lp256, lp512, cpu_freq, seconds_per_wave);
    RunTypeBenchmark<f64>("f64", elementCount, (f64)100, (f64)1, (f64)10000, lp<f64>, lp2<f64>, lp256, lp512, cpu_freq, seconds_per_wave);

    // ---- L1-resident pass: same buffer reused thousands of times per timed sample, so it never
    // leaves L1D (32KB on Zen4) - this takes memory/cache bandwidth out of the equation entirely,
    // isolating raw compute/instruction throughput instead.
    printf("\n==== L1-resident (compute-bound) ====\n\n");
    size_t l1Bytes = 4096;
    size_t repeatsPerSample = 4096;

    RunTypeBenchmark<u8>("u8 (L1)", l1Bytes / sizeof(u8), (u8)100, (u8)0, (u8)255, lpScalar, lpScalarBin, lp256, lp512, cpu_freq, seconds_per_wave, repeatsPerSample);
    RunTypeBenchmark<u16>("u16 (L1)", l1Bytes / sizeof(u16), (u16)100, (u16)1, (u16)((std::numeric_limits<u16>::max)() / 100), lp<u16>, lp2<u16>, lp256_u16, lp512_u16, cpu_freq, seconds_per_wave, repeatsPerSample);
    RunTypeBenchmark<u32>("u32 (L1)", l1Bytes / sizeof(u32), (u32)100, (u32)1, (u32)((std::numeric_limits<u32>::max)() / 100), lp<u32>, lp2<u32>, lp256_u32, lp512_u32, cpu_freq, seconds_per_wave, repeatsPerSample);
    RunTypeBenchmark<u64>("u64 (L1)", l1Bytes / sizeof(u64), (u64)100, (u64)1, (u64)((std::numeric_limits<u64>::max)() / 100), lp<u64>, lp2<u64>, lp256_u64, lp512_u64, cpu_freq, seconds_per_wave, repeatsPerSample);
    RunTypeBenchmark<f32>("f32 (L1)", l1Bytes / sizeof(f32), (f32)100, (f32)1, (f32)10000, lp<f32>, lp2<f32>, lp256, lp512, cpu_freq, seconds_per_wave, repeatsPerSample);
    RunTypeBenchmark<f64>("f64 (L1)", l1Bytes / sizeof(f64), (f64)100, (f64)1, (f64)10000, lp<f64>, lp2<f64>, lp256, lp512, cpu_freq, seconds_per_wave, repeatsPerSample);
}

