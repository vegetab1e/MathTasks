#include <cstdint>
#ifdef DIAGNOSTIC_MODE
#include <iostream>
#endif
#include <omp.h>
#include <immintrin.h>
#include <emmintrin.h>

#include "vxor.h"
#ifdef BENCHMARK_MODE
#include "utils.h"
#endif

namespace
{
#ifdef BENCHMARK_MODE
const int NUM_THREADS = 1;
#else
const int NUM_THREADS = omp_get_num_procs();
#endif

#ifdef __GNUC__
inline __attribute__((always_inline))
#else
__forceinline
#endif
void
vXor16A(__m128i* buffer,
        __m128i const* p,
        __m128i const* end) noexcept
{
    __m128i vector = _mm_load_si128(p++);

    do
    {
        vector = _mm_xor_si128(vector, _mm_load_si128(p));
    }
    while (++p < end);

    _mm_store_si128(buffer, vector);
}

#ifdef __GNUC__
inline __attribute__((always_inline))
#else
__forceinline
#endif
void
vXor16U(__m128i* buffer,
        __m128i const* p,
        __m128i const* end) noexcept
{
    __m128i vector = _mm_loadu_si128(p++);

    do
    {
        vector = _mm_xor_si128(vector, _mm_loadu_si128(p));
    }
    while (++p < end);

    _mm_store_si128(buffer, vector);
}

#ifdef __GNUC__
inline __attribute__((always_inline))
#else
__forceinline
#endif
void
vXor32A(__m256i* buffer,
        __m256i const* p,
        __m256i const* end) noexcept
{
    __m256i vector = _mm256_load_si256(p++);

    do
    {
        vector = _mm256_xor_si256(vector, _mm256_load_si256(p));
    }
    while (++p < end);

    _mm256_store_si256(buffer, vector);
}

#ifdef __GNUC__
inline __attribute__((always_inline))
#else
__forceinline
#endif
void
vXor32U(__m256i* buffer,
        __m256i const* p,
        __m256i const* end) noexcept
{
    __m256i vector = _mm256_loadu_si256(p++);

    do
    {
        vector = _mm256_xor_si256(vector, _mm256_loadu_si256(p));
    }
    while (++p < end);

    _mm256_store_si256(buffer, vector);
}

//
// SSE2
//
#ifdef __GNUC__
inline __attribute__((always_inline))
#else
__forceinline
#endif
std::uint8_t
vXor(__m128i const* p, int n) noexcept
{
    alignas(__m128i) std::uint8_t buffer[sizeof(__m128i)];
    if ((reinterpret_cast<std::uintptr_t>(p) & 0xF) == 0)
        vXor16A(reinterpret_cast<__m128i*>(buffer), p, p + n);
    else
        vXor16U(reinterpret_cast<__m128i*>(buffer), p, p + n);

    *reinterpret_cast<std::uint64_t*>(buffer) ^= *(reinterpret_cast<std::uint64_t*>(buffer) + 1);
    *reinterpret_cast<std::uint32_t*>(buffer) ^= *(reinterpret_cast<std::uint32_t*>(buffer) + 1);
    *reinterpret_cast<std::uint16_t*>(buffer) ^= *(reinterpret_cast<std::uint16_t*>(buffer) + 1);

    return *buffer ^ *(buffer + 1);
}

//
// AVX2
//
#ifdef __GNUC__
inline __attribute__((always_inline))
#else
__forceinline
#endif
std::uint8_t
vXor(__m256i const* p, int n) noexcept
{
    alignas(__m256i) std::uint8_t buffer[sizeof(__m256i)];
    if ((reinterpret_cast<std::uintptr_t>(p) & 0x1F) == 0)
        vXor32A(reinterpret_cast<__m256i*>(buffer), p, p + n);
    else
        vXor32U(reinterpret_cast<__m256i*>(buffer), p, p + n);

    *reinterpret_cast<std::uint64_t*>(buffer) ^= *(reinterpret_cast<std::uint64_t*>(buffer) + 1) ^
                                                 *(reinterpret_cast<std::uint64_t*>(buffer) + 2) ^
                                                 *(reinterpret_cast<std::uint64_t*>(buffer) + 3);
    *reinterpret_cast<std::uint32_t*>(buffer) ^= *(reinterpret_cast<std::uint32_t*>(buffer) + 1);
    *reinterpret_cast<std::uint16_t*>(buffer) ^= *(reinterpret_cast<std::uint16_t*>(buffer) + 1);

    return *buffer ^ *(buffer + 1);
}

template<class T>
auto
vXor(const char* &p, int n)
{
    char byte = 0b0;
    const int chunk = (n /= sizeof(T)) / NUM_THREADS;
#pragma omp parallel if (chunk > 1) num_threads(NUM_THREADS) reduction(^:byte)
    if (omp_in_parallel())
    {
#if defined(DIAGNOSTIC_MODE) && !defined(BENCHMARK_MODE)
#pragma omp single
        std::cout << "\x1b[36mNumber of threads: \x1b[1m"
                  << omp_get_num_threads()
                  << "\x1b[0m\n";
#endif
        byte = vXor(reinterpret_cast<const T*>(p) + chunk * omp_get_thread_num(), chunk);
#pragma omp barrier
#pragma omp atomic update
        p += chunk * sizeof(T);
    }
    else
    {
        byte = vXor(reinterpret_cast<const T*>(p), n);
        p += n * sizeof(T);
    }

    return byte;
}

}

inline namespace OptimizedAlgorithms
{

char my_xor(const char* p, int n, bool force_sse2)
{
    if (!p || !n)
        return 0b0;

    omp_set_dynamic(0);
    omp_set_num_threads(NUM_THREADS);

#ifdef BENCHMARK_MODE
    uint64_t tsc;
    unsigned aux;
    tsc = __rdtscp(&aux);
#endif

    char const* end = p + n;
    char byte = ((n >= 2 * sizeof(__m256i)) && not force_sse2
                  ? vXor<__m256i>(p, n)
                  : (n >= 2 * sizeof(__m128i)
                     ? vXor<__m128i>(p, n)
                     : *p++));

    while (p < end)
        byte ^= *p++;

#ifdef BENCHMARK_MODE
    tsc = __rdtscp(&aux) - tsc;
    std::cout << "\x1b[4mПриблизительное\x1b[0m количество циклов на "
                 "\x1b[1mXOR с векторизацией:  \x1b[32m"
              << tsc << "\x1b[0m\n";
#endif

    return byte;
}

}

namespace ReferenceAlgorithms
{

char my_xor(const char* p, int n)
{
    if (!p || !n)
        return 0b0;

    omp_set_dynamic(0);
    omp_set_num_threads(NUM_THREADS);

#ifdef BENCHMARK_MODE
    uint64_t tsc;
    unsigned aux;
    tsc = __rdtscp(&aux);
#endif

    char byte = 0b0;
#pragma omp parallel for num_threads(NUM_THREADS) reduction(^:byte)
    for (int i = 0; i < n; ++i)
        byte ^= p[i];

#ifdef BENCHMARK_MODE
    tsc = __rdtscp(&aux) - tsc;
    std::cout << "\x1b[4mПриблизительное\x1b[0m количество циклов на "
                 "\x1b[1mXOR без векторизации: \x1b[32m"
              << tsc << "\x1b[0m\n";
#endif

    return byte;
}

}
