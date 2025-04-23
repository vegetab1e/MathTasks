#include <cstdint>

#include <omp.h>

#include <immintrin.h>
#include <emmintrin.h>

#if defined(BENCHMARK_MODE) && defined(__GNUC__)
#include <sys/resource.h>
#endif

#if defined(DIAGNOSTIC_MODE) || defined(BENCHMARK_MODE)
#include <iostream>
#include "utils.h"
#endif

#include "vxor.h"

#define XOR_PAIR(p) (*(p) ^= *((p) + 1))

extern const int NUM_THREADS;

namespace
{

#ifdef BENCHMARK_MODE
#ifdef __GNUC__
rusage usage[2];
#endif
unsigned aux;
uint64_t tsc[2];
#endif

inline
#ifdef __GNUC__
__attribute__((always_inline))
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

inline
#ifdef __GNUC__
__attribute__((always_inline))
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

inline
#ifdef __GNUC__
__attribute__((always_inline))
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

inline
#ifdef __GNUC__
__attribute__((always_inline))
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
inline
#ifdef __GNUC__
__attribute__((always_inline))
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

    XOR_PAIR(reinterpret_cast<std::uint64_t*>(buffer));
    XOR_PAIR(reinterpret_cast<std::uint32_t*>(buffer));
    XOR_PAIR(reinterpret_cast<std::uint16_t*>(buffer));

    return *buffer ^ *(buffer + 1);
}

//
// AVX2
//
inline
#ifdef __GNUC__
__attribute__((always_inline))
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

    _mm_store_si128(reinterpret_cast<__m128i*>(buffer),
                    _mm_xor_si128(_mm_load_si128(reinterpret_cast<const __m128i*>(buffer)),
                                  _mm_load_si128(reinterpret_cast<const __m128i*>(buffer) + 1)));

    XOR_PAIR(reinterpret_cast<std::uint64_t*>(buffer));
    XOR_PAIR(reinterpret_cast<std::uint32_t*>(buffer));
    XOR_PAIR(reinterpret_cast<std::uint16_t*>(buffer));

    return *buffer ^ *(buffer + 1);
}

template<class T>
inline
#ifdef __GNUC__
__attribute__((always_inline))
#else
__forceinline
#endif
char
vXor(const char* &p, int &n)
{
    char const* end = p + n;
    char byte = 0b0;

    const int chunk_size = (n /= sizeof(T)) / NUM_THREADS;
#pragma omp parallel if (chunk_size > 1) num_threads(NUM_THREADS) reduction(^:byte)
    if (omp_in_parallel())
    {
#if defined(DIAGNOSTIC_MODE) && !defined(BENCHMARK_MODE)
#pragma omp single
        std::cout << "\x1b[1;36mNumber of threads: "
                  << omp_get_num_threads()
                  << "\x1b[0m\n";
#endif
        byte = vXor(reinterpret_cast<const T*>(p) + chunk_size * omp_get_thread_num(),
                    chunk_size);
#pragma omp barrier
#pragma omp atomic update
        p += chunk_size * sizeof(T);
    }
    else
    {
        byte = vXor(reinterpret_cast<const T*>(p), n);
        p += n * sizeof(T);
    }

    n = static_cast<int>(end - p);

    return byte;
}

}

inline namespace OptimizedAlgorithms
{

char my_xor(const char* p, int n, bool force_sse2)
{
    if (!p || !n)
        return 0b0;

#ifdef BENCHMARK_MODE
#ifdef __GNUC__
    getrusage(RUSAGE_SELF, &usage[0]);
#endif
    tsc[0] = __rdtscp(&aux);
#endif

    char const* end = p + n;
    char byte = 0b0;
    
    if ((n >= 2 * sizeof(__m256i)) and not force_sse2)
        byte = vXor<__m256i>(p, n);
    else if (n >= 2 * sizeof(__m128i))
        byte = vXor<__m128i>(p, n);
    else
        byte = *p++;

    while (p < end)
        byte ^= *p++;

#ifdef BENCHMARK_MODE
    tsc[1] = __rdtscp(&aux);
#ifdef __GNUC__
    getrusage(RUSAGE_SELF, &usage[1]);
#endif

    printPerfInfo(tsc, 
#ifdef __GNUC__
                  usage,
#endif // __GNUC__
                  "XOR с векторизацией");
#endif // BENCHMARK_MODE

    return byte;
}

}

namespace ReferenceAlgorithms
{

char my_xor(const char* p, int n)
{
    if (!p || !n)
        return 0b0;

#ifdef BENCHMARK_MODE
#ifdef __GNUC__
    getrusage(RUSAGE_SELF, &usage[0]);
#endif
    tsc[0] = __rdtscp(&aux);
#endif

    char byte = 0b0;
#pragma omp parallel for num_threads(NUM_THREADS) reduction(^:byte)
    for (int i = 0; i < n; ++i)
        byte ^= p[i];

#ifdef BENCHMARK_MODE
    tsc[1] = __rdtscp(&aux);
#ifdef __GNUC__
    getrusage(RUSAGE_SELF, &usage[1]);
#endif

    printPerfInfo(tsc, 
#ifdef __GNUC__
                  usage,
#endif // __GNUC__
                  "XOR без векторизации");
#endif // BENCHMARK_MODE

    return byte;
}

}

#undef XOR_PAIR
