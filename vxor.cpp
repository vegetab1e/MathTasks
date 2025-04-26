#include <cstdint>
#ifdef DIAGNOSTIC_MODE
#include <iostream>
#endif
#include <omp.h>
#include <immintrin.h>
#include <emmintrin.h>

#include "vxor.h"

namespace
{

constexpr int NUM_THREADS = 8;

inline void
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

inline void
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

inline void
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

inline void
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

inline std::uint8_t
vXor16(__m128i const* p, int n) noexcept
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

inline std::uint8_t
vXor32(__m256i const* p, int n) noexcept
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

}

inline namespace OptimizedAlgorithms
{

char my_xor(const char* p, int n) noexcept
{
    if (!p || !n)
        return 0b0;

    char const* end = p + n;
    char byte = 0b0;

    omp_set_dynamic(0);
    omp_set_num_threads(NUM_THREADS);

    if (n >= 2 * sizeof(__m256i))
    {
        const int chunk = (n /= sizeof(__m256i)) / NUM_THREADS;
#pragma omp parallel if (chunk > 1) num_threads(NUM_THREADS) reduction(^:byte)
        if (omp_in_parallel())
        {
#ifdef DIAGNOSTIC_MODE
#pragma omp single
            std::cout << "\x1b[36mNumber of threads: \x1b[1m"
                      << omp_get_num_threads()
                      << "\x1b[0m\n";
#endif
            byte = vXor32(reinterpret_cast<__m256i const*>(p) + chunk * omp_get_thread_num(), chunk);
        }
        else
        {
            byte = vXor32(reinterpret_cast<__m256i const*>(p), n);
        }

        p += (chunk ? chunk * NUM_THREADS : n) * sizeof(__m256i);
    }
    else if (n >= 2 * sizeof(__m128i))
    {
        byte = vXor16(reinterpret_cast<__m128i const*>(p), n /= sizeof(__m128i));
        p += n * sizeof(__m128i);
    }

    while (p < end)
        byte ^= *p++;

    return byte;
}

}

namespace ReferenceAlgorithms
{

char my_xor(const char* p, int n) noexcept
{
    if (!p || !n)
        return 0b0;

    char byte = 0b0;

    omp_set_dynamic(0);
    omp_set_num_threads(NUM_THREADS);

#pragma omp parallel for num_threads(NUM_THREADS) reduction(^:byte)
    for (int i = 0; i < n; ++i)
        byte ^= p[i];

    return byte;
}

}
