#include <cstdint>

#include <omp.h>
#include <immintrin.h>
#include <emmintrin.h>

#include "vxor.h"

namespace
{

constexpr int NUM_THREADS = 8;

inline void
vXor16A(__m128i* buffer,
      __m128i const* &p,
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
vXor32A(__m256i* buffer,
        __m256i const* &p,
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
vXor16U(__m128i* buffer,
      __m128i const* &p,
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
vXor32U(__m256i* buffer,
        __m256i const* &p,
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
vXor16(__m128i const* &p, int n) noexcept
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
vXor32(__m256i const* &p, int n) noexcept
{
    alignas(__m256i) std::uint8_t buffer[sizeof(__m256i)];
    if ((reinterpret_cast<std::uintptr_t>(p) & 0x1F) == 0)
        vXor32A(reinterpret_cast<__m256i*>(buffer), p, p + n);
    else
        vXor32U(reinterpret_cast<__m256i*>(buffer), p, p + n);

    auto pointer = reinterpret_cast<__m128i const*>(buffer);

    return vXor16(pointer, 2);
}

}

namespace OptimizedAlgorithms
{

char my_xor(const char* p, int n) noexcept
{
    if (!p || !n)
        return 0b0;

    char const* end = p + n;
    char byte = (n >= 2 * sizeof(__m256i))
                ? vXor32(reinterpret_cast<__m256i const* &>(p), n / sizeof(__m256i))
                : ((n >= 2 * sizeof(__m128i))
                   ? vXor16(reinterpret_cast<__m128i const* &>(p), n / sizeof(__m128i))
                   : *p++);

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
#ifdef NDEBUG
    omp_set_dynamic(0);
    omp_set_num_threads(NUM_THREADS);
#pragma omp parallel for num_threads(NUM_THREADS) reduction(^:byte)
#endif
    for (int i = 0; i < n; ++i)
        byte ^= p[i];

    return byte;
}

}
