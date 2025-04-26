#include <cstdint>
#include <cstring>

#include <limits>

#include <omp.h>
#include <immintrin.h>

#ifdef DIAGNOSTIC_MODE
#include <iostream>
#include "utils.h"
#endif

#include "mmul.h"

namespace
{

constexpr std::size_t CHUNK_SIZE = sizeof(__m256d) / sizeof(double);
const int NUM_THREADS = omp_get_num_procs();
#ifdef DIAGNOSTIC_MODE
unsigned threads = 0;
#endif
inline void
rcMulA(const double* row, const double* end, const double* col, double* x)
{
#ifdef DIAGNOSTIC_MODE
#pragma omp atomic update
    threads |= (1 << omp_get_thread_num());
#endif
    while (row <= end - CHUNK_SIZE)
    {
        // Умножение строки матрицы А на вектор b (поэлементно)
        __m256d vector = _mm256_mul_pd(_mm256_load_pd(row),
                                       _mm256_load_pd(col));

        // Горизонтальное сложение (попарно)
        vector = _mm256_hadd_pd(vector, vector);
        // Вертикальное сложение (поэлементно) с перестановкой частей вектора (пополам)
        vector = _mm256_add_pd(vector, _mm256_permute2f128_pd(vector, vector, 0b00000001));

        // Получение итогового значения
        *x += _mm256_cvtsd_f64(vector);

        row += CHUNK_SIZE;
        col += CHUNK_SIZE;
    }

    // Обработка оставшихся элементов
    while (row < end)
        *x += *row++ * *col++;
}

inline void
rcMulU(const double* row, const double* end, const double* col, double* x)
{
#ifdef DIAGNOSTIC_MODE
#pragma omp atomic update
    threads |= (1 << omp_get_thread_num());
#endif
    while (row <= end - CHUNK_SIZE)
    {
        // Умножение строки матрицы А на вектор b (поэлементно)
        __m256d vector = _mm256_mul_pd(_mm256_loadu_pd(row),
                                       _mm256_loadu_pd(col));

        // Горизонтальное сложение (попарно)
        vector = _mm256_hadd_pd(vector, vector);
        // Вертикальное сложение (поэлементно) с перестановкой частей вектора (пополам)
        vector = _mm256_add_pd(vector, _mm256_permute2f128_pd(vector, vector, 0b00000001));

        // Получение итогового значения
        *x += _mm256_cvtsd_f64(vector);

        row += CHUNK_SIZE;
        col += CHUNK_SIZE;
    }

    // Обработка оставшихся элементов
    while (row < end)
        *x += *row++ * *col++;
}

}

inline namespace OptimizedAlgorithms
{

void mul(const double* A, const double* b, double* x, int n)
{
    static_assert(std::numeric_limits<double>::is_iec559,
                  "Requires IEEE 754 floating point!");

    if (!A || !b || !x || !n)
        return;

    omp_set_dynamic(0);
    omp_set_num_threads(NUM_THREADS);

    std::memset(x, 0, n * sizeof(*x));

#ifdef DIAGNOSTIC_MODE
#pragma omp atomic write
    threads = 0;
#endif
    if ((reinterpret_cast<std::uintptr_t>(A) & 0b11111) == 0 &&
        (reinterpret_cast<std::uintptr_t>(b) & 0b11111) == 0)
#pragma omp parallel for num_threads(NUM_THREADS)
        for (int i = 0; i < n; ++i)
            rcMulA(A + i * n,
                   A + (i + 1) * n,
                   b,
                   x + i);
    else
#pragma omp parallel for num_threads(NUM_THREADS)
        for (int i = 0; i < n; ++i)
            rcMulU(A + i * n,
                   A + (i + 1) * n,
                   b,
                   x + i);
#ifdef DIAGNOSTIC_MODE
    std::cout << "\x1b[36mNumber of threads: \x1b[1m"
              << trueBits(threads)
              << "\x1b[0m\n";
#endif
}

}

namespace ReferenceAlgorithms
{

void mul(const double* A, const double* b, double* x, int n) noexcept
{
    if (!A || !b || !x || !n)
        return;

    omp_set_dynamic(0);
    omp_set_num_threads(NUM_THREADS);

#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < n; ++i)
    {
        const double* row = &A[i * n];

        x[i] = row[0] * b[0];
        for (int j = 1; j < n; ++j)
            x[i] += row[j] * b[j];
    }
}

}
