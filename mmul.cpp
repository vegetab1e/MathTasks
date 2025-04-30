#include <cstdint>
#include <cstring>

#include <limits>

#include <omp.h>

#include <immintrin.h>

#if defined(BENCHMARK_MODE) && defined(__GNUC__)
#include <sys/resource.h>
#endif

#if defined(DIAGNOSTIC_MODE) || defined(BENCHMARK_MODE)
#include <iostream>
#include "utils.h"
#endif

#include "mmul.h"

extern const int NUM_THREADS;

namespace
{

constexpr std::size_t CHUNK_SIZE = sizeof(__m256d) / sizeof(double);

#if defined(DIAGNOSTIC_MODE) && !defined(BENCHMARK_MODE)
unsigned threads_map;
#endif

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
rcMulA(const double* row, const double* end, const double* col, double* x) noexcept
{
#if defined(DIAGNOSTIC_MODE) && !defined(BENCHMARK_MODE)
#pragma omp atomic update
    threads_map |= (1 << omp_get_thread_num());
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

inline
#ifdef __GNUC__
__attribute__((always_inline))
#else
__forceinline
#endif
void
rcMulU(const double* row, const double* end, const double* col, double* x) noexcept
{
#if defined(DIAGNOSTIC_MODE) && !defined(BENCHMARK_MODE)
#pragma omp atomic update
    threads_map |= (1 << omp_get_thread_num());
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

    std::memset(x, 0, n * sizeof(*x));

#if defined(DIAGNOSTIC_MODE) && !defined(BENCHMARK_MODE)
#pragma omp atomic write
    threads_map = 0b0;
#endif

#ifdef BENCHMARK_MODE
#ifdef __GNUC__
    getrusage(RUSAGE_SELF, &usage[0]);
#endif
    tsc[0] = __rdtscp(&aux);
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

#ifdef BENCHMARK_MODE
    tsc[1] = __rdtscp(&aux);
#ifdef __GNUC__
    getrusage(RUSAGE_SELF, &usage[1]);
#endif

    printPerfInfo(tsc, 
#ifdef __GNUC__
                  usage,
#endif // __GNUC__
                  "MUL с векторизацией");
#endif // BENCHMARK_MODE

#if defined(DIAGNOSTIC_MODE) && !defined(BENCHMARK_MODE)
    std::cout << "\x1b[1;36mNumber of threads: "
              << trueBits(threads_map)
              << "\x1b[0m\n";
#endif
}

}

namespace ReferenceAlgorithms
{

void mul(const double* A, const double* b, double* x, int n)
{
    if (!A || !b || !x || !n)
        return;

#ifdef BENCHMARK_MODE
#ifdef __GNUC__
    getrusage(RUSAGE_SELF, &usage[0]);
#endif
    tsc[0] = __rdtscp(&aux);
#endif

#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < n; ++i)
    {
        const double* row = &A[i * n];

        x[i] = row[0] * b[0];
        for (int j = 1; j < n; ++j)
            x[i] += row[j] * b[j];
        // Если массив размещён в памяти
        // не по строкам, а по столбцам,
        // то нужно изменить перебор так
        //  x[j] += row[j] * b[i];
        // и начинать тогда нужно с нуля
    }

#ifdef BENCHMARK_MODE
    tsc[1] = __rdtscp(&aux);
#ifdef __GNUC__
    getrusage(RUSAGE_SELF, &usage[1]);
#endif

    printPerfInfo(tsc, 
#ifdef __GNUC__
                  usage,
#endif // __GNUC__
                  "MUL без векторизации");
#endif // BENCHMARK_MODE
}

}
