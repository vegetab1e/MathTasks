#pragma once

#include <cstdlib>

#include <limits>
#include <typeinfo>
#include <type_traits>

#include <algorithm>

#include <iostream>
#include <iomanip>

#include <string>

#if defined(_MSC_VER)

#include <intrin.h>

#define CHAR_SIZE CHAR_BIT 

#elif defined(__GNUC__)

#include <sys/resource.h>
#include <sys/time.h>

#include <cpuid.h>
#include <cxxabi.h>
#include <x86intrin.h>

#define CHAR_SIZE __CHAR_BIT__

#else
#error Failing compilation
#endif

template<class T>
using IsUInt = std::enable_if_t<std::is_integral<T>::value &&
                                std::is_unsigned<T>::value,
                                T>;

/**
 * @brief Получить (рассчитать) количество байт для XORа
 * @details Для удобства проверки результата XORа:
 * - если i чётное, то j бит из восьми после XORа будут равны единице, а остальные - нулю;
 * - если i нечётное, то j бит из восьми после XORа будут равны нулю, а остальные - единице.
 * Исполдьзовать нужно совместно с fillByteArray(), иначе такого результата не получится
 * @param[in] i Количество групп по восемь байт (с одним единичным битом в заданной позиции)
 * @param[in] j Количество контрольных байт (от нуля до восьми включительно)
 * @return Суммарное количество байт
 */
template<class T>
constexpr IsUInt<T>
getByteCount(T i, T j) noexcept
{
    return (i * CHAR_SIZE + j);
}

template<std::size_t N, class T, IsUInt<T> = T()>
constexpr void 
fillByteArray(T (&bytes)[N]) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        // В каждый байт проставляется один единичный
        // бит от нулевого до старшего и так по кругу
        bytes[i] = 1ULL << (i % (sizeof(T) * CHAR_SIZE));
}

template<class T, IsUInt<T> = T()>
void 
fillByteArrayEx(T* bytes, std::size_t N)
{
    if (!bytes || !N)
        return;

    std::srand(static_cast<unsigned>(__rdtsc()));

    for (std::size_t i = 0; i < N; ++i)
        bytes[i] = std::rand() % (std::numeric_limits<T>::max() + 1U);
}

template<std::size_t N, class T>
void 
fillByteArrayEx(T (&bytes)[N])
{
    fillByteArrayEx(bytes, N);
}

template<class T>
std::enable_if_t<std::is_arithmetic<T>::value>
fillVector(T* vector, std::size_t N)
{
    if (!vector || !N)
        return;

    std::srand(static_cast<unsigned>(__rdtsc()));

    for (std::size_t i = 0; i < N; ++i)
        vector[i] = (std::rand() % 100 + 1) / 1E+5;
}

template<std::size_t N, class T>
void
fillVector(T (&vector)[N])
{
    fillVectorEx(vector, N);
}

template<std::size_t N, class T, IsUInt<T> = T()>
constexpr void
printBits(char (&buffer)[N], T bytes) noexcept
{
    constexpr std::size_t bit_count = std::min(N - 1, sizeof(T) * CHAR_SIZE);
    for (std::size_t i = 0; i < bit_count; ++i)
        buffer[i] = !!(bytes & (1ULL << i)) + '0';

    buffer[bit_count] = '\0';
}

template<class T, IsUInt<T> = T()>
constexpr std::size_t
trueBits(T bytes) noexcept
{
    std::size_t bit_count = 0;
    for (std::size_t i = 0; i < sizeof(T) * CHAR_SIZE; ++i)
        bit_count += !!(bytes & (1ULL << i));

    return bit_count;
}

template<class T, IsUInt<T> = T()>
void
printBytes(T bytes)
{
    // Последним будет признак конца строки
    char buffer[sizeof(T) * CHAR_SIZE + 1];

    printBits(buffer, bytes);

    std::cout << "Результат побайтового XORа массива: [\x1b[1m"
#ifdef __GNUC__
              << abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, nullptr)
#else
              << typeid(T).name()
#endif
              << "\x1b[0m] \x1b[1m" << buffer
              << "\x1b[0m (bit order - \x1b[1mLE\x1b[0m)\n";
}

template<class T>
std::enable_if_t<std::is_arithmetic<T>::value>
printVector(const T* vector, std::size_t vector_length, std::size_t line_length = 5)
{
    if (!vector || !vector_length)
        return;

    using std::cout;

    std::cout << "Результат умножения матрицы на вектор:\n";

    const auto precision{cout.precision()};
    cout << std::fixed << std::setprecision(std::numeric_limits<T>::digits10);

    for (std::size_t i = 0; i < vector_length; ++i)
        cout << std::setw(2 * std::numeric_limits<T>::digits10)
             << vector[i]
             << (!((i + 1) % line_length) && (i < vector_length - 1)
                 ? "\n"
                 : " ");

    cout << std::defaultfloat << std::setprecision(precision) << std::endl;
}

template<class T>
std::enable_if_t<std::is_arithmetic<T>::value>
printPolynom(const T* coefficients, std::size_t degree, T x, T value)
{
    std::cout << "Значение многочлена \x1b[1m";

    for (std::size_t i = 0; i <= degree; ++i)
        std::cout << (i
                      ? (coefficients[i] < 0 ? " - " : " + ")
                      : (coefficients[i] < 0 ? "-" : ""))
                  << std::abs(coefficients[i])
                  << (i
                      ? ("x^" + std::to_string(i)).c_str()
                      : "");
    
    std::cout << "\x1b[0m в точке \x1b[1m" << x
              << "\x1b[0m равно \x1b[1m" << value
              << "\x1b[0m\n";
}

inline void printPerfInfo(
    uint64_t (&tsc)[2],
#ifdef __GNUC__
    rusage (&usage)[2],
#endif
    const char* op_name)
{
    std::cout << "\x1b[4mКоличество тактов\x1b[0m на "
              << "\x1b[1m" + std::string(op_name) + ":\t\x1b[32m"
              << tsc[1] - tsc[0] << "\x1b[0m\n";
#ifdef __GNUC__
    timeval tsv[3]{};
    timersub(&usage[1].ru_utime, &usage[0].ru_utime, &tsv[0]);
    timersub(&usage[1].ru_stime, &usage[0].ru_stime, &tsv[1]);
    timeradd(&tsv[0], &tsv[1], &tsv[2]);

    std::cout << "\x1b[4mВремя затраченное\x1b[0m на "
                 "\x1b[1m" + std::string(op_name) + ":\t\x1b[32m"
              << (tsv[2].tv_sec  ? std::to_string(tsv[2].tv_sec)  + " с "  : "")
              << (tsv[2].tv_usec ? std::to_string(tsv[2].tv_usec) + " мкс" : "")
              << "\x1b[0m\n";
#endif
}

inline bool checkRdtscp() noexcept
{
#if defined (__GNUC__)
    unsigned cpu_id[4];
    __get_cpuid(0x80000001,
                &cpu_id[0],
                &cpu_id[1],
                &cpu_id[2],
                &cpu_id[3]);
#elif defined(_MSC_VER)
    int cpu_id[4];
    __cpuid(cpu_id, 0x80000001);
#else
#error Failing compilation
#endif
    return (cpu_id[3] & (1 << 27));
}

#undef CHAR_SIZE
