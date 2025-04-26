#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <cfloat>
#include <cmath>

#include <clocale>

#include <iostream>

#include "utils.h"
#include "vxor.h"
#include "peval.h"
#include "mmul.h"
#include "rational.h"

namespace
{

constexpr double TOLERANCE = 0.0001;

constexpr std::size_t OPERATION_COUNT = 100U;
constexpr std::size_t VECTOR_LENGTH = 1'000U;
constexpr std::size_t ARRAY_SIZE = 500'000U;

void runTest1()
{
    // Первое число в функции - количество восьмёрок байт, т.е.
    // групп по восемь байт, где в первом байте единичным будет
    // нулевой бит, а в последнем байте единичным будет седьмой
    // бит. Т.о. если сделать побайтовый XOR чётного количества
    // таких групп, то в результате будет 0, а если нечётного -
    // 255. Второе число в функции - количество бит, равных или
    // единице, если количество групп чётное, или нулю, если их
    // количество нечётное. Т.е. второе число - тоже количество
    // байт с соответствующим единичным битом, однако XOR этого
    // остатка байт с тем, что получилось после XORа восьмёрок,
    // даст именно такой результат. В описании функции тоже про
    // это есть. Лично для меня - просто и понятно, но кому как
    alignas(32) std::uint8_t bytes[getByteCount(120'000U, 3U)];

    fillByteArray(bytes);

    const std::uint8_t byte = my_xor(reinterpret_cast<char*>(bytes),
                                     sizeof(bytes));

    printBytes(byte);
}

void runTest1Ex(std::size_t array_size)
{
    std::uint8_t* bytes = static_cast<std::uint8_t*>(_mm_malloc(array_size, 32U));

    fillByteArrayEx(bytes, array_size);

    const std::uint8_t byte0 = my_xor(reinterpret_cast<char*>(bytes),
                                      static_cast<int>(array_size));
    const std::uint8_t byte1 = ReferenceAlgorithms::
                               my_xor(reinterpret_cast<char*>(bytes),
                                      static_cast<int>(array_size));

    if (byte0 != byte1)
    {
        std::cout << "\x1b[1;31mНесовпадают байты!\x1b[0m\n";

        printBytes(byte0);
        printBytes(byte1);
    }
    else
    {
        std::cout << "Размер массива для побайтового XORа: \x1b[1m"
                  << array_size
                  << "\x1b[0m. Выполнено без ошибок\n";
    }

    _mm_free(bytes);
}

void runTest2()
{
    const int n = 4;
    const double a[n + 1] = { 9.35, -7.412, 5.0, 3.83, -1.3033 };
    const double x = 0.578;

    const double value = polyeval(a, n, x);

    printPolynom(a, n, x, value);
}

void runTest3()
try {
    auto r0 = Rational<unsigned>(1, 2);
    auto r1 = Rational<short>(3, -4);
    auto r2 = Rational<int>(r0);
    auto r3 = Rational<int>(r2);
    auto r4 = Rational<int>(145638, 9540);
    // Порождают исключения,
    // оставил для проверки!
#if 0
    auto r5 = Rational<unsigned>(r1);
    auto r6 = Rational<unsigned>(0, 0);
#endif
    Rational<long> r7;
    Rational<int> r8 = r0 + r1;
    Rational<int> r9 = r2 + r3;
    
    std::cout << "Сумма рациональных чисел " \
                 "\x1b[1m" + r0.print() + "\x1b[0m и " \
                 "\x1b[1m" + r1.print() + "\x1b[0m равна " \
                 "\x1b[1m" << r8 << "\x1b[0m\n";
}
catch (const std::exception& e)
{
    std::cout << e.what() << "\n";
}

void runTest4Ex(std::size_t vector_length)
{
    const std::size_t byte_count = vector_length * sizeof(double);
    double* A  = static_cast<double*>(_mm_malloc(byte_count * vector_length, 32U));
    double* b  = static_cast<double*>(_mm_malloc(byte_count, 32U));
    double* x0 = static_cast<double*>(_mm_malloc(byte_count, 32U));
    double* x1 = static_cast<double*>(_mm_malloc(byte_count, 32U));

    if (!A || !b || !x0 || !x1)
    {
        std::cerr << "\x1b[1;31mОшибка выделения памяти!\x1b[0m\n";
        return;
    }

    fillVector(A, vector_length * vector_length);
    fillVector(b, vector_length);

    mul(A, b, x0, static_cast<int>(vector_length));
    ReferenceAlgorithms::
    mul(A, b, x1, static_cast<int>(vector_length));

    std::size_t error_count = 0;
    for (std::size_t i = 0; i < vector_length; ++i)
        if (std::fabs(x0[i] - x1[i]) >= TOLERANCE)
            ++error_count;
    
    if (error_count)
    {
        std::cout << "Несовпадает \x1b[1;31m" << error_count
                  << "\x1b[0m значений из \x1b[1;31m" << vector_length
                  << "\x1b[0m!\n";

        printVector(x0, vector_length);
        printVector(x1, vector_length);
    }
    else
    {
        std::cout << "Размер квадратной матрицы и вектора-столбца для умножения: \x1b[1m"
                  << vector_length
                  << "\x1b[0m. Выполнено без ошибок\n";
    }

    _mm_free(A);
    _mm_free(b);
    _mm_free(x0);
    _mm_free(x1);
}

}

int main(int argc, char** argv)
{
    std::setlocale(LC_ALL, "");

    runTest2();
    runTest3();

#ifdef BENCHMARK_MODE
    checkRdtscp();
#endif

    std::srand(static_cast<unsigned>(__rdtsc()));
    for (std::size_t i = 0; i < OPERATION_COUNT; ++i)
    {
        runTest1Ex(std::rand() % ARRAY_SIZE    + ARRAY_SIZE);
        runTest4Ex(std::rand() % VECTOR_LENGTH + VECTOR_LENGTH);
#ifdef DIAGNOSTIC_MODE
        std::cout << "\x1b[35mOperation number: \x1b[1m"
                  << i + 1
                  << "\x1b[0m\n";
#endif
    }

    return 0;
}
