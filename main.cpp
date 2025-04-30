#include <cstdlib>
#include <clocale>

#include <cstdint>
#include <cfloat>

#include <iostream>

#include <omp.h>

#include "utils.h"
#include "vxor.h"
#include "mmul.h"
#include "peval.h"
#include "rational.h"

#ifdef BENCHMARK_MODE
extern const int NUM_THREADS = 1;
#else
extern const int NUM_THREADS = omp_get_num_procs();
#endif

namespace
{

// Десяти итераций более чем достаточно для
// оценки эффективности реализаций, так как
// объёмы используемых данных очень большие
constexpr std::size_t OPERATION_COUNT = 10U;
// ЭТО МИНИМАЛЬНОЕ ЗНАЧЕНИЕ, МАКСИМАЛЬНОЕ
// БУДЕТ В ДВА РАЗА БОЛЬШЕ МИНУС ЕДИНИЦА!
// (12 800 ^ 2 * 8) / (1024 ^ 2) = 1 250 мегабайт
// на квадратную матрицу из 64-битных даблов и по
// (12 800 * 8) / (1024 ^ 2) = 100 килобайт на
// три вектора-столбца из 64-битных даблов для
// умножения на эту матрицу одного из них двумя
// способами и сравнения полученных двух других
constexpr std::size_t VECTOR_LENGTH = 12'800U;
// ЭТО МИНИМАЛЬНОЕ ЗНАЧЕНИЕ, МАКСИМАЛЬНОЕ
// БУДЕТ В ДВА РАЗА БОЛЬШЕ МИНУС ЕДИНИЦА!
// 1 гигабайт на массив для сворачивания XORом
// двумя способами и сравнения полученной пары
// байт
constexpr std::size_t ARRAY_SIZE = 1'073'741'824U;

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
    std::cout << "Размер массива байт для сворачивания XORом:\t\x1b[1m"
              << array_size / 1'048'576. << " МБ\x1b[0m\n";

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
        std::cout << "Выполнено без ошибок\n";
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
    auto r4 = Rational<int>(145'638, 9'540);
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

    // Пример с переполнением, исправить!
    auto r10 = Rational<int>(1, 1'000'000);
    auto r11 = Rational<int>(1, 1'000'000);
    auto r12 = r10 + r11;
    std::cout << "Сумма рациональных чисел " \
                 "\x1b[1m" + r10.print() + "\x1b[0m и " \
                 "\x1b[1m" + r11.print() + "\x1b[0m равна " \
                 "\x1b[1m" << r12 << "\x1b[0m\n";
}
catch (const std::exception& e)
{
    std::cout << e.what() << "\n";
}

void runTest4Ex(std::size_t vector_length)
{
    std::cout << "Размер матрицы и вектора для перемножения:\t\x1b[1m"
              << vector_length << "\x1b[0m\n";

    const std::size_t byte_count = vector_length * sizeof(double);
    double* A  = static_cast<double*>(_mm_malloc(byte_count * vector_length, 32U));
    double* b  = static_cast<double*>(_mm_malloc(byte_count, 32U));
    double* x0 = static_cast<double*>(_mm_malloc(byte_count, 32U));
    double* x1 = static_cast<double*>(_mm_malloc(byte_count, 32U));

    if (!A || !b || !x0 || !x1)
    {
        std::cerr << "\x1b[1;31mОшибка выделения памяти!\x1b[0m\n";
        std::exit(EXIT_FAILURE);
    }

    fillVector(A, vector_length * vector_length);
    fillVector(b, vector_length);

    mul(A, b, x0, static_cast<int>(vector_length));
    ReferenceAlgorithms::
    mul(A, b, x1, static_cast<int>(vector_length));

    std::size_t error_count = 0;
    for (std::size_t i = 0; i < vector_length; ++i)
        if (std::fabs(x0[i] - x1[i]) >= DBL_EPSILON)
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
        std::cout << "Выполнено без ошибок\n";
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

#ifdef BENCHMARK_MODE
    if (checkRdtscp())
        std::cout << "\x1b[1;33mRDTSCP is supported\x1b[0m\n";
    else
        std::cout << "\x1b[1;31mRDTSCP is not supported\x1b[0m\n";
#endif

    omp_set_dynamic(0);
    omp_set_num_threads(NUM_THREADS);

    runTest2();
    runTest3();

    std::srand(static_cast<unsigned>(__rdtsc()));
    for (std::size_t i = 0; i < OPERATION_COUNT; ++i)
    {
#ifdef DIAGNOSTIC_MODE
        std::cout << "\x1b[1;34mOperation number: "
                  << i + 1
                  << "\x1b[0m\n";
#endif
        runTest1Ex(std::rand() % ARRAY_SIZE    + ARRAY_SIZE);
        runTest4Ex(std::rand() % VECTOR_LENGTH + VECTOR_LENGTH);
    }

    return 0;
}
