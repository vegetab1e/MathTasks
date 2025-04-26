#pragma once

inline namespace OptimizedAlgorithms
{

/**
 * @brief Умножение матрицы A на вектор b
 * @details Тестовое задание #4
 * В данный момент оптимизирован не сам алгоритм,
 * а его реализация, то еесть одноимённая функция
 * из пространства имён ReferenceAlgorithms
 * @param[in] A Квадратная матрица размера n на n
 * @param[in] b Вектор-столбец размера n
 * @param[out] x Вектор-столбец размера n
 * @param[in] n Размер A, b и x
 */
void mul(const double* A,
         const double* b,
         double* x,
         int n);

}

namespace ReferenceAlgorithms
{

/**
 * @brief Референтный вариант функции
 * @details Для проверки результата оптимизации
 * базового алгоритма. Эффективность низкая, но
 * зато высокая надёжность. Сигнатура идентична
 */
void mul(const double* A,
         const double* b,
         double* x,
         int n);

}
