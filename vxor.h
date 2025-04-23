#pragma once

inline namespace OptimizedAlgorithms
{

/**
 * @brief Побайтовый XOR массива
 * @details Тестовое задание #1
 * В данный момент оптимизирован не сам алгоритм,
 * а его реализация, то еесть одноимённая функция
 * из пространства имён ReferenceAlgorithms
 * @param[in] p Массив байт
 * @param[in] n Размер массива
 * @return Свёрнутый XORом в один байт массив
 */
char my_xor(const char* p, int n, bool force_sse2 = false);

}

namespace ReferenceAlgorithms
{

/**
 * @brief Референтный вариант функции
 * @details Для проверки результата оптимизации
 * базового алгоритма. Эффективность низкая, но
 * зато высокая надёжность. Сигнатура идентична
 */
char my_xor(const char* p, int n);

}
