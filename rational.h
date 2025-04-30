#pragma once

#include <string>
#include <iostream>

#include <type_traits>

#if __cplusplus >= 201703L
#include <numeric>
#else
// Функция для нахождения НОД,
// реализует алгоритм Евклида
template<class T>
auto gcd(T a, T b)
{
    while (b != 0)
    {
        T t = b;
        b = a % b;
        a = t;
    }

    return a;
}
#endif

template<class T, class U>
constexpr
std::enable_if_t<std::is_signed<T>::value || std::is_signed<U>::value,
std::make_signed_t<std::common_type_t<T, U>>>
getCommonType() noexcept
{
    static_assert(false, "Calling getCommonType() is ill-formed");
}

template<class T, class U>
constexpr
std::enable_if_t<std::is_unsigned<T>::value && std::is_unsigned<U>::value,
std::common_type_t<T, U>>
getCommonType() noexcept
{
    static_assert(false, "Calling getCommonType() is ill-formed");
}

template<class T, class U>
using CommonType = decltype(getCommonType<T, U>());

template<class T, class EnableIf = void>
class Rational;

template<class T>
class Rational<T, std::enable_if_t<std::is_integral<T>::value>> final
{
public:
    template<class U, class EnableIf>
    friend class Rational;

    explicit Rational(T num = T(), T den = T(1))
        : num_(num)
        , den_(den)
    {
        if (den == 0)
            throw std::invalid_argument("Denominator cannot be zero!");

        simplify();
    }

    Rational(const Rational& rhs) noexcept
    {
        num_ = rhs.num_;
        den_ = rhs.den_;

        simplify();
    }

    template<class U>
    Rational(const Rational<U>& rhs)
    {
        if (std::is_unsigned<T>::value && (rhs.num_ < 0 || rhs.den_ < 0))
            throw std::invalid_argument("An unsigned number cannot be less than zero!");

        num_ = static_cast<T>(rhs.num_);
        den_ = static_cast<T>(rhs.den_);

        simplify();
    }

    auto operator+(const Rational& rhs) const
    {
        // Нужно расширить это
        // решение сокращением
        if (den_ == rhs.den_)
            return Rational(num_ + rhs.num_, den_);

        return Rational(num_ * rhs.den_ + rhs.num_ * den_,
                        den_ * rhs.den_);
    }

    template<class U>
    auto operator+(const Rational<U>& rhs) const
    {
        return Rational<CommonType<T, U>>(num_ * rhs.den_ + rhs.num_ * den_,
                                          den_ * rhs.den_);
    }

    std::string print() const
    {
        return (std::to_string(num_) + "/" + std::to_string(den_));
    }

    friend
    std::ostream& operator<<(std::ostream& out, const Rational& rhs)
    {
        return (out << rhs.num_ << "/" << rhs.den_);
    }

private:
    void simplify() noexcept
    {
#if __cplusplus >= 201703L
        const auto gcd = std::gcd(num_, den_);
#else
        const auto gcd = ::gcd(num_, den_);
#endif
        num_ /= gcd;
        den_ /= gcd;

#if __cplusplus >= 201703L
        if constexpr(std::is_signed_v<T>)
            if (den_ < 0)
            {
                num_ = -num_;
                den_ = -den_;
            }
#else
        if (den_ < 0)
            negate<T>();
#endif
    }

#if __cplusplus < 201703L
    template<class U>
    std::enable_if_t<std::is_signed<U>::value>
    negate() noexcept
    {
        num_ = -num_;
        den_ = -den_;
    }

    template<class U>
    std::enable_if_t<std::is_unsigned<U>::value>
    negate() noexcept
    {
    }
#endif

    T num_;
    T den_;
};
