#include <fixed.hpp>
#include <ios.hpp>
#include <iostream>
// #include "alphas.hpp"
// #include "bn_variables.hpp"
#include "weights.hpp"
using namespace fpm;
using position = fpm::fixed<std::int32_t, std::int64_t, 24>;
using namespace std;
int main()
{
    fixed_8_24 fixed_array[] = {
        fixed_8_24(0.5),
        fixed_8_24(1.25),
        fixed_8_24(2.75),
        fixed_8_24(3.0),
        fixed_8_24(4.5)}; // OK: explicit construction from float
    fixed_8_24 a{127};
    cout << a << endl;
    // printing size of b
    // cout << sizeof(fixed_array) << endl;
}