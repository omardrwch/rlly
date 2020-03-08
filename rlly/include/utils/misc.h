#ifndef __RLLY_MISC_H__
#define __RLLY_MISC_H__

#include <iostream>

/**
 * @file 
 * Other utils.
 */

namespace rlly
{
namespace utils
{

    // this should be defined in C++17
    /**
     * @brief Clamp a value between an upper and lower bound. Requires C++17.
     * @param v value to be clampled
     * @param lo lower bound
     * @param hi upper bound
     * @tparam T type of v, lo and hi
     */
    template<class T>
    constexpr const T& clamp( const T& v, const T& lo, const T& hi )
    {
        assert( !(hi < lo) );
        return (v < lo) ? lo : (hi < v) ? hi : v;
    }

    /**
     *  Map a value x in [x0, x1] linearly to the range [y1, y2]
     */
    double linear_map(double x, double x1, double x2, double y1, double y2);
}
}  

#endif
