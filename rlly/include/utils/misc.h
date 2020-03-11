#ifndef __RLLY_MISC_H__
#define __RLLY_MISC_H__

#include <iostream>
#include <vector>

/**
 * @file 
 * Other utils.
 */

namespace rlly
{
namespace utils
{

    /**
     * @brief Clamp a value between an upper and lower bound.
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

    /**
     * @brief Binary search in a sorted vector with increasing values.
     * @details if vec = {x_0, x_1, ..., x_n}, with x_0 <= x_1 <= ... <= x_n,
     * returns the value i such that vec[i] <= val < vec[i+1]. If there is no
     * such value, returns -1.
     * @param val value to be searched
     * @param vec vector in which to search the interval where val is.
     * @param l   index where to start the search (default = 0)
     * @param r   index where to end the search (default = -1). If -1, it is set to vec.size()-1
     */
    int binary_search(double val, std::vector<double> vec, int l = 0, int r = -1);


    /**
     * @brief Binary search in d dimensions. Returns flat index. 
     * @param d_val vector to be searched
     * @param bins 2d array such that bins[i] represents the intervals where to search for d_val[i]. Represents
     * a set of hypercubes in R^d
     * @return flat index in column-major order corresponding to the hypercube where d_val lives
     */
    int binary_search_nd(std::vector<double> d_val, std::vector<std::vector<double>> bins);
}
}  

#endif
