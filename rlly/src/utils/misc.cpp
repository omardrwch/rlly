#include <vector>
#include "misc.h"

namespace rlly
{
namespace utils
{

/**
 *  Map a value x in [x0, x1] linearly to the range [y1, y2]
 */
double linear_map(double x, double x1, double x2, double y1, double y2)
{
    if (x1 == x2 || y1 == y2) return 0.0;
    double a = (y2 - y1)/(x2 - x1);
    double b = y1 - a*x1;
    return a*x + b;
}

int binary_search(double val, std::vector<double> vec, int l /*= 0*/, int r /*= -1*/) 
{ 
    if (r == - 1) r = vec.size()-1;

    if (r > l) 
    { 
        int mid = l + (r - l) / 2; 
        if (vec[mid] <= val && vec[mid+1] > val) 
            return mid;
        if (val >= vec[mid+1])
            return binary_search(val, vec, mid+1, r);
        if (val < vec[mid])
            return binary_search(val, vec, l, mid);
    } 
    return -1; 
} 

int binary_search_nd(std::vector<double> d_val, std::vector<std::vector<double>> bins)
{
    unsigned int dim = bins.size();
    int flat_index = 0;
    int aux = 1;
    if (dim != d_val.size()) throw;
    for(unsigned int dd = 0; dd < dim; dd++)
    {
        int index_dd = binary_search(d_val[dd], bins[dd]);
        if (index_dd == -1) throw;
        flat_index += aux*index_dd;
        aux *= (bins[dd].size()-1);
    }
    return flat_index;
}

}
}
