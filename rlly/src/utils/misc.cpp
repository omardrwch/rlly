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
}
}
