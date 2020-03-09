#include <stdlib.h>
#include <random>
#include <assert.h> 
#include "box.h"

namespace rlly
{
namespace spaces
{

/*
Members of Box
*/
Box::Box()
{
    // Do nothing. low and high are empty vectors.
}

Box::Box(std::vector<double> _low, std::vector<double> _high, unsigned _seed /* = 42 */)
{
    low = _low;
    high = _high;
    size = _low.size();
    generator.seed(_seed);
    assert(size == _high.size() && "The size of _low and _high must be the same.");
}    

void Box::set_bounds(std::vector<double> _low, std::vector<double> _high)
{
    low = _low; 
    high = _high;
    size = _low.size();
    assert(size == _high.size() && "The size of _low and _high must be the same.");
}

bool Box::contains(std::vector<double> x)
{
    bool contains = true;
    if (x.size() != size)
    {
        contains = false;
    }
    for(unsigned int i = 0; i < x.size(); i++)
    {
        contains = contains && (x[i] >= low[i] && x[i] <= high[i]);
    }
    return contains;
}

std::vector<double> Box::sample()
{
    // uniform real distribution
    std::uniform_real_distribution<double> distribution(0.0,1.0);

    std::vector<double> sampled_state(size);
    for(unsigned int i = 0; i < size; i++)
    {
        double a;
        double b;
        a = low[i];
        b = high[i];
        sampled_state[i] = a + (b-a)*distribution(generator);
    } 
    return sampled_state;
}

}
}