#include <stdlib.h>
#include <random>
#include <assert.h> 
#include "discrete.h"

namespace rlly
{
namespace spaces
{

/*
Members of Discrete
*/ 

Discrete::Discrete()
{
    n = 0;
    name = discrete;
}

Discrete::Discrete(int _n, unsigned _seed /* = 42 */) 
{
    name = discrete;
    n = _n;
    generator.seed(_seed);
}

void Discrete::set_n(int _n)
{
    name = discrete;
    n = _n;
}

bool Discrete::contains(int x)
{
    return (x >= 0 && x < n);
}

int Discrete::sample()
{
    std::uniform_int_distribution<int> distribution(0,n-1);
    return distribution(generator);
}

}
}