#ifndef __RLLY_DISCRETE_SPACE_H__
#define __RLLY_DISCRETE_SPACE_H__

#include <vector>
#include "abstractspace.h"

/**
 * @file
 * @brief Class for definining discrete observation and action spaces.
 */

namespace rlly
{
namespace spaces
{

/**
 * @brief Class for discrete spaces.
 * Set of possible values = {0, ..., n-1}
 */
class Discrete: public Space<int>
{
public:

    /**
     * @brief Default constructor
     */
    Discrete();

    /**
     * @param _n: Value of n
     * @param _seed: seed for random number generation (default = 42)
     */
    Discrete(int _n, unsigned _seed = 42);
    ~Discrete(){};

    // Methods

    /**
     * @brief Set the value of n, that defines the discrete space {0, ..., n-1}
     */
    void set_n(int _n);

    // Methods of base class
    int sample();
    bool contains(int x);
};


}
}


#endif
