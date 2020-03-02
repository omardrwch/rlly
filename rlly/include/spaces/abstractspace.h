#ifndef __RLLY_ABSTRACT_SPACE_H__
#define __RLLY_ABSTRACT_SPACE_H__

/**
 * @file
 * @brief Class for definining observation and action spaces. 
 */
#include <vector>
#include <random>

namespace rlly
{
namespace spaces
{

/**
 * @brief Possible space names.
 */
enum spc_name {undefined, discrete, box}; 

/**
 * @brief Base class for observation and action spaces.
 * 
 */
template <typename T> 
class Space
{
public:
    Space(/* args */){};
    ~Space(){};
    /**
     * @brief Sample a value of the space with a uniform distribution
     */
    virtual T sample() {T foo; return foo;};

    /**
     * @brief Returns true if x belongs to the space, and false otherwise.
     * @param x 
     */
    virtual bool contains(T x) {return false;};

    /**
     * Name of the space (discrete, box, etc.)
     */
    spc_name name = undefined; 

    /**
     * Random number generator
     */
    std::mt19937 generator;

    /**
     * Size of the space (-1 for infinity or undefined )
     */
    int n = -1;
};

} // namespace spaces
} // namespace rlly

#endif
