#ifndef __RLLY_BOX_SPACE_H__
#define __RLLY_BOX_SPACE_H__

#include <vector>
#include "abstractspace.h"


/**
 * @file
 * @brief Class for definining box observation and action spaces.
 */

namespace rlly
{
namespace spaces
{

/**
 * @brief Class for box spaces in R^n
 */
class Box: public Space<std::vector<double>>
{
public:

    /**
     * @brief Default constructor
     */
    Box();

    /**
     * @param _low: array contaning the lower bounds of the box
     * @param _high: array containing the upper bounds of the box
     * @param _seed: seed for random number generation (default = 42)
     */
    Box(std::vector<double> _low, std::vector<double> _high, unsigned _seed = 42);
    ~Box(){};

    // Methods
    /**
     * @brief Define the lower and upper bounds of the box
     * @param _low: array contaning the lower bounds of the box
     * @param _high: array containing the upper bounds of the box
     */ 
    void set_bounds(std::vector<double> _low, std::vector<double> _high);

    // Methods of base class
    std::vector<double> sample();
    bool contains(std::vector<double> x);

    /**
     * Name of the space
     */
    spc_name name = box;

    // Attributes

    /**
     * Size of the fox
     */
    int size; 

    /**
     * lower bounds of the box
     */
    std::vector<double> low;

    /**
     * upper bounds of the box
     */
    std::vector<double> high;
};

}
}


#endif
