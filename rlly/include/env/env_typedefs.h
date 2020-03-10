#ifndef __RLLY_ENV_TYPEDEFS_H__
#define __RLLY_ENV_TYPEDEFS_H__

#include <vector>
#include "abstractenv.h"
#include "space.h"

/**
 * @file
 * @brief Useful type definitions
 */


namespace rlly
{
namespace env
{

/**
 * @brief Base class for environments with finite states and finite actions.
 */
typedef Env<int, int, spaces::Discrete, spaces::Discrete> FiniteEnv;

/**
 * @brief Base class for continuous-state environments with finite actions.
 */
typedef Env<std::vector<double>, int, spaces::Box, spaces::Discrete> ContinuousStateEnv;

/**
 * @brief Base class for continuous-state environments with continuous actions.
 */
typedef Env<std::vector<double>, std::vector<double>, spaces::Box, spaces::Box> ContinuousEnv;

/**
 * @brief Base class for discrete-state environments with continuous actions.
 */
typedef Env<int, std::vector<double>, spaces::Discrete, spaces::Box> ContinuousActionEnv;

}  // namespace env
}  // namespace rlly


#endif