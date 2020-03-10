#ifndef __RLLY_ENV_TYPEDEFS_H__
#define __RLLY_ENV_TYPEDEFS_H__

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


}  // namespace env
}  // namespace rlly


#endif