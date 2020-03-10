#ifndef __RLLY_WRAPPER_TYPEDEFS_H__
#define __RLLY_WRAPPER_TYPEDEFS_H__

#include "basewrapper.h"
#include "space.h"

/**
 * @file
 * @brief Useful type definitions for wrappers
 */


namespace rlly
{
namespace wrappers
{

/**
 * @brief Base class for wrappers for environments with finite states and finite actions.
 */
typedef Wrapper<int, int, spaces::Discrete, spaces::Discrete> FiniteEnvWrapper;

/**
 * @brief Base class for wrappers for continuous-state environments with finite actions.
 */
typedef Wrapper<std::vector<double>, int, spaces::Box, spaces::Discrete> ContinuousStateEnvWrapper;


}  // namespace env
}  // namespace rlly


#endif