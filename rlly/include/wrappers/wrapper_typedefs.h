#ifndef __RLLY_WRAPPER_TYPEDEFS_H__
#define __RLLY_WRAPPER_TYPEDEFS_H__

#include "isomorphicwrapper.h"
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
 * @brief FiniteEnv -> FiniteEnv wrapper
 */
typedef IsomorphicWrapper<spaces::Discrete, spaces::Discrete> FiniteEnvWrapper;

/**
 * @brief ContinuousStateEnv -> ContinuousStateEnv wrapper
 */
typedef IsomorphicWrapper<spaces::Box, spaces::Discrete> ContinuousStateEnvWrapper;

/**
 * @brief ContinuousEnv -> ContinuousEnv wrapper
 */
typedef IsomorphicWrapper<spaces::Box, spaces::Box> ContinuousEnvWrapper;

/**
 * @brief ContinuousActionEnv -> ContinuousActionEnv wrapper
 */
typedef IsomorphicWrapper<spaces::Discrete, spaces::Box> ContinuousActionEnvWrapper;


}  // namespace env
}  // namespace rlly


#endif