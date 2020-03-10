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
typedef IsomorphicWrapper<int, int, spaces::Discrete, spaces::Discrete> FiniteEnvWrapper;

/**
 * @brief ContinuousStateEnv -> ContinuousStateEnv wrapper
 */
typedef IsomorphicWrapper<std::vector<double>, int, spaces::Box, spaces::Discrete> ContinuousStateEnvWrapper;

/**
 * @brief ContinuousEnv -> ContinuousEnv wrapper
 */
typedef IsomorphicWrapper<std::vector<double>, std::vector<double>, spaces::Box, spaces::Box> ContinuousEnvWrapper;

/**
 * @brief ContinuousActionEnv -> ContinuousActionEnv wrapper
 */
typedef IsomorphicWrapper<int, std::vector<double>, spaces::Discrete, spaces::Box> ContinuousActionEnvWrapper;


}  // namespace env
}  // namespace rlly


#endif