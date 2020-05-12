#ifndef __RLLY_ENV_FACTORY_H__
#define __RLLY_ENV_FACTORY_H__

/**
 * @file 
 * Factory methods to create environments.
 */

#include <string>
#include "env_typedefs.h"
#include "utils.h"

namespace rlly
{
namespace env
{

FiniteEnv*           make_finite_env(           std::string env_name, utils::params::Params* env_params = nullptr);
ContinuousStateEnv*  make_continuous_state_env( std::string env_name, utils::params::Params* env_params = nullptr);
ContinuousEnv*       make_continuous_env(       std::string env_name, utils::params::Params* env_params = nullptr);
ContinuousActionEnv* make_continuous_action_env(std::string env_name, utils::params::Params* env_params = nullptr);
} // namespace env
} // namespace rlly


#endif