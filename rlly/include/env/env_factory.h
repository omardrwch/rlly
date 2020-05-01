#ifndef __RLLY_ENV_FACTORY_H__
#define __RLLY_ENV_FACTORY_H__

/**
 * @file 
 * Factory methods to create environments.
 */

#include <string>
#include "env_typedefs.h"

namespace rlly
{
namespace env
{

FiniteEnv*           make_finite_env(std::string env_name);
ContinuousStateEnv*  make_continuous_state_env(std::string env_name);
ContinuousEnv*       make_continuous_env(std::string env_name);
ContinuousActionEnv* make_continuous_action_env(std::string env_name);

} // namespace env
} // namespace rlly


#endif