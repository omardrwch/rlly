#include "env.h"
#include "env_factory.h"


namespace rlly
{
namespace env
{

FiniteEnv* make_finite_env(std::string env_name, utils::params::Params* env_params /*= nullptr*/)
{
    return nullptr;
}

ContinuousStateEnv*  make_continuous_state_env(std::string env_name, utils::params::Params* env_params /*= nullptr*/)
{
    if      (env_name == "SquareWorld")     return new SquareWorld();
    else if (env_name == "WallSquareWorld") return new WallSquareWorld();
    else if (env_name == "MountainCar")     return new MountainCar();
    else if (env_name == "CartPole")        return new CartPole();
    else if (env_name == "WallSquareWorld") return new WallSquareWorld();
    else if (env_name == "ChangingSquareWorld")
    {
        if(env_params != nullptr && env_params->is_defined("period", "int"))
            return new ChangingSquareWorld(env_params->int_params["period"]);                  
    }
    else if (env_name == "ChangingWallSquareWorld")
    {
        if(env_params != nullptr && env_params->is_defined("period", "int"))
            return new ChangingWallSquareWorld(env_params->int_params["period"]);   
    }
    else if(env_name == "ChangingRewardSquareWorld")
    {
        if(env_params != nullptr && env_params->is_defined("period", "int"))
            return new ChangingSquareWorld(env_params->int_params["period"], true, false); 
    }
    else if(env_name == "ChasingBlobs")
    {
        if(env_params != nullptr && env_params->is_defined("period", "int"))
            return new ChasingBlobs(env_params->int_params["period"]); 
    }
    std::cerr << " Error! Invalid name or missing parameters in make_continuous_state_env. Returning nullptr" << std::endl;
    return nullptr;
}

ContinuousEnv*       make_continuous_env(std::string env_name, utils::params::Params* env_params /*= nullptr*/)
{
    if       (env_name == "MovingBall") return new MovingBall();
    return nullptr;
}

ContinuousActionEnv* make_continuous_action_env(std::string env_name, utils::params::Params* env_params /*= nullptr*/)
{
    return nullptr;
}


} // namespace env
} // namespace rlly
