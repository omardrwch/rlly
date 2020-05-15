#include "env.h"
#include "env_factory.h"


namespace rlly
{
namespace env
{

FiniteEnv* make_finite_env(std::string env_name, utils::params::Params* env_params /*= nullptr*/)
{
    if (env_name == "GridWorld")
    {
        //  GridWorld(int _nrows, int _ncols, double fail_p = 0, double reward_smoothness = 0, double reward_sigma = 0)
        if(env_params != nullptr 
           && env_params->is_defined("nrows", "int")
           && env_params->is_defined("ncols", "int") 
           && env_params->is_defined("fail_p", "double")
           && env_params->is_defined("reward_smoothness", "double")
           && env_params->is_defined("reward_sigma", "double") 
           )
        {
            return new GridWorld( 
                                env_params->int_params["nrows"],
                                env_params->int_params["ncols"],
                                env_params->double_params["fail_p"],
                                env_params->double_params["reward_smoothness"],
                                env_params->double_params["reward_sigma"]
                                );
        }
    }
    std::cerr << " Error! Invalid name or missing parameters in make_finite_env. Returning nullptr" << std::endl;
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
    
    std::cerr << " Error! Invalid name or missing parameters in make_continuous_env. Returning nullptr" << std::endl;
    return nullptr;
}

ContinuousActionEnv* make_continuous_action_env(std::string env_name, utils::params::Params* env_params /*= nullptr*/)
{
    std::cerr << " Error! Invalid name or missing parameters in make_continuous_action_env. Returning nullptr" << std::endl;
    return nullptr;
}


} // namespace env
} // namespace rlly
