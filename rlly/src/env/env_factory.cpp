#include "env.h"
#include "env_factory.h"


namespace rlly
{
namespace env
{

FiniteEnv* make_finite_env(std::string env_name)
{
    return nullptr;
}

ContinuousStateEnv*  make_continuous_state_env(std::string env_name)
{
    if      (env_name == "SquareWorld")     return new SquareWorld();
    else if (env_name == "WallSquareWorld") return new WallSquareWorld();
    else if (env_name == "MountainCar")     return new MountainCar();
    else if (env_name == "CartPole")        return new CartPole();
    else if (env_name == "WallSquareWorld") return new WallSquareWorld();
    else if (env_name.rfind("ChangingWallSquareWorld", 0) == 0)
    {
        std::string period_str = env_name.substr(env_name.find("_") + 1); 
        int period = 1;
        try
        {
            period = std::stoi(period_str);
            return new ChangingWallSquareWorld(period);        
        }
        catch(const std::exception& e){}
    }
    std::cerr << " Error! Invalid name in make_continuous_state_env. Returning nullptr" << std::endl;
    return nullptr;
}

ContinuousEnv*       make_continuous_env(std::string env_name)
{
    if       (env_name == "MovingBall") return new MovingBall();
    return nullptr;
}

ContinuousActionEnv* make_continuous_action_env(std::string env_name)
{
    return nullptr;
}


} // namespace env
} // namespace rlly