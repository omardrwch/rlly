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