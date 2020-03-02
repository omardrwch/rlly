#include "discrete_reward.h"

namespace rlly
{
namespace env
{

DiscreteReward::DiscreteReward()
{
    noise_type = "none";
}

DiscreteReward::DiscreteReward(utils::vec::vec_3d _mean_rewards)
{
    mean_rewards = _mean_rewards;
    noise_type = "none";
}

DiscreteReward::DiscreteReward(utils::vec::vec_3d _mean_rewards, std::string _noise_type, std::vector<double> _noise_params)
{
    mean_rewards = _mean_rewards;
    noise_type = _noise_type;
    noise_params = _noise_params;
}

double DiscreteReward::sample(int state, int action, int next_state, utils::rand::Random randgen)
{
    double mean_r = mean_rewards[state][action][next_state];
    double noise;
    if (noise_type == "none")
        noise = 0;
    else if(noise_type == "gaussian")
    {
        assert(noise_params.size() == 1 && "noise type and noise params are not compatible");
        noise = randgen.sample_gaussian(0, noise_params[0]);
    }
    else
    {
        std::cerr << "Invalid noise type in DiscreteReward" << std::endl;
    }        
    return mean_r + noise;
}

}  // namespace env
}  // namespace rlly
