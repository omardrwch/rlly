#ifndef __RLLY_DISCRETE_REWARD_H__
#define __RLLY_DISCRETE_REWARD_H__

/**
 * @file
 * @brief Contains class for defining rewards in finite MDPs.
 */

#include <vector>
#include <string>
#include <iostream>
#include <assert.h>
#include "utils.h"

namespace rlly
{
namespace env
{
    
/**
 * Class for defining reward functions in finite MDPs.
 */ 
class DiscreteReward
{

public:
    /**
     * Default constructor
     */
    DiscreteReward();
    /**
     * Create "reward without noise" object
     * @param _mean_rewards
     */
    DiscreteReward(utils::vec::vec_3d _mean_rewards);
    /**
     * Create "reward with noise" object
     * @param _mean_rewards
     * @param _noise_type 
     * @param _noise_params
     */
    DiscreteReward(utils::vec::vec_3d _mean_rewards, std::string _noise_type, std::vector<double> _noise_params);
    ~DiscreteReward(){};

    /**
    * 3d vector such that mean_rewards[s][a][s'] is the mean reward obtained when the
    * state s' is reached by taking action a in state s.
    */
    utils::vec::vec_3d mean_rewards;

    /**
     * String describing the type of noise
     * "none": zero noise
     * "gaussian": zero-mean Gaussian distribution with variance given in noise_params
     */
    std::string noise_type;

    /**
    * Vector of noise parameters
    * - Gaussian noise: noise_params = [variance]
    */
    std::vector<double> noise_params;

    /**
     * Get a reward sample at (state, action, next_state)
     * @param state
     * @param action
     * @param next_state
     * @param randgen random number generator for sampling the noise
     */
    double sample(int state, int action, int next_state, utils::rand::Random randgen);
};

} // namespace env
} // namespace rlly


#endif
