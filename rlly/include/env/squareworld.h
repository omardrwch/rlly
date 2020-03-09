#ifndef __RLLY_SQUAREWORLD_H__
#define __RLLY_SQUAREWORLD_H__

#include <vector>
#include "abstractenv.h"
#include "utils.h"

/**
 * @file 
 * @brief Contains a class for the SquareWorld environment.
 */

namespace rlly
{
namespace env
{

/**
 * @brief SquareWorld environment with states in [0, 1]^2 and 4 actions 
 * @details 
 *      The agent starts at (start_x, start_y) and, in each state, it can take for actions (0 to 3) representing a
 *      displacement of (-d, 0), (d, 0), (0, -d) and (0, d), respectively.
 *          
 *      The immediate reward received in each state s = (s_x, s_y) is, for any action a,
 *          r(s, a) = exp( - ((s_x-goal_x)^2 + (s_y-goal_y)^2)/(2*reward_smoothness^2)  )
 */
class SquareWorld: public Env<std::vector<double>, int>
{
private:
    // Coordinates of start position
    double start_x = 0.1;
    double start_y = 0.1;

    // Coordinates of goal position (where reward is max)
    double goal_x = 0.75;
    double goal_y = 0.75;

    // Action displacement
    double displacement = 0.1;

    // Reward smoothness
    double reward_smoothness = 0.1;

    // Standard dev of reward noise (gaussian)
    double reward_noise_stdev = 0.01;

    // Standard dev of transition noise (gaussian) 
    double transition_noise_stdev = 0.01;

public:
    SquareWorld();
    ~SquareWorld(){};


    /**
    * State (observation) space
    */
    spaces::Box observation_space;

    /**
    *  Action space
    */
    spaces::Discrete action_space;

    std::unique_ptr<Env<std::vector<double>, int>> clone() const override;
    std::vector<double> reset() override;
    env::StepResult<std::vector<double>> step(int action) override;

    utils::render::Scene get_scene_for_render2d(std::vector<double> state_var) override;    
    utils::render::Scene get_background_for_render2d();

};


}  // namespace env
}  // namespace rlly


#endif