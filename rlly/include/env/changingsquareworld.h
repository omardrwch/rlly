#ifndef __RLLY_CHANGING_SQUAREWORLD_H__
#define __RLLY_CHANGING_SQUAREWORLD_H__

#include <vector>
#include "env_typedefs.h"
#include "utils.h"

/**
 * @file 
 * @brief Contains a class for the ChangingSquareWorld environment.
 */

namespace rlly
{
namespace env
{

/**
 * @brief ChangingSquareWorld environment with states in [0, 1]^2 and 4 actions 
 * @details 
 *      The agent starts at (start_x, start_y) and, in each state, it can take for actions (0 to 3) representing a
 *      displacement of (-d, 0), (d, 0), (0, -d) and (0, d), respectively.
 *          
 *      The immediate reward received in each state s = (s_x, s_y) is, for any action a,
 *          r(s, a) = exp( - ((s_x-goal_x)^2 + (s_y-goal_y)^2)/(2*reward_smoothness^2)  )
 * 
 *      The position of the reward changes every <period> episodes.
 *      The direction of the actions also change.
 *   
 */
class ChangingSquareWorld: public ContinuousStateEnv, public rlly::utils::render::RenderInterface2D<std::vector<double>>
{
private:
    // Coordinates of start position
    double start_x = 0.5;
    double start_y = 0.5;

    // Index of the current configuration (from 0 to 3)
    int current_reward_configuration = 0;
    int current_transition_configuration = 0;

    
    // Number of configurations
    int n_configurations = 4;

    // Coordinates of the goal position in each configuration
    std::vector<double> goal_x_vec = {0.9, 0.9, 0.1, 0.1};
    std::vector<double> goal_y_vec = {0.9, 0.1, 0.9, 0.1};

    // (-d, 0), (d, 0), (0, -d) and (0, d)
    // Action displacement vector in each configuration
    std::vector<std::vector<double>> action_0_displacement_vec =  {  
                                                                    {-0.1,  0.0},  // left
                                                                    { 0.1,  0.0},  // right
                                                                    { 0.0, -0.1},  // down
                                                                    { 0.0,  0.1}   // up
                                                                  };

    std::vector<std::vector<double>> action_1_displacement_vec =  {  
                                                                    { 0.1,  0.0},  // right
                                                                    {-0.1,  0.0},  // left
                                                                    { 0.0,  0.1},  // up
                                                                    { 0.0, -0.1}   // down
                                                                  }; 

    std::vector<std::vector<double>> action_2_displacement_vec =  {  
                                                                    { 0.0, -0.1},  // down
                                                                    { 0.0,  0.1},  // up
                                                                    {-0.1,  0.0},  // left
                                                                    { 0.1,  0.0}   // right
                                                                  }; 

    std::vector<std::vector<double>> action_3_displacement_vec =  {  
                                                                    { 0.0,  0.1},   // up
                                                                    { 0.0, -0.1},   // down
                                                                    { 0.1,  0.0},   // right
                                                                    {-0.1,  0.0}    // left
                                                                  }; 

    // Reward smoothness
    double reward_smoothness = 0.05;

    // Standard dev of reward noise (gaussian)
    double reward_noise_stdev = 0.01;

    // Standard dev of transition noise (gaussian) 
    double transition_noise_stdev = 0.01;

    // Clip to domain
    void clip_to_domain(double &xx, double &yy);

public:
    ChangingSquareWorld(int _period, bool _changing_reward = true, bool _changing_transition = true);
    ~ChangingSquareWorld(){};

    // period of changes in the environment
    int period;

    // True if rewards change
    bool changing_reward = true;

    // True if transitions change
    bool changing_transition = true;

    // current episode, increased every time reset() is called
    int current_episode = 0;

    std::unique_ptr<ContinuousStateEnv> clone() const override;
    std::vector<double> reset() override;
    env::StepResult<std::vector<double>> step(int action) override;

    utils::render::Scene2D get_scene_for_render2d(std::vector<double> state_var) override;    
    utils::render::Scene2D get_background_for_render2d();

};

}  // namespace env
}  // namespace rlly


#endif