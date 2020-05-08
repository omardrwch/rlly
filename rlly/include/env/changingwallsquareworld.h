#ifndef __RLLY_CHANGING_WALL_SQUAREWORLD_H__
#define __RLLY_CHANGING_WALL_SQUAREWORLD_H__

#include <vector>
#include "env_typedefs.h"
#include "utils.h"

/**
 * @file 
 * @brief Contains a class for the ChangingWallSquareWorld environment.
 */

namespace rlly
{
namespace env
{

/**
 * @brief ChangingWallSquareWorld environment with states in [0, 1]^2 and 4 actions 
 * @details 
 *      The agent starts at (start_x, start_y) and, in each state, it can take for actions (0 to 3) representing a
 *      displacement of (-d, 0), (d, 0), (0, -d) and (0, d), respectively.
 *          
 *      The immediate reward received in each state s = (s_x, s_y) is, for any action a,
 *          r(s, a) = exp( - ((s_x-goal_x)^2 + (s_y-goal_y)^2)/(2*reward_smoothness^2)  )
 * 
 * 
 *      There is a wall that makes it more difficult for the agent to find the reward. 
 *      The position of passage in the wall changes every N episodes.
 * 
 *      The position of the reward also changes.
 * 
 *      The direction of the actions also change (left <-> right / up <-> down)
 *   
 */
class ChangingWallSquareWorld: public ContinuousStateEnv, public rlly::utils::render::RenderInterface2D<std::vector<double>>
{
private:
    // Coordinates of start position
    double start_x = 0.1;
    double start_y = 0.1;

    // Index of the current configuration (from 0 to 3)
    int current_configuration = 0;
    
    // Number of configurations
    int n_configurations = 4;

    // Coordinates of the goal position in each configuration
    std::vector<double> goal_x_vec = {0.85, 0.85, 0.85, 0.15};
    std::vector<double> goal_y_vec = {0.85, 0.85, 0.15, 0.85};

    // Coordinates of the wall position in each configuration
    std::vector<double> wall_x0_vec = {0.45, 0.20, 0.45, 0.00};
    std::vector<double> wall_x1_vec = {0.55, 1.00, 0.55, 0.80};
    std::vector<double> wall_y0_vec = {0.20, 0.45, 0.00, 0.45};
    std::vector<double> wall_y1_vec = {1.00, 0.55, 0.80, 0.55};

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

    // Check if (xx, yy) is a point inside a wall
    bool is_inside_wall(double xx, double yy);

    // Clip to domain
    void clip_to_domain(double &xx, double &yy);

public:
    ChangingWallSquareWorld(int _period);
    ~ChangingWallSquareWorld(){};

    // period of changes in the environmen
    int period;

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