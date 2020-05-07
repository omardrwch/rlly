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
 */
class ChangingWallSquareWorld: public ContinuousStateEnv, public rlly::utils::render::RenderInterface2D<std::vector<double>>
{
private:
    // Coordinates of start position
    double start_x = 0.1;
    double start_y = 0.1;

    // Coordinates of goal position (where reward is max)
    double goal_x = 0.85;
    double goal_y = 0.85;

    // Initial coordinates of the walls
    double wall_1_x0 = 0.45;
    double wall_1_x1 = 0.55;
    double wall_1_y0 = 0.0;
    double wall_1_y1 = 0.7;

    double wall_2_x0 = 0.45;
    double wall_2_x1 = 0.55;
    double wall_2_y0 = 0.8;
    double wall_2_y1 = 1.0;

    // y-displacement of the wall passage when there is a change
    double wall_displacement = 0.5;

    // bool representing the current passage position
    bool current_position = 0;

    // Action displacement
    double displacement = 0.1;

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