#ifndef __RLLY_MOUNTAINCAR_H__
#define __RLLY_MOUNTAINCAR_H__

#include <vector>
#include "env_typedefs.h"
#include "utils.h"

namespace rlly
{
namespace env
{
/**
 * @brief 1d Mountain car environment
 * @details
 *    State space = (position, velocity)
 *                   position: value in [-1.2, 0.6]
 *                   velocity: value in [-0.07, 0.07]
 *    Action space: Discrete(3)
 * 
 *    The initial position is a random number (in the position range). 
 *    The initial velocity is 0.
 * 
 *   Action 0: negative force
 *   Action 1: do nothing
 *   Action 2: positive force
 * 
 *   The terminal state is (goal_position, goal_velocity)
 * 
 *   A reward of 0 is obtained everywhere, except for the terminal state, where the reward is 1.
 */
class MountainCar: public ContinuousStateEnv, public rlly::utils::render::RenderInterface2D<std::vector<double>>
{
    
public:
    /**
     * Indices of position and velocity in the state vector.
     */
    enum StateLabel
    {
        position = 0, velocity = 1
    };

    MountainCar();
    std::vector<double> reset();
    StepResult<std::vector<double>> step(int action) override;

    /**
     * Get scene representing a given state
     */
    utils::render::Scene2D get_scene_for_render2d(std::vector<double> state_var) override;

    /**
     * Returns background for rendering 
     */
    utils::render::Scene2D get_background_for_render2d() override;

    /**
     * Clone 
     */
    std::unique_ptr<ContinuousStateEnv> clone() const override;

protected:
    /**
     * @brief Returns true if the state is terminal.
     */
    bool is_terminal(std::vector<double> state);
    /**
     * Position at the terminal state
     */
    double goal_position;
    /**
     * Velocity at the terminal state
     */
    double goal_velocity;


private:
    /**
     * Force magnitude.
     */
    static constexpr double force = 0.001;
    /**
     * Gravity.
     */
    static constexpr double gravity = 0.0025;

};
} // namespace env
} // namespace rlly

#endif
