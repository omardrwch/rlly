#ifndef __RLLY_FALLINGBALL_H__
#define __RLLY_FALLINGBALL_H__

#include <vector>
#include "lqr.h"
#include "utils.h"

namespace rlly
{
namespace env
{

/**
 * @brief Simple Linear–quadratic regulator problem to control a particle in a 2D space.
 * @details The particle is a ball in the space [-1,1]^2
 * The state is (position_x, position_y, velocity_x, velocity_y). 
 * The action is a force applied to the particle (force_x, force_y). 
 * 
 * 
 * State space:    Low         High
 * position_x       -1.0         1.0
 * position_y       -1.0         1.0
 * velocity_x     -250.0       250.0
 * velocity_y     -250.0       250.0
 * 
 * Action space:   Low        High
 * force_x        -1.5        1.5
 * force_y        -1.5        1.5
 */
class MovingBall: public LQR, public rlly::utils::render::RenderInterface2D<std::vector<double>>
{
private:
    const double ball_mass =  0.1;
    const double delta_t   =  0.01;
public:
    MovingBall(/* args */);
    ~MovingBall() {};

    // reset and step functions
    std::vector<double> reset() override;
    StepResult<std::vector<double>> step(std::vector<double> action) override;

    // clone function
    std::unique_ptr<ContinuousEnv> clone() const override;

    // Get scene representing a given state
    utils::render::Scene2D get_scene_for_render2d(std::vector<double> state_var) override;

    // Returns background for rendering 
    utils::render::Scene2D get_background_for_render2d() override;
};


} // namespace env
} // namespace rlly

#endif
