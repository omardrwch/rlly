#ifndef __RLLY_CARTPOLE_H__
#define __RLLY_CARTPOLE_H__

#include <vector>
#include <assert.h>
#include <cmath>
#include <limits>
#include <iostream>
#include "abstractenv.h"
#include "utils.h"


namespace rlly
{
namespace env
{

/**
 * CartPole environment, as in https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
 */
class CartPole: public Env<std::vector<double>, int>
{
private:
    
    const float pi = std::atan(1.0)*4.0;
    const float gravity = 9.8;
    const float mass_cart = 1.0;
    const float mass_pole = 0.1;
    const float total_mass = mass_cart + mass_pole;
    const float half_pole_length = 0.5;
    const float pole_mass_times_length = half_pole_length*mass_pole;
    const float force_magnitude = 10.0;
    const float delta_t = 0.02;

    // angle threshold
    const float theta_threshold_radians = 12.0*pi/180.0;
    // position threshold
    const float x_threshold = 2.4;

    int steps_beyond_done = -1;

public:
    CartPole();
    ~CartPole(){};

    std::vector<double> reset();
    StepResult<std::vector<double>> step(int action) override;


    /**
    * State (observation) space
    */
    spaces::Box observation_space;

    /**
    *  Action space
    */
    spaces::Discrete action_space;

    /**
     *  Clone the object
     */
    std::unique_ptr<Env<std::vector<double>, int>> clone() const override;

    /**
     * Get scene representing a given state
     */
    utils::render::Scene get_scene_for_render2d(std::vector<double> state_var) override;

    /**
     * Returns background for rendering 
     */
    utils::render::Scene get_background_for_render2d() override;
};


} // namespace env
} // namespace rlly

#endif
