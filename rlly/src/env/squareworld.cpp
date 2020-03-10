#include "squareworld.h"


namespace rlly
{
namespace env
{

SquareWorld::SquareWorld(/* args */)
{
    id = "SquareWorld";
    state.push_back(start_x);
    state.push_back(start_y);

    // observation and action spaces
    std::vector<double> _low  = {0.0, 0.0};
    std::vector<double> _high = {1.0, 1.0};
    observation_space.set_bounds(_low, _high);
    action_space.set_n(4);

    // set seed
    int _seed = std::rand();
    set_seed(_seed);

    // SquareWorld supports 2d rendering
    refresh_interval_for_render2d = 500;
    clipping_area_for_render2d[0] = 0.0;
    clipping_area_for_render2d[1] = 1.0;
    clipping_area_for_render2d[2] = 0.0;
    clipping_area_for_render2d[3] = 1.0;
}


env::StepResult<std::vector<double>> SquareWorld::step(int action)
{
    // for rendering
    if (rendering_enabled) append_state_for_rendering(state);

    //
    bool done = false; 
    double reward = std::exp( -0.5*(std::pow(state[0]-goal_x, 2) + std::pow(state[1]-goal_y, 2))/(std::pow(reward_smoothness, 2)));
    reward += randgen.sample_gaussian(0, reward_noise_stdev);

    double noise_x = randgen.sample_gaussian(0, transition_noise_stdev);
    double noise_y = randgen.sample_gaussian(0, transition_noise_stdev);

    state[0] = std::min(1.0, std::max(0.0, state[0] + noise_x));
    state[1] = std::min(1.0, std::max(0.0, state[1] + noise_y)); 

    if      (action == 0) state[0] = std::max(0.0, state[0] - displacement);
    else if (action == 1) state[0] = std::min(1.0, state[0] + displacement);
    else if (action == 2) state[1] = std::max(0.0, state[1] - displacement);
    else if (action == 3) state[1] = std::min(1.0, state[1] + displacement);

    env::StepResult<std::vector<double>> result(state, reward, done);
    return result;
}

std::vector<double> SquareWorld::reset()
{
    std::vector<double> initial_state {start_x, start_y};
    state = initial_state;
    return initial_state; 
}


std::unique_ptr<ContinuousStateEnv> SquareWorld::clone() const
{
    return std::make_unique<SquareWorld>(*this);
}

utils::render::Scene2D SquareWorld::get_scene_for_render2d(std::vector<double> state_var)
{
    utils::render::Scene2D agent_scene;
    utils::render::Geometric2D agent;
    agent.type = "GL_QUADS";
    agent.set_color(0.0, 0.0, 0.5);

    float size = 0.025;
    float x = state_var[0];
    float y = state_var[1];
    agent.add_vertex(x-size/4.0, y-size);
    agent.add_vertex(x+size/4.0, y-size);
    agent.add_vertex(x+size/4.0, y+size);
    agent.add_vertex(x-size/4.0, y+size);

    agent.add_vertex(x-size, y-size/4.0);
    agent.add_vertex(x+size, y-size/4.0);
    agent.add_vertex(x+size, y+size/4.0);
    agent.add_vertex(x-size, y+size/4.0);

    agent_scene.add_shape(agent);
    return agent_scene;
}

utils::render::Scene2D SquareWorld::get_background_for_render2d()
{
    utils::render::Scene2D background;
    
    float epsilon = 0.01;
    float x = 0.0;
    while (x < 1.0)
    {
        float y = 0.0;
        while (y < 1.0)
        {
            utils::render::Geometric2D shape;
            shape.type = "GL_QUADS";
            float reward = std::exp( -0.5*(std::pow(x-goal_x, 2) + std::pow(y-goal_y, 2))/(std::pow(reward_smoothness, 2)));

            shape.set_color(0.4, 0.7*reward + 0.3, 0.4);
            shape.add_vertex(x-epsilon, y-epsilon);
            shape.add_vertex(x+epsilon, y-epsilon);
            shape.add_vertex(x+epsilon, y+epsilon);
            shape.add_vertex(x-epsilon, y+epsilon);
            background.add_shape(shape);
            y += epsilon;
        }
        x += epsilon;
    }
    

    return background;
}

}  // namespace env
}  // namespace rlly

