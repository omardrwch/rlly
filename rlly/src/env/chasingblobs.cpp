#include "chasingblobs.h"

namespace rlly
{
namespace env
{

ChasingBlobs::ChasingBlobs(int _period): period(_period)
{
    id = "ChasingBlobs";
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

    // ChasingBlobs supports 2d rendering
    refresh_interval_for_render2d = 500;
    clipping_area_for_render2d[0] = 0.0;
    clipping_area_for_render2d[1] = 1.0;
    clipping_area_for_render2d[2] = 0.0;
    clipping_area_for_render2d[3] = 1.0;
}


env::StepResult<std::vector<double>> ChasingBlobs::step(int action)
{

    // for rendering
    if (rendering_enabled)
    { 
        std::vector<double> state_with_env_info = state;
        for(int ii = 0; ii < 4; ii ++)
            state_with_env_info.push_back(reward_multipliers[current_configuration][ii]);
        append_state_for_rendering(state_with_env_info);
    }

    // done flag
    bool done = false; 
    // compute reward
    double reward =  0; 
    for(int ii = 0; ii < 4; ii ++)
    {
        double blob_ii_x = blob_x_vec[ii];
        double blob_ii_y = blob_y_vec[ii];
        double c_ii      = reward_multipliers[current_configuration][ii];
        reward          += c_ii * std::exp( -0.5*(std::pow(state[0]-blob_ii_x, 2) + std::pow(state[1]-blob_ii_y, 2))/(std::pow(reward_smoothness, 2)));
    }

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

std::vector<double> ChasingBlobs::reset()
{
     // increase episode counter
    current_episode += 1;
    // change environment according to period
    if (current_episode > 0 && current_episode % period == 0)
    {
        current_configuration = (current_configuration + 1) % n_configurations;
    }

    std::vector<double> initial_state {start_x, start_y};
    state = initial_state;
    return initial_state; 
}


std::unique_ptr<ContinuousStateEnv> ChasingBlobs::clone() const
{
    return std::make_unique<ChasingBlobs>(*this);
}

utils::render::Scene2D ChasingBlobs::get_scene_for_render2d(std::vector<double> state_var)
{
    utils::render::Scene2D agent_scene;

    // Draw agent
    utils::render::Geometric2D agent;
    agent.type = "GL_QUADS";
    agent.set_color(0.75, 0.0, 0.5);

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

    // Draw rewards (blobs)
    float radius = reward_smoothness;
    int n_points = 20;
    for(int ii = 0; ii < 4; ii ++)
    {
        double blob_ii_x = blob_x_vec[ii];
        double blob_ii_y = blob_y_vec[ii];
        double c_ii      = state_var[2+ii];       

        if (c_ii == 0) continue;

        utils::render::Geometric2D blob;
        blob.type = "GL_POLYGON";
        blob.set_color(0.0f, (float) c_ii , 0.0f);
        for(int ii = 0; ii < n_points; ii++)
        {
            float angle = 2.0*3.141592*ii/n_points;
            float xcirc = blob_ii_x + radius*std::cos(angle);
            float ycirc = blob_ii_y + radius*std::sin(angle);
            blob.add_vertex(xcirc, ycirc);    
        }
        agent_scene.add_shape(blob);
    }

    return agent_scene;
}

utils::render::Scene2D ChasingBlobs::get_background_for_render2d()
{
    utils::render::Scene2D background;
    
    // float epsilon = 0.01;
    // float x = 0.0;
    // while (x < 1.0)
    // {
    //     float y = 0.0;
    //     while (y < 1.0)
    //     {
    //         utils::render::Geometric2D shape;
    //         shape.type = "GL_QUADS";
    //         float reward = std::exp( -0.5*(std::pow(x-goal_x, 2) + std::pow(y-goal_y, 2))/(std::pow(reward_smoothness, 2)));

    //         shape.set_color(0.1, 0.9*reward + 0.1, 0.1);
    //         shape.add_vertex(x-epsilon, y-epsilon);
    //         shape.add_vertex(x+epsilon, y-epsilon);
    //         shape.add_vertex(x+epsilon, y+epsilon);
    //         shape.add_vertex(x-epsilon, y+epsilon);
    //         background.add_shape(shape);
    //         y += epsilon;
    //     }
    //     x += epsilon;
    // }
    

    return background;
}

}  // namespace env
}  // namespace rlly

