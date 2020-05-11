#include "changingwallsquareworld.h"


namespace rlly
{
namespace env
{

ChangingWallSquareWorld::ChangingWallSquareWorld(int _period): period(_period)
{
    id = "ChangingWallSquareWorld";
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


env::StepResult<std::vector<double>> ChangingWallSquareWorld::step(int action)
{
    // get current env parameters
    double goal_x  = goal_x_vec[current_configuration];
    double goal_y  = goal_y_vec[current_configuration];

    if (rendering_enabled)
    { 
        // state for rendering -> the renderer needs to know the wall and reward positions!
        double wall_x0 = wall_x0_vec[current_configuration];
        double wall_x1 = wall_x1_vec[current_configuration]; 
        double wall_y0 = wall_y0_vec[current_configuration]; 
        double wall_y1 = wall_y1_vec[current_configuration]; 

        std::vector<double> state_with_env_info = state;
        state_with_env_info.push_back(goal_x);
        state_with_env_info.push_back(goal_y);
        state_with_env_info.push_back(wall_x0);
        state_with_env_info.push_back(wall_x1);
        state_with_env_info.push_back(wall_y0);
        state_with_env_info.push_back(wall_y1);
        append_state_for_rendering(state_with_env_info);
    }

    //
    bool done = false; 
    double reward = std::exp( -0.5*(std::pow(state[0]-goal_x, 2) + std::pow(state[1]-goal_y, 2))/(std::pow(reward_smoothness, 2)));
    reward += randgen.sample_gaussian(0, reward_noise_stdev);

    double noise_x = randgen.sample_gaussian(0, transition_noise_stdev);
    double noise_y = randgen.sample_gaussian(0, transition_noise_stdev);

    double prev_x = state[0];
    double prev_y = state[1];

    // compute action dispacement
    double action_x, action_y;
    if      (action == 0)
    {
        action_x = action_0_displacement_vec[current_configuration][0];
        action_y = action_0_displacement_vec[current_configuration][1];
    }
    else if (action == 1)
    {
        action_x = action_1_displacement_vec[current_configuration][0];
        action_y = action_1_displacement_vec[current_configuration][1];
    }
    else if (action == 2)
    {
        action_x = action_2_displacement_vec[current_configuration][0];
        action_y = action_2_displacement_vec[current_configuration][1];
    }
    else if (action == 3)
    {
        action_x = action_3_displacement_vec[current_configuration][0];
        action_y = action_3_displacement_vec[current_configuration][1];        
    }
    state[0] = state[0] + action_x + noise_x;
    state[1] = state[1] + action_y + noise_y;
    clip_to_domain(state[0], state[1]);

    // Check walls
    double delta_x = state[0] - prev_x;
    double delta_y = state[1] - prev_y;
    double xx = prev_x; 
    double yy = prev_y;
    int N = 10;
    double eps_x = delta_x / N;
    double eps_y = delta_y / N;
    for(int ii = 1; ii <= N; ii++)
    {
        if( ! is_inside_wall( xx + eps_x, yy + eps_y) )
        {
            xx = xx + eps_x;
            yy = yy + eps_y;
        }
    }
    clip_to_domain(xx, yy);
    state[0] = xx;
    state[1] = yy;

    env::StepResult<std::vector<double>> result(state, reward, done);
    return result;
}

void ChangingWallSquareWorld::clip_to_domain(double &xx, double &yy)
{
    xx = std::max(0.0, xx);
    xx = std::min(1.0, xx);
    yy = std::max(0.0, yy);
    yy = std::min(1.0, yy);
}

bool ChangingWallSquareWorld::is_inside_wall(double xx, double yy)
{
    double wall_x0 = wall_x0_vec[current_configuration];
    double wall_x1 = wall_x1_vec[current_configuration]; 
    double wall_y0 = wall_y0_vec[current_configuration]; 
    double wall_y1 = wall_y1_vec[current_configuration]; 
    clip_to_domain(xx, yy);
    bool flag = false;
    flag = flag ||  ( xx <= wall_x1 && xx >= wall_x0 && yy <= wall_y1 && yy >= wall_y0);
    return flag;
}

std::vector<double> ChangingWallSquareWorld::reset()
{
    // increase episode counter
    current_episode += 1;
    // change wall position according to period
    if (current_episode % period == 0)
    {
        // current_configuration = (current_configuration + 1) % n_configurations;
        current_configuration = randgen.sample_int_uniform(0, n_configurations-1);
    }
    // set initial state
    std::vector<double> initial_state {start_x, start_y};
    state = initial_state;
    return initial_state; 
}


std::unique_ptr<ContinuousStateEnv> ChangingWallSquareWorld::clone() const
{
    return std::make_unique<ChangingWallSquareWorld>(*this);
}

utils::render::Scene2D ChangingWallSquareWorld::get_scene_for_render2d(std::vector<double> state_var)
{
    double goal_x  = state_var[2];
    double goal_y  = state_var[3];
    double wall_x0 = state_var[4];
    double wall_x1 = state_var[5];
    double wall_y0 = state_var[6];
    double wall_y1 = state_var[7];

    utils::render::Scene2D agent_scene;
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

    // wall visualization
    utils::render::Geometric2D wall_1;
    wall_1.type = "GL_QUADS";
    wall_1.set_color(0.0, 0.0, 0.0);
    wall_1.add_vertex(wall_x0, wall_y0);
    wall_1.add_vertex(wall_x0, wall_y1);
    wall_1.add_vertex(wall_x1, wall_y1);
    wall_1.add_vertex(wall_x1, wall_y0);    
    agent_scene.add_shape(wall_1);


    // Flag
    utils::render::Geometric2D flag;
    flag.set_color(0.0, 0.5, 0.0);
    flag.type     = "GL_TRIANGLES";
    flag.add_vertex(goal_x, goal_y);
    flag.add_vertex(goal_x+0.025f, goal_y+0.075f);
    flag.add_vertex(goal_x-0.025f, goal_y+0.075f);
    agent_scene.add_shape(flag);

    return agent_scene;
}

utils::render::Scene2D ChangingWallSquareWorld::get_background_for_render2d()
{
    utils::render::Scene2D background;
    
    return background;
}

}  // namespace env
}  // namespace rlly

