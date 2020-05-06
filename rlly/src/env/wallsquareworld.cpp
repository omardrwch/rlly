#include "wallsquareworld.h"


namespace rlly
{
namespace env
{

WallSquareWorld::WallSquareWorld(/* args */)
{
    id = "WallSquareWorld";
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


env::StepResult<std::vector<double>> WallSquareWorld::step(int action)
{
    // for rendering
    if (rendering_enabled) append_state_for_rendering(state);

    //
    bool done = false; 
    double reward = std::exp( -0.5*(std::pow(state[0]-goal_x, 2) + std::pow(state[1]-goal_y, 2))/(std::pow(reward_smoothness, 2)));
    reward += randgen.sample_gaussian(0, reward_noise_stdev);

    double noise_x = randgen.sample_gaussian(0, transition_noise_stdev);
    double noise_y = randgen.sample_gaussian(0, transition_noise_stdev);

    double prev_x = state[0];
    double prev_y = state[1];
    state[0] = std::min(1.0, std::max(0.0, state[0] + noise_x));
    state[1] = std::min(1.0, std::max(0.0, state[1] + noise_y)); 

    if      (action == 0) state[0] = std::max(0.0, state[0] - displacement);
    else if (action == 1) state[0] = std::min(1.0, state[0] + displacement);
    else if (action == 2) state[1] = std::max(0.0, state[1] - displacement);
    else if (action == 3) state[1] = std::min(1.0, state[1] + displacement);

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
    state[0] = xx;
    state[1] = yy;

    env::StepResult<std::vector<double>> result(state, reward, done);
    return result;
}

void WallSquareWorld::clip_to_domain(double &xx, double &yy)
{
    xx = std::max(0.0, xx);
    xx = std::min(1.0, xx);
    yy = std::max(0.0, yy);
    yy = std::min(1.0, yy);
}

bool WallSquareWorld::is_inside_wall(double xx, double yy)
{
    clip_to_domain(xx, yy);
    bool flag = false;
    flag = flag ||  ( xx <= wall_1_x1 && xx >= wall_1_x0 && yy <= wall_1_y1 && yy >= wall_1_y0);
    flag = flag ||  ( xx <= wall_2_x1 && xx >= wall_2_x0 && yy <= wall_2_y1 && yy >= wall_2_y0);
    return flag;
}

std::vector<double> WallSquareWorld::reset()
{
    std::vector<double> initial_state {start_x, start_y};
    state = initial_state;
    return initial_state; 
}


std::unique_ptr<ContinuousStateEnv> WallSquareWorld::clone() const
{
    return std::make_unique<WallSquareWorld>(*this);
}

utils::render::Scene2D WallSquareWorld::get_scene_for_render2d(std::vector<double> state_var)
{
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
    return agent_scene;
}

utils::render::Scene2D WallSquareWorld::get_background_for_render2d()
{
    utils::render::Scene2D background;
    

    // Flag
    utils::render::Geometric2D flag;
    flag.set_color(0.0, 0.5, 0.0);
    flag.type     = "GL_TRIANGLES";
    flag.add_vertex(goal_x, goal_y);
    flag.add_vertex(goal_x+0.025f, goal_y+0.075f);
    flag.add_vertex(goal_x-0.025f, goal_y+0.075f);
    background.add_shape(flag);

    // // reward visualization
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

    //         // shape.set_color(0.1, 0.9*reward + 0.1, 0.1);
    //         shape.set_color(0.7*(1.0-reward), 0.7, 0.7*(1.0-reward));
    //         shape.add_vertex(x-epsilon, y-epsilon);
    //         shape.add_vertex(x+epsilon, y-epsilon);
    //         shape.add_vertex(x+epsilon, y+epsilon);
    //         shape.add_vertex(x-epsilon, y+epsilon);
    //         background.add_shape(shape);
    //         y += epsilon;
    //     }
    //     x += epsilon;
    // }

    // wall visualization
    utils::render::Geometric2D wall_1;
    wall_1.type = "GL_QUADS";
    wall_1.set_color(0.0, 0.0, 0.0);
    wall_1.add_vertex(wall_1_x0, wall_1_y0);
    wall_1.add_vertex(wall_1_x0, wall_1_y1);
    wall_1.add_vertex(wall_1_x1, wall_1_y1);
    wall_1.add_vertex(wall_1_x1, wall_1_y0);    
    background.add_shape(wall_1);

    utils::render::Geometric2D wall_2;
    wall_2.type = "GL_QUADS";
    wall_2.set_color(0.0, 0.0, 0.0);
    wall_2.add_vertex(wall_2_x0, wall_2_y0);
    wall_2.add_vertex(wall_2_x0, wall_2_y1);
    wall_2.add_vertex(wall_2_x1, wall_2_y1);
    wall_2.add_vertex(wall_2_x1, wall_2_y0);    
    background.add_shape(wall_2);

    return background;
}

}  // namespace env
}  // namespace rlly

