#include <assert.h>
#include <cmath>
#include <algorithm>
#include "mountaincar.h"
#include "utils.h"


namespace rlly
{
namespace env
{

MountainCar::MountainCar()
{
    // Set seed
    int _seed = std::rand();
    set_seed(_seed);

    // observation and action spaces
    std::vector<double> _low = {-1.2, -0.07};
    std::vector<double> _high = {0.6, 0.07};
    observation_space.set_bounds(_low, _high);
    action_space.set_n(3);

    goal_position = 0.5;
    goal_velocity = 0;

    state.push_back(0);
    state.push_back(0);

    id = "MountainCar";

    // 2D rendering is enabled for MountainCar
    clipping_area_for_render2d[0] = -1.2;
    clipping_area_for_render2d[1] =  0.6;
    clipping_area_for_render2d[2] = -0.2;
    clipping_area_for_render2d[3] =  1.1;
}

std::vector<double> MountainCar::reset()
{
    state[position] = randgen.sample_real_uniform(observation_space.low[position], observation_space.high[position]);
    state[velocity] = 0;
    return state;
}

StepResult<std::vector<double>> MountainCar::step(int action)
{
    assert(action_space.contains(action));

    // for rendering
    if (rendering_enabled) append_state_for_rendering(state);
    //
    
    std::vector<double>& lo = observation_space.low;
    std::vector<double>& hi = observation_space.high;


    double p = state[position];
    double v = state[velocity];

    v += (action-1)*force + std::cos(3*p)*(-gravity);
    v = utils::clamp(v, lo[velocity], hi[velocity]);
    p += v;
    p = utils::clamp(p, lo[position], hi[position]);
    if ((std::abs(p-lo[position])<1e-10) && (v<0))
    { 
        v = 0;
    }
    bool done = is_terminal(state);
    double reward = 0.0;
    if (done) reward = 1.0;

    state[position] = p;
    state[velocity] = v;

    StepResult<std::vector<double>> step_result(state, reward, done);
    return step_result;
}

bool MountainCar::is_terminal(std::vector<double> state)
{
    return ((state[position] >= goal_position) && (state[velocity]>=goal_velocity));
}


std::unique_ptr<ContinuousStateEnv> MountainCar::clone() const
{
    return std::make_unique<MountainCar>(*this);
}



utils::render::Scene2D MountainCar::get_scene_for_render2d(std::vector<double> state_var)
{
    float y = std::sin(3*state_var[position])*0.45 + 0.55;
    float x = state_var[position];


    utils::render::Scene2D car_scene;
    utils::render::Geometric2D car;
    car.type = "GL_POLYGON";
    car.set_color(0.0f, 0.0f, 0.0f);
    
    float size = 0.025;
    car.add_vertex(x - size, y - size);
    car.add_vertex(x + size, y - size);
    car.add_vertex(x + size, y + size);
    car.add_vertex(x - size, y + size);

    car_scene.add_shape(car);
    return car_scene;
}

utils::render::Scene2D MountainCar::get_background_for_render2d()
{
    utils::render::Scene2D background;
    utils::render::Geometric2D mountain;
    utils::render::Geometric2D flag;
    mountain.type = "GL_TRIANGLE_FAN";
    mountain.set_color(0.6, 0.3, 0.0);
    flag.type     = "GL_TRIANGLES";
    flag.set_color(0.0, 0.5, 0.0);


    std::vector<std::vector<float>> vertices1 = {{-1.0, -1.0}};

    // Mountain
    mountain.add_vertex( -0.3f, -1.0f);
    mountain.add_vertex(  0.6f, -1.0f);

    int n_points = 100;
    double range = observation_space.high[0] - observation_space.low[0];
    double eps = range/(n_points-1.0);
    for(int ii = n_points-1; ii >= 0; ii--)
    {
        double x = observation_space.low[0] + ii*eps;
        double y = std::sin(3*x)*0.45 + 0.55;
        mountain.add_vertex(x, y);
    }
    mountain.add_vertex(-1.2f, -1.0f);

    // Flag
    float goal_x = goal_position;
    float goal_y = std::sin(3*goal_position)*0.45 + 0.55;
    flag.add_vertex(goal_x, goal_y);
    flag.add_vertex(goal_x+0.025f, goal_y+0.075f);
    flag.add_vertex(goal_x-0.025f, goal_y+0.075f);

    background.add_shape(mountain);
    background.add_shape(flag);
    return background;
}


}  // namespace env
}  // namespace rlly
