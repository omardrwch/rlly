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
    // Initialize pointers in the base class
    p_action_space = &action_space;
    p_observation_space = &observation_space;

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

    // Graph rendering is enabled for MountainCar
    graph_rendering_n_nodes = 1;
    graph_rendering_enabled = true;
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


std::vector<std::vector<float>> MountainCar::get_nodes_for_graph_render(std::vector<double> state_var)
{
    std::vector<std::vector<float>> nodes = {{0.0, 0.0}};
    float y = std::sin(3*state_var[position])*0.45 + 0.55;
    float x = state_var[position];
    nodes[0][0] = (10.0/9.0)*x + 1.0/3.0 ;
    nodes[0][1] = y;

    // std::cout << "nodes_xy = " << x << ", " << y << std::endl;

    return nodes;
}



}  // namespace env
}  // namespace rlly
