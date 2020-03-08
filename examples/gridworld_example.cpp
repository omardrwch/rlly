/*
    To run this example:
    $ bash scripts/compile.sh gridworld_example && ./build/examples/gridworld_example
*/

#include <iostream>
#include <vector>
#include "rlly.hpp"

int main(void)
{
    double fail_prob = 0.0;          // failure probability
    double reward_smoothness = 0.0;  // reward = exp( - distance(next_state, goal_state)^2 / reward_smoothness^2)
    double sigma = 0.1;              // reward noise (Gaussian)
    rlly::env::GridWorld env(5, 5, fail_prob, reward_smoothness, sigma);
    env.set_seed(123);

     // Rendering (graphic)
    int state = env.reset();
    std::vector<int> states;
    int horizon = 50;
    for(int hh = 0; hh < horizon; hh++)
    {
        int action = env.action_space.sample();
        auto step_result = env.step(action);
        states.push_back(step_result.next_state);
    }
    rlly::render::render_env(states, env);

    // Rendering (text)
    env.render();
    return 0;
}
