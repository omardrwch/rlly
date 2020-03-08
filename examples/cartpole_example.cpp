/*
    To run this example:
    $ bash scripts/compile.sh cartpole_example && ./build/examples/cartpole_example
*/

#include <iostream>
#include "env.h"
#include "utils.h"
#include "render.h"

int main()
{
    rlly::env::CartPole env;

    std::vector<double> state = env.reset();
    rlly::utils::vec::vec_2d states;

    int horizon = 200;
    for(int ii = 0; ii < horizon; ii++)
    {
        auto action = env.action_space.sample();
        auto step_result = env.step(action);
        states.push_back(step_result.next_state);
        std::cout << "state = "; rlly::utils::vec::printvec(step_result.next_state);
        std::cout << "reward = " << step_result.reward << std::endl;
        if (step_result.done) break;
    }

    rlly::render::render_env(states, env);

    return 0;
}