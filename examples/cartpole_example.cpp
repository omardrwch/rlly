/*
    To run this example:
    $ bash scripts/compile.sh cartpole_example && ./build/examples/cartpole_example
*/

#include <iostream>
#include "env.h"
#include "render.h"

int main()
{
    // create environment and enable rendering
    rlly::env::CartPole env;
    env.enable_rendering();
    // run
    int horizon = 200;
    for(int ii = 0; ii < horizon; ii++)
    {
        auto action = env.action_space.sample();
        auto step_result = env.step(action);
        std::cout << "state = "; rlly::utils::vec::printvec(step_result.next_state);
        std::cout << "reward = " << step_result.reward << std::endl;
        if (step_result.done) break;
    }
    // render
    rlly::render::render_env(env);
    return 0;
}