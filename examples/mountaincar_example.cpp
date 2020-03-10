/*
    To run this example:
    $ bash scripts/compile.sh mountaincar_example && ./build/examples/mountaincar_example
*/

#include <vector>
// the header needs to be generated with -rendering option
#include "rlly.hpp"


int main()
{
    // create environment, set seed and enable rendering
    rlly::env::MountainCar env;
    env.set_seed(123);
    env.enable_rendering();
    // run
    int horizon = 200;
    for(int hh = 0; hh < horizon; hh++)
    {
        auto action = env.action_space.sample();
        auto step_result = env.step(action);
    }
    // render
    rlly::render::render_env(env);
    return 0;
}