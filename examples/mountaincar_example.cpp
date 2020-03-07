/*
    To run this example:
    $ bash scripts/compile.sh mountaincar_example && ./build/examples/mountaincar_example
*/

#include <vector>
// the header needs to be generated with -rendering option
#include "rlly.hpp"


int main()
{
    rlly::env::MountainCar env;
    env.set_seed(123);

    int horizon = 200;
    rlly::utils::vec::vec_2d states; // or std::vector<std::vector<double>> states;
    for(int hh = 0; hh < horizon; hh++)
    {
        auto action = env.action_space.sample();
        auto step_result = env.step(action);
        states.push_back(step_result.next_state);
    }

    rlly::render::render_env(states, env);
    return 0;
}