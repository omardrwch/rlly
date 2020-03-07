/*
    To run this example:
    $ bash scripts/compile.sh mountaincar_example && ./build/examples/mountaincar_example
*/

#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include "env.h"
#include "space.h"
#include "utils.h"

// #include "rlly.hpp"

#include "render.h"


int main()
{
    rlly::env::MountainCar env;
    int horizon = 200;
    
    std::vector<std::vector<double>> states;
    env.set_seed(123);
    for(int hh = 1; hh < horizon; hh++)
    {
        auto action = env.action_space.sample();
        auto step_result = env.step(action);
        // std::cout << "action  " << action << ", angle = " << step_result.next_state[0] <<std::endl;
        states.push_back(step_result.next_state);
    }

    rlly::render::render_env(states, env);
    return 0;
}