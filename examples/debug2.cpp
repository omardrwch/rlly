/*

    bash scripts/compile.sh debug2 && ./build/examples/debug2

*/

#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include "env.h"
#include "utils.h"
#include "space.h"

using namespace std;
using namespace rlly;

int main(void)
{
    rlly::env::MountainCar env;
    int horizon = 20;
    
    std::vector<std::vector<double>> states;
    env.set_seed(789);
    for(int hh = 1; hh < horizon; hh++)
    {
        auto action = env.action_space.sample(); 
        auto step_result = env.step(action);
        std::cout << "action  " << action << ", angle = " << step_result.next_state[0] << ", speed = " << step_result.next_state[1]  <<std::endl;
        states.push_back(step_result.next_state);
    }
}
