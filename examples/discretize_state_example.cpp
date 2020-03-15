/*
    To run this example:
    $ bash scripts/compile.sh discretize_state_example && ./build/examples/discretize_state_example
*/

#include <iostream>
#include "rlly.hpp"


int main()
{
    // create environment, set seed and enable rendering
    rlly::env::MountainCar env;
    env.set_seed(123);
    env.enable_rendering();
    // wrapper that discretizes the state space
    rlly::wrappers::DiscretizeStateWrapper env_discrete(env, 10);
    // run
    int horizon = 50;
    for(int hh = 0; hh < horizon; hh++)
    {
        auto action = env_discrete.action_space.sample();
        auto step_result = env_discrete.step(action);
        std::cout << "state = " << step_result.next_state << std::endl;
    }
    // render wrapped environment
    rlly::render::render_env(env_discrete.p_env);

    // checking
    int Nx = 20;
    int Ny = 20;
    double range_x = env.observation_space.high[0] - env.observation_space.low[0];
    double range_y = env.observation_space.high[1] - env.observation_space.low[1];
    double epsilon_x = range_x/(Nx-1.0);
    double epsilon_y = range_y/(Ny-1.0);
    std::vector<double> states_x;
    std::vector<double> states_y;
    for(int ii=0; ii< Nx; ii++)
    {
        states_x.push_back( env.observation_space.low[0] + epsilon_x*ii );
    }   
    for(int jj=0; jj< Ny; jj++)
    {
        states_y.push_back( env.observation_space.low[1] + epsilon_y*jj );
    }
    rlly::utils::vec::printvec(states_x);
    rlly::utils::vec::printvec(states_y);

    std::cout << "Number of states = " << env_discrete.observation_space.n  << std::endl;
    std::cout << std::endl;
    std::vector<double> foo_state(2);
    for(int jj=0; jj< Ny; jj++)
    {
        for(int ii=0; ii< Nx; ii++)
        {   
            foo_state[0] = states_x[ii];
            foo_state[1] = states_y[jj];
            std::cout << env_discrete.get_state_index(foo_state) << " ";
        }
        std::cout << std::endl;
    }   
    return 0;
}