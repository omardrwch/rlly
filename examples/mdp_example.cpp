/*
    To run this example:
    $ bash scripts/compile.sh mdp_example && ./build/examples/mdp_example
*/

#include <iostream>
#include <vector>
#include <string>
#include "rlly.hpp"

using namespace std;
using namespace rlly;

int main(void)
{
    /*   

            Defining a simple MDP with 3 states and 2 actions  

            
    */

    mdp::Chain mdp(20);
    cout << mdp.id << endl << endl;

    int max_t = 15;

    // Setting up the history, with two extra variables stored.
    mdp.history.reserve_mem(max_t, 2);
    std::vector<std::string> names = {"extra_var_1", "extra_var_2"};
    mdp.history.set_names(names);

    for(int i = 0; i < max_t; i++)
    {
        int state = mdp.state;
        int action = mdp.action_space.sample();
        std::vector<double> extra_vars = {0.001, 0.002}; // values of extra variables

        // take step
        mdp::StepResult<int> step_result = mdp.step(action);

        // append to history
        mdp.history.append(state, action, step_result.reward, step_result.next_state, extra_vars);

        if (step_result.done) break;
    }


    // If we clear the history, nothing is printed, of course :) 
    // mdp.history.clear();
    // std::cout << "states size: " << mdp.history.states.size() << std::endl;
    // std::cout << "actions size: " << mdp.history.actions.size() << std::endl;
    // std::cout << "rewards size: " << mdp.history.rewards.size() << std::endl;
    // std::cout << "next states size: " << mdp.history.next_states.size() << std::endl;

    // print history
    mdp.history.print(max_t);

    // save history in csv file
    mdp.history.to_csv("data/temp.csv");

    
    /* 
    
    
        A continuous MDP: mountain car
    
    
     */

    mdp::MountainCar env;
    std::cout << env.id << std::endl;

    env.history.reserve_mem(max_t, 0);

    std::vector<double> cstate = env.reset();
    for(int i = 0; i < max_t; i++)
    {
        cstate = env.state;
        int action = env.action_space.sample();
        mdp::StepResult<std::vector<double>> step_result = env.step(action);
        env.history.append(cstate, action, step_result.reward, step_result.next_state);
    }


    // print history
    env.history.print(max_t);

    return 0;
}

