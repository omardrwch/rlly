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

    env::Chain mdp(20);
    cout << mdp.id << endl << endl;
    int max_t = 15;

    for(int i = 0; i < max_t; i++)
    {
        int state = mdp.state;
        int action = mdp.action_space.sample();
        std::vector<double> extra_vars = {0.001, 0.002}; // values of extra variables

        // take step
        auto step_result = mdp.step(action);
        if (step_result.done) break;

        std::cout << "state = " << state << ", action = " << action << ", reward = " << step_result.reward 
                  << ", next state = " <<   step_result.next_state << std::endl;
    }
    
    /* 
    
    
        A continuous MDP: mountain car
    
    
     */
    env::MountainCar mountain_car;
    std::cout << mountain_car.id << std::endl << std::endl;

    std::vector<double> cstate = mountain_car.reset();
    for(int i = 0; i < max_t; i++)
    {
        cstate = mountain_car.state;
        int action = mountain_car.action_space.sample();
        auto step_result = mountain_car.step(action);
        std::cout << "state = ";
        utils::vec::printvec(cstate); 
    }
    return 0;
}

