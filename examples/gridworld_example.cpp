/*
    To run this example:
    $ bash scripts/compile.sh gridworld_example && ./build/examples/gridworld_example
*/

#include <iostream>
#include <vector>
#include <string>
#include "rlly.hpp"

using namespace std;
using namespace rlly;
using namespace rlly::utils::vec;


int main(void)
{
    /*   

        Defining a GridWorld
            
    */
   
    double fail_prob = 0.0;  // failure probability
    double reward_smoothness = 0.0;      // reward = exp( - dist(next_state, goal_state)^2 / reward_smoothness^2)
    double sigma = 0.1;  // reward noise (Gaussian)
    env::GridWorld mdp(5, 5, fail_prob, reward_smoothness, sigma);

    cout << endl << mdp.id << endl;
    cout << endl << mdp.reward_function.noise_type << endl;


    // render 
    mdp.render();
    cout << endl;

    // set mdp seed
    mdp.set_seed(11);

    /*   

        Take some steps
            
    */

    env::StepResult<int> outcome; 
    cout << "Right " << endl;
    outcome = mdp.step(1);
    mdp.render();
    cout << "Reward = " << outcome.reward  << endl << endl; 
   
    cout << "Down " << endl;
    outcome = mdp.step(3);
    mdp.render();
    cout << "Reward = " << outcome.reward << endl << endl; 

    /* 

        Checking transition probabilities

    */
    int state = 0; 
    std::cout << "Transitions at state " << state << ", action left: " << std::endl;
    mdp.render_values(mdp.transitions[state][0]);
    std::cout << "Transitions at state " << state << ", action right: " << std::endl;
    mdp.render_values(mdp.transitions[state][1]);
    std::cout << "Transitions at state " << state << ", action up: " << std::endl;
    mdp.render_values(mdp.transitions[state][2]);
    std::cout << "Transitions at state " << state << ", action down: " << std::endl;
    mdp.render_values(mdp.transitions[state][3]);

    /*

        Graphic rendering

    */
    state = mdp.reset();
    std::vector<int> states;
    int horizon = 50;
    for(int hh = 0; hh < horizon; hh++)
    {
        int action = mdp.action_space.sample();
        auto step_result = mdp.step(action);
        states.push_back(step_result.next_state);
    }
    render::render_env(states, mdp);

    return 0;
}
