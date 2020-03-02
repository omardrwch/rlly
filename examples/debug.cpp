/*

    There is nothing important in this file. I use this just to test a few things :)

*/

#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include "rlly.hpp"

using namespace std;
using namespace rlly;

int main(void)
{
    // auto space = spaces::Discrete(5);
    // std::cout << space.name << std::endl;
    // std::cout << space.n << std::endl;

    // spaces::Space<int>& pointer = space; 
    // std::cout << pointer.name << std::endl;

    env::Chain chain(5);
    env::Env<int, int>& mdp = chain;

    std::cout << "Number of states and actions: " << std::endl;
    std::cout << chain.observation_space.n << std::endl;
    std::cout << chain.action_space.n << std::endl;

    std::cout << "This should be the same: " << std::endl; 
    std::cout << (*mdp.p_observation_space).n << std::endl;
    std::cout << (*mdp.p_action_space).n << std::endl;
}
