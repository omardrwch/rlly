#include <cmath>
#include <vector>

#include "catch.hpp"
#include "env.h"

TEST_CASE( "Testing chain", "[chain]" )
{
    rlly::env::Chain chain(3);
    rlly::env::Env<int, int>& p_env = chain;

    REQUIRE( chain.id.compare("Chain") == 0);  
    REQUIRE( p_env.id.compare("Chain") == 0);  
    REQUIRE( chain.state == 0);

    rlly::env::StepResult<int> step_result; 
    
    step_result = chain.step(0);
    REQUIRE( (chain.state == 1 && step_result.reward == 0) );

    step_result = chain.step(1); 
    REQUIRE( (chain.state == 0 && step_result.reward == 0) );

    step_result = chain.step(0);
    step_result = chain.step(0);
    REQUIRE( (chain.state == 2 && step_result.reward == 1.0) );
}

TEST_CASE( "Testing GridWorld", "[gridworld]" )
{
    double fail_prob = 0.0;          // failure probability
    double reward_smoothness = 0.0;  // reward = exp( - distance(next_state, goal_state)^2 / reward_smoothness^2)
    double sigma = 0.0;              // reward noise (Gaussian)
    rlly::env::GridWorld env(5, 10, fail_prob, reward_smoothness, sigma);
    rlly::env::Env<int, int>& p_env = env;

    REQUIRE( env.id.compare("GridWorld") == 0); 
    REQUIRE( p_env.id.compare("GridWorld") == 0);
    REQUIRE( (*p_env.p_action_space).n == 4 );
    REQUIRE( (*p_env.p_observation_space).n == 50 );  



    REQUIRE( env.state == 0);  
    REQUIRE( (env.index2coord[0][0] == 0 &&  env.index2coord[0][1] == 0) );

    // Left action must do nothing
    env.step(0);
    REQUIRE( (env.index2coord[0][0] == 0 &&  env.index2coord[0][1] == 0) );

    // Right action must increment  column
    env.step(1);
    REQUIRE( (env.index2coord[env.state][0] == 0 &&  env.index2coord[env.state][1] == 1) );

    // Down action must increment row
    env.step(3);
    REQUIRE( (env.index2coord[env.state][0] == 1 &&  env.index2coord[env.state][1] == 1) );

    // Up action must decrement row
    env.step(2);
    REQUIRE( (env.index2coord[env.state][0] == 0 &&  env.index2coord[env.state][1] == 1) );
}


TEST_CASE( "Testing MountainCar", "[mountaincar]" )
{   
    rlly::env::MountainCar env;
    rlly::env::Env<std::vector<double>, int>& p_env = env;

    REQUIRE( env.id.compare("MountainCar") == 0);
    REQUIRE( p_env.id.compare("MountainCar") == 0);
    REQUIRE( (*p_env.p_action_space).n == 3 );
    REQUIRE( (*p_env.p_observation_space).n == -1 );
    REQUIRE( (*p_env.p_observation_space).contains(env.reset()) );  
}

TEST_CASE( "Testing CartPole", "[cartpole]" )
{   
    rlly::env::CartPole env;
    rlly::env::Env<std::vector<double>, int>& p_env = env;

    REQUIRE( env.id.compare("CartPole") == 0);
    REQUIRE( p_env.id.compare("CartPole") == 0);
    REQUIRE( (*p_env.p_action_space).n == 2 );
    REQUIRE( (*p_env.p_observation_space).n == -1 );
    // REQUIRE( (*p_env.p_observation_space).contains(env.reset()) );  
}