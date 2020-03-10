#include <cmath>
#include <vector>
#include <iostream>
#include <memory>

#include "catch.hpp"
#include "env.h"
#include "utils.h"
#include "wrappers.h"

TEST_CASE( "Testing CartPole Wrapper", "[cartpole_wrapper]" )
{   
    rlly::env::CartPole cartpole;
    rlly::wrappers::ContinuousStateEnvWrapper env(cartpole);
    rlly::env::ContinuousStateEnv& p_env = env; 

    REQUIRE( env.id.compare("CartPoleIsomorphicWrapper") == 0);

    auto step_result = env.step(env.action_space.sample());
    REQUIRE( env.observation_space.contains(step_result.next_state) );

    REQUIRE( p_env.id.compare("CartPoleIsomorphicWrapper") == 0);
    REQUIRE( p_env.observation_space.name == rlly::spaces::box);
    REQUIRE( p_env.action_space.name == rlly::spaces::discrete);
    REQUIRE( p_env.action_space.n == 2 );
    REQUIRE( p_env.observation_space.n == -1 );
    REQUIRE( p_env.observation_space.contains(env.reset()) ); 
}

TEST_CASE( "Testing Chain Wrapper", "[chain_wrapper]" )
{
    rlly::env::Chain chain(3);
    rlly::wrappers::FiniteEnvWrapper env(chain);

    REQUIRE(env.id.compare("ChainIsomorphicWrapper") == 0);  
    REQUIRE(env.observation_space.name == rlly::spaces::discrete);
    REQUIRE(env.action_space.name == rlly::spaces::discrete);
    REQUIRE( env.reset() == 0);

    rlly::env::StepResult<int> step_result; 
    
    step_result = env.step(0);
    REQUIRE( (step_result.next_state == 1 && step_result.reward == 0) );

    step_result = env.step(1); 
    REQUIRE( (step_result.next_state == 0 && step_result.reward == 0) );

    step_result = env.step(0);
    step_result = env.step(0);
    REQUIRE( (step_result.next_state == 2 && step_result.reward == 1.0) );
    REQUIRE( env.observation_space.contains(step_result.next_state) );
}


// TEST_CASE( "Testing DiscretizeStateWrapper", "[discretize_state_wrapper]" )
// {
//     rlly::env::MountainCar mountaincar;
//     int n_bins = 10;
//     rlly::wrappers::DiscretizeStateWrapper discrete_env(mountaincar, n_bins);
// }