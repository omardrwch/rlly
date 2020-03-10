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
    rlly::wrappers::Wrapper<std::vector<double>, int> env(cartpole);
    rlly::env::Env<std::vector<double>, int>& p_env = env; 

    REQUIRE( env.id.compare("CartPoleWrapper") == 0);

    auto step_result = env.step(env.p_action_space->sample());
    REQUIRE( env.p_observation_space->contains(step_result.next_state) );

    REQUIRE( p_env.id.compare("CartPoleWrapper") == 0);
    REQUIRE( (*p_env.p_observation_space).name == rlly::spaces::box);
    REQUIRE( (*p_env.p_action_space).name == rlly::spaces::discrete);
    REQUIRE( (*p_env.p_action_space).n == 2 );
    REQUIRE( (*p_env.p_observation_space).n == -1 );
    REQUIRE( (*p_env.p_observation_space).contains(env.reset()) ); 
}

TEST_CASE( "Testing Chain Wrapper", "[chain_wrapper]" )
{
    rlly::env::Chain chain(3);
    rlly::wrappers::Wrapper<int, int> env(chain);

    REQUIRE(env.id.compare("ChainWrapper") == 0);  
    REQUIRE(env.p_observation_space->name == rlly::spaces::discrete);
    REQUIRE(env.p_action_space->name == rlly::spaces::discrete);
    REQUIRE( env.get_state() == 0);

    rlly::env::StepResult<int> step_result; 
    
    step_result = env.step(0);
    REQUIRE( (env.get_state() == 1 && step_result.reward == 0) );

    step_result = env.step(1); 
    REQUIRE( (env.get_state() == 0 && step_result.reward == 0) );

    step_result = env.step(0);
    step_result = env.step(0);
    REQUIRE( (env.get_state() == 2 && step_result.reward == 1.0) );
    REQUIRE( env.p_observation_space->contains(step_result.next_state) );
}