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
    rlly::env::Env<std::vector<double>, int>& p_env = env; // This is a problem: this should be possible

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
