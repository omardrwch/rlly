#include <cmath>
#include <vector>
#include <iostream>
#include <memory>

#include "catch.hpp"
#include "env.h"
#include "space.h"
#include "utils.h"

TEST_CASE( "Testing discrete spaces", "[discrete_space]" )
{
    rlly::spaces::Discrete space(5);
    REQUIRE( space.name == rlly::spaces::discrete);
    REQUIRE( space.n == 5 );
    REQUIRE( space.contains(0) );
    REQUIRE( space.contains(4) );
    REQUIRE( !space.contains(5) );

    bool ok = true;
    for(int i = 0; i<15; i++)
    {
        ok = ok && space.contains(space.sample());
    }
    REQUIRE( ok );
}

TEST_CASE( "Testing box spaces", "[box_space]")
{
    std::vector<double> low = {-1, -2, -3, -4, -5};
    std::vector<double> high = {1, 10, 20, 30, 100};

    std::vector<double> valid = {0, 5, 17, 20, 99};
    std::vector<double> not_valid = {-1, 5, 30, 20, 101};

    rlly::spaces::Box space(low, high);

    REQUIRE( space.name == rlly::spaces::box);
    REQUIRE( space.contains(valid) );
    REQUIRE( !space.contains(not_valid) );

    bool space_contains_sample = true;
    for(int i = 0; i<25; i++)
    {
        space_contains_sample = space_contains_sample && space.contains(space.sample());
    }
    
    REQUIRE( space_contains_sample );
}


TEST_CASE( "Testing chain", "[chain]" )
{
    rlly::env::Chain chain(3);
    rlly::env::FiniteEnv& p_env = chain;

    REQUIRE( chain.id.compare("Chain") == 0);  
    REQUIRE( p_env.id.compare("Chain") == 0);  
    REQUIRE(chain.observation_space.name == rlly::spaces::discrete);
    REQUIRE(chain.action_space.name == rlly::spaces::discrete);
    REQUIRE( chain.reset() == 0);

    rlly::env::StepResult<int> step_result; 
    
    step_result = chain.step(0);
    REQUIRE( (step_result.next_state == 1 && step_result.reward == 0) );

    step_result = chain.step(1); 
    REQUIRE( (step_result.next_state == 0 && step_result.reward == 0) );

    step_result = chain.step(0);
    step_result = chain.step(0);
    REQUIRE( (step_result.next_state == 2 && step_result.reward == 1.0) );

    REQUIRE( chain.observation_space.contains(step_result.next_state) );
}

TEST_CASE( "Testing GridWorld", "[gridworld]" )
{
    double fail_prob = 0.0;          // failure probability
    double reward_smoothness = 0.0;  // reward = exp( - distance(next_state, goal_state)^2 / reward_smoothness^2)
    double sigma = 0.0;              // reward noise (Gaussian)
    rlly::env::GridWorld env(5, 10, fail_prob, reward_smoothness, sigma);
    rlly::env::FiniteEnv& p_env = env;

    REQUIRE( env.id.compare("GridWorld") == 0); 
    REQUIRE( p_env.id.compare("GridWorld") == 0);
    REQUIRE(env.observation_space.name == rlly::spaces::discrete);
    REQUIRE(env.action_space.name == rlly::spaces::discrete);
    REQUIRE( p_env.action_space.n == 4 );
    REQUIRE( p_env.observation_space.n == 50 );  



    REQUIRE( env.reset() == 0);  
    REQUIRE( (env.index2coord[0][0] == 0 &&  env.index2coord[0][1] == 0) );

    // Left action must do nothing
    env.step(0);
    REQUIRE( (env.index2coord[0][0] == 0 &&  env.index2coord[0][1] == 0) );

    // Right action must increment  column
    auto step_result = env.step(1);
    REQUIRE( (env.index2coord[step_result.next_state][0] == 0 &&  env.index2coord[step_result.next_state][1] == 1) );

    // Down action must increment row
    step_result = env.step(3);
    REQUIRE( (env.index2coord[step_result.next_state][0] == 1 &&  env.index2coord[step_result.next_state][1] == 1) );

    // Up action must decrement row
    step_result = env.step(2);
    REQUIRE( (env.index2coord[step_result.next_state][0] == 0 &&  env.index2coord[step_result.next_state][1] == 1) );

    step_result = env.step(env.action_space.sample());
    REQUIRE( env.observation_space.contains(step_result.next_state) );
}


TEST_CASE( "Testing MountainCar", "[mountaincar]" )
{   
    rlly::env::MountainCar env;
    rlly::env::ContinuousStateEnv& p_env = env;
    auto unique_p_env = env.clone();

    REQUIRE( env.id.compare("MountainCar") == 0);
    REQUIRE(env.observation_space.name == rlly::spaces::box);
    REQUIRE(env.action_space.name == rlly::spaces::discrete);

    REQUIRE( p_env.id.compare("MountainCar") == 0);
    REQUIRE( p_env.observation_space.name == rlly::spaces::box);
    REQUIRE( p_env.action_space.name == rlly::spaces::discrete);
    REQUIRE( p_env.action_space.n == 3 );
    REQUIRE( p_env.observation_space.n == -1 );
    REQUIRE( p_env.observation_space.contains(env.reset()) );  

    REQUIRE( (*unique_p_env).id.compare("MountainCar") == 0);
    REQUIRE( (*unique_p_env).observation_space.name == rlly::spaces::box);
    REQUIRE( (*unique_p_env).action_space.name == rlly::spaces::discrete);
    REQUIRE( (*unique_p_env).action_space.n == 3 );
    REQUIRE( (*unique_p_env).observation_space.n == -1 );
    REQUIRE( (*unique_p_env).observation_space.contains(env.reset()) );

    auto step_result = env.step(env.action_space.sample());
    REQUIRE( env.observation_space.contains(step_result.next_state) );
}

TEST_CASE( "Testing CartPole", "[cartpole]" )
{   
    rlly::env::CartPole env;
    rlly::env::ContinuousStateEnv& p_env = env;
    auto unique_p_env = env.clone();

    REQUIRE( env.id.compare("CartPole") == 0);
    REQUIRE(env.observation_space.name == rlly::spaces::box);
    REQUIRE(env.action_space.name == rlly::spaces::discrete);

    auto step_result = env.step(env.action_space.sample());
    REQUIRE( env.observation_space.contains(step_result.next_state) );

    REQUIRE( p_env.id.compare("CartPole") == 0);
    REQUIRE( p_env.observation_space.name == rlly::spaces::box);
    REQUIRE( p_env.action_space.name == rlly::spaces::discrete);
    REQUIRE( p_env.action_space.n == 2 );
    REQUIRE( p_env.observation_space.n == -1 );
    REQUIRE( p_env.observation_space.contains(env.reset()) ); 

    REQUIRE( (*unique_p_env).id.compare("CartPole") == 0);
    REQUIRE( (*unique_p_env).observation_space.name == rlly::spaces::box);
    REQUIRE( (*unique_p_env).action_space.name == rlly::spaces::discrete);
    REQUIRE( (*unique_p_env).action_space.n == 2 );
    REQUIRE( (*unique_p_env).observation_space.n == -1 );
    REQUIRE( (*unique_p_env).observation_space.contains(env.reset()) ); 
}

TEST_CASE( "Testing SquareWorld", "[squareworld]" )
{   
    rlly::env::SquareWorld env;
    rlly::env::ContinuousStateEnv& p_env = env;
    auto unique_p_env = env.clone();

    REQUIRE( env.id.compare("SquareWorld") == 0);
    REQUIRE(env.observation_space.name == rlly::spaces::box);
    REQUIRE(env.action_space.name == rlly::spaces::discrete);

    auto step_result = env.step(env.action_space.sample());
    REQUIRE( env.observation_space.contains(step_result.next_state) );


    REQUIRE( p_env.id.compare("SquareWorld") == 0);
    REQUIRE( p_env.observation_space.name == rlly::spaces::box);
    REQUIRE( p_env.action_space.name == rlly::spaces::discrete);
    REQUIRE( p_env.action_space.n == 4 );
    REQUIRE( p_env.observation_space.n == -1 );
    REQUIRE( p_env.observation_space.contains(env.reset()) ); 

    REQUIRE( (*unique_p_env).id.compare("SquareWorld") == 0);
    REQUIRE( (*unique_p_env).observation_space.name == rlly::spaces::box);
    REQUIRE( (*unique_p_env).action_space.name == rlly::spaces::discrete);
    REQUIRE( (*unique_p_env).action_space.n == 4 );
    REQUIRE( (*unique_p_env).observation_space.n == -1 );
    REQUIRE( (*unique_p_env).observation_space.contains(env.reset()) ); 
}


TEST_CASE( "Testing WallSquareWorld", "[wallsquareworld]" )
{   
    rlly::env::WallSquareWorld env;
    rlly::env::ContinuousStateEnv& p_env = env;
    auto unique_p_env = env.clone();

    REQUIRE( env.id.compare("WallSquareWorld") == 0);
    REQUIRE(env.observation_space.name == rlly::spaces::box);
    REQUIRE(env.action_space.name == rlly::spaces::discrete);

    auto step_result = env.step(env.action_space.sample());
    REQUIRE( env.observation_space.contains(step_result.next_state) );


    REQUIRE( p_env.id.compare("WallSquareWorld") == 0);
    REQUIRE( p_env.observation_space.name == rlly::spaces::box);
    REQUIRE( p_env.action_space.name == rlly::spaces::discrete);
    REQUIRE( p_env.action_space.n == 4 );
    REQUIRE( p_env.observation_space.n == -1 );
    REQUIRE( p_env.observation_space.contains(env.reset()) ); 

    REQUIRE( (*unique_p_env).id.compare("WallSquareWorld") == 0);
    REQUIRE( (*unique_p_env).observation_space.name == rlly::spaces::box);
    REQUIRE( (*unique_p_env).action_space.name == rlly::spaces::discrete);
    REQUIRE( (*unique_p_env).action_space.n == 4 );
    REQUIRE( (*unique_p_env).observation_space.n == -1 );
    REQUIRE( (*unique_p_env).observation_space.contains(env.reset()) ); 
}

TEST_CASE( "Testing MovingBall", "[movingball]" )
{   
    rlly::env::MovingBall env;
    rlly::env::ContinuousEnv& p_env = env;
    auto unique_p_env = env.clone();

    REQUIRE( env.id.compare("MovingBall") == 0);
    REQUIRE(env.observation_space.name == rlly::spaces::box);
    REQUIRE(env.action_space.name == rlly::spaces::box);

    REQUIRE( p_env.id.compare("MovingBall") == 0);
    REQUIRE( p_env.observation_space.name == rlly::spaces::box);
    REQUIRE( p_env.action_space.name == rlly::spaces::box);
    REQUIRE( p_env.action_space.n == -1 );
    REQUIRE( p_env.observation_space.n == -1 );
    REQUIRE( p_env.observation_space.contains(env.reset()) );  

    REQUIRE( (*unique_p_env).id.compare("MovingBall") == 0);
    REQUIRE( (*unique_p_env).observation_space.name == rlly::spaces::box);
    REQUIRE( (*unique_p_env).action_space.name == rlly::spaces::box);
    REQUIRE( (*unique_p_env).action_space.n == -1 );
    REQUIRE( (*unique_p_env).observation_space.n == -1 );
    REQUIRE( (*unique_p_env).observation_space.contains(env.reset()) );

    auto step_result = env.step(env.action_space.sample());
    REQUIRE( env.observation_space.contains(step_result.next_state) );
}


TEST_CASE( "Testing Factory", "[factory]" )
{   
    rlly::env::ContinuousStateEnv* p_env_1;
    rlly::env::ContinuousStateEnv* p_env_2;
    rlly::env::ContinuousStateEnv* p_env_3;

    p_env_1 = rlly::env::make_continuous_state_env("SquareWorld");
    p_env_2 = rlly::env::make_continuous_state_env("MountainCar");
    p_env_3 = rlly::env::make_continuous_state_env("CartPole");

    REQUIRE( p_env_1->id.compare("SquareWorld") == 0);
    REQUIRE( p_env_2->id.compare("MountainCar") == 0);
    REQUIRE( p_env_3->id.compare("CartPole")    == 0);

    delete p_env_1;
    delete p_env_2;
    delete p_env_3;
}


template <class EnvType>
void test_continuous_state_env_seeding()
{
    EnvType env;
    std::vector<uint> seeds = {1, 4294967295/8, 4294967295/4, 4294967295/2, 4294967295}; 
    for(uint ii = 0; ii < seeds.size(); ii++)
    {
        env.set_seed(seeds[ii]);
        REQUIRE(env.randgen.get_seed() == seeds[ii]);
    }

    EnvType env1;
    EnvType env2;
    env1.set_seed(4294967295);
    env2.set_seed(4294967295);

    int dim = env1.observation_space.low.size();
    for(int ii = 0; ii < 10; ii++)
    {
        auto action = env1.action_space.sample();
        auto result1 = env1.step(action);
        auto result2 = env2.step(action);
        for(int dd = 0; dd < dim; dd++)
        {
            REQUIRE( result1.next_state[dd] ==  result2.next_state[dd]);
        }
    }
}

TEST_CASE( "SquareWorld seeding test", "[SquareWorld_seeding]" )
{
    test_continuous_state_env_seeding<rlly::env::SquareWorld>();
}

TEST_CASE( "MountainCar seeding test", "[MountainCar_seeding]" )
{
    test_continuous_state_env_seeding<rlly::env::MountainCar>();
}

TEST_CASE( "CartPole seeding test", "[CartPole_seeding]" )
{
    test_continuous_state_env_seeding<rlly::env::CartPole>();
}

TEST_CASE( "MovingBall seeding test", "[MovingBall_seeding]" )
{
    test_continuous_state_env_seeding<rlly::env::MovingBall>();
}


