#include <vector>
#include "catch.hpp"
#include "random.h"

TEST_CASE( "Testing Random::choice", "[choice]" )
{
    rlly::utils::rand::Random randgen(42); 
    std::vector<double> prob = {0.1, 0.2, 0.3, 0.4};
    std::vector<double> prob_backup = {0.1, 0.2, 0.3, 0.4};

    std::vector<double> prob2 = {1.0, 0.0, 0.0, 0.0};
    std::vector<double> prob3 = {0.0, 1.0, 0.0, 0.0};
    std::vector<double> prob4 = {0.0, 0.0, 1.0, 0.0};
    std::vector<double> prob5 = {0.0, 0.0, 0.0, 1.0};
    std::vector<double> prob6 = {1.0};

    REQUIRE( randgen.choice(prob, 0)    == 0 );
    REQUIRE( randgen.choice(prob, 0.05) == 0 );
    REQUIRE( randgen.choice(prob, 0.10001) == 1 );
    REQUIRE( randgen.choice(prob, 0.15) == 1 );
    REQUIRE( randgen.choice(prob, 0.30001) == 2 );
    REQUIRE( randgen.choice(prob, 0.59999) == 2 );
    REQUIRE( randgen.choice(prob, 0.60001) == 3 );
    REQUIRE( randgen.choice(prob, 0.8) == 3 );
    REQUIRE( randgen.choice(prob, 1.0) == 3 );

    REQUIRE( randgen.choice(prob2) == 0 );
    REQUIRE( randgen.choice(prob3) == 1 );
    REQUIRE( randgen.choice(prob4) == 2 );
    REQUIRE( randgen.choice(prob5) == 3 );
    REQUIRE( randgen.choice(prob6) == 0 );

    // Verify that calls to choice do not change the vector prob
    bool prob_unchanged = true;
    for(int i = 0; i < prob.size(); i++)
    {
        prob_unchanged = prob_unchanged && (prob[i] == prob_backup[i]);
    }
    REQUIRE( prob_unchanged );
}