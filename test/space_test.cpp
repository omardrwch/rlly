#include "catch.hpp"
#include "space.h"
// #include "discrete.h"
// #include "box.h"
#include <vector>


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