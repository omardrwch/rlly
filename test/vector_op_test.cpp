#include <vector>
#include <cmath>
#include "catch.hpp"
#include "vector_op.h"

TEST_CASE( "Testing vector_op mean and standard dev", "[mean_stdev]" )
{
    std::vector<double> vec1 = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> vec2 = {0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<double> vec3 = {0.0, 0.0, 0.0, 0.0, 1.0};
    std::vector<double> vec4 = {2.0};

    REQUIRE( std::fabs(rlly::utils::vec::mean(vec1) - 3.0) < 1.0e-16);
    REQUIRE( std::fabs(rlly::utils::vec::stdev(vec1) - std::sqrt(2.0)) < 1.0e-16);

    REQUIRE( rlly::utils::vec::mean(vec2) == 0.0);
    REQUIRE( rlly::utils::vec::stdev(vec2) == 0.0);

    REQUIRE( rlly::utils::vec::mean(vec3) == 1.0/5.0);
    REQUIRE( rlly::utils::vec::stdev(vec3) == 0.4);

    REQUIRE( rlly::utils::vec::mean(vec4) == 2.0);
    REQUIRE( rlly::utils::vec::stdev(vec4) == 0.0);
}

TEST_CASE( "Testing inner product", "[inner_prod]")
{
    std::vector<double> vec1 = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> vec2 = {0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<double> vec3 = {0.0, 0.0, 0.0, 0.0, 1.0};  
    std::vector<double> vec4 = {2.0};
    std::vector<double> vec5 = {0.0, 0.0, 2.0, 0.0, 1.0};  


    REQUIRE( rlly::utils::vec::inner_prod(vec2, vec2) == 0.0);
    REQUIRE( rlly::utils::vec::inner_prod(vec3, vec3) == 1.0);
    REQUIRE( rlly::utils::vec::inner_prod(vec4, vec4) == 4.0);
    REQUIRE( rlly::utils::vec::inner_prod(vec1, vec2) == 0.0);
    REQUIRE( rlly::utils::vec::inner_prod(vec1, vec3) == 5.0);
    REQUIRE( rlly::utils::vec::inner_prod(vec1, vec5) == 11.0);
    REQUIRE( rlly::utils::vec::inner_prod(vec3, vec5) == 1.0);
}