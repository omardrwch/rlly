#include <vector>
#include <cmath>
#include <string>
#include "catch.hpp"
#include "utils.h"

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
    for(unsigned int i = 0; i < prob.size(); i++)
    {
        prob_unchanged = prob_unchanged && (prob[i] == prob_backup[i]);
    }
    REQUIRE( prob_unchanged );
}


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


TEST_CASE( "Testing binary search", "[binary_search]")
{
    std::vector<double> vec0 = {};
    std::vector<double> vec1 = {1.0};
    std::vector<double> vec2 = {1.0, 2.0};
    std::vector<double> vec3 = {1.0, 2.0, 3.0};
    std::vector<double> vec4 = {1.0, 2.0, 3.0, 4.0};

    std::vector<double> vec5 = {1.0, 4.0, 4.0, 4.0};
    std::vector<double> vec6 = {1.0, 1.0, 1.0, 4.0};
    std::vector<double> vec7 = {1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0};


    double val0 = 0.0;
    double val1 = 0.5;
    double val2 = 1.0;
    double val3 = 1.5;
    double val4 = 2.0;
    double val5 = 2.5;
    double val6 = 3.0;
    double val7 = 3.5;
    double val8 = 4.5;


    REQUIRE(rlly::utils::binary_search(val0, vec0) == -1);

    REQUIRE(rlly::utils::binary_search(val0, vec1) == -1);
    REQUIRE(rlly::utils::binary_search(val1, vec1) == -1);

    REQUIRE(rlly::utils::binary_search(val0, vec2) == -1);
    REQUIRE(rlly::utils::binary_search(val1, vec2) == -1);
    REQUIRE(rlly::utils::binary_search(val2, vec2) ==  0);
    REQUIRE(rlly::utils::binary_search(val3, vec2) ==  0);
    REQUIRE(rlly::utils::binary_search(val4, vec2) == -1);
    REQUIRE(rlly::utils::binary_search(val5, vec2) == -1);

    REQUIRE(rlly::utils::binary_search(val0, vec3) == -1);
    REQUIRE(rlly::utils::binary_search(val1, vec3) == -1);
    REQUIRE(rlly::utils::binary_search(val2, vec3) ==  0);
    REQUIRE(rlly::utils::binary_search(val3, vec3) ==  0);
    REQUIRE(rlly::utils::binary_search(val4, vec3) ==  1);
    REQUIRE(rlly::utils::binary_search(val5, vec3) ==  1);
    REQUIRE(rlly::utils::binary_search(val6, vec3) == -1);
    REQUIRE(rlly::utils::binary_search(val7, vec3) == -1);

    REQUIRE(rlly::utils::binary_search(val0, vec4) == -1);
    REQUIRE(rlly::utils::binary_search(val1, vec4) == -1);
    REQUIRE(rlly::utils::binary_search(val2, vec4) ==  0);
    REQUIRE(rlly::utils::binary_search(val3, vec4) ==  0);
    REQUIRE(rlly::utils::binary_search(val4, vec4) ==  1);
    REQUIRE(rlly::utils::binary_search(val5, vec4) ==  1);
    REQUIRE(rlly::utils::binary_search(val6, vec4) ==  2);
    REQUIRE(rlly::utils::binary_search(val7, vec4) ==  2);
    REQUIRE(rlly::utils::binary_search(val8, vec4) == -1);

    REQUIRE(rlly::utils::binary_search(val2, vec5) ==  0);
    REQUIRE(rlly::utils::binary_search(val7, vec5) ==  0);
    REQUIRE(rlly::utils::binary_search(val8, vec5) == -1);

    REQUIRE(rlly::utils::binary_search(val2, vec6) ==  2);
    REQUIRE(rlly::utils::binary_search(val7, vec6) ==  2);
    REQUIRE(rlly::utils::binary_search(val8, vec6) == -1);

    REQUIRE(rlly::utils::binary_search(val0, vec7) == -1);
    REQUIRE(rlly::utils::binary_search(val1, vec7) == -1);
    REQUIRE(rlly::utils::binary_search(val2, vec7) ==  2);
    REQUIRE(rlly::utils::binary_search(val3, vec7) ==  2);
    REQUIRE(rlly::utils::binary_search(val8, vec7) ==  5);
}

TEST_CASE( "Testing d-dimensional binary search", "[d_dim_binary_search]")
{
    std::vector<std::vector<double>> all_bins;

    std::vector<double> bin1 = {0.0, 1.0, 2.0, 3.0}; // 3 intervals
    std::vector<double> bin2 = {1.0, 2.0, 3.0, 4.0}; // 3 intervals
    std::vector<double> bin3 = {2.0, 3.0, 4.0, 5.0, 6.0}; // 4 intervals

    all_bins.push_back(bin1);
    all_bins.push_back(bin2);
    all_bins.push_back(bin3);
    
    std::vector<double> vec1 = {0.0, 1.0, 2.0};
    std::vector<double> vec2 = {2.9, 3.9, 5.9};

    REQUIRE( rlly::utils::binary_search_nd (vec1, all_bins) == 0 );
    REQUIRE( rlly::utils::binary_search_nd (vec2, all_bins) == 3*3*4-1);
}


TEST_CASE( "Testing params class", "[params_class]")
{
    rlly::utils::params::Params params;

    REQUIRE( params.is_defined("string1") ==  false);
    params.append("string1", std::string("hello_world"));
    REQUIRE( params.string_params["string1"] ==  "hello_world");
    REQUIRE( params.is_defined("string1") ==  true);
    REQUIRE( params.is_defined("string2") ==  false);
    REQUIRE( params.is_defined("string1", "string") ==  true);
    REQUIRE( params.is_defined("string1", "double") ==  false);
    REQUIRE( params.is_defined("string1", "int")    ==  false);

    REQUIRE( params.is_defined("double1") ==  false);
    params.append("double1", 0.005);
    REQUIRE( params.double_params["double1"] == 0.005);
    REQUIRE( params.is_defined("double1") ==  true);
    REQUIRE( params.is_defined("double2") ==  false);
    REQUIRE( params.is_defined("double1", "string") ==  false);
    REQUIRE( params.is_defined("double1", "double") ==  true);
    REQUIRE( params.is_defined("double1", "int")    ==  false);

    REQUIRE( params.is_defined("int1") ==  false);
    params.append("int1", 10);
    REQUIRE( params.int_params["int1"] == 10);
    REQUIRE( params.is_defined("int1") ==  true);
    REQUIRE( params.is_defined("int2") ==  false);
    REQUIRE( params.is_defined("int1", "string") ==  false);
    REQUIRE( params.is_defined("int1", "double") ==  false);
    REQUIRE( params.is_defined("int1", "int")    ==  true);
}

