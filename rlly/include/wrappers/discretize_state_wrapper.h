// #ifndef __RLLY_DISCRETIZE_STATE_WRAPPER_H___
// #define __RLLY_DISCRETIZE_STATE_WRAPPER_H___

// #include <vector>
// #include <iostream>
// #include "basic_wrapper.h"
// #include "utils.h"
// #include "space.h"
// #include "env.h"

// namespace rlly
// {
// namespace wrappers
// {


// class DiscretizeStateWrapper: public ContinuousStateEnvWrapper
// {
// public:
//     /**
//      * @param env
//      * @param n_bins number of intervals in the discretization of each dimension of the state space
//      */
//     DiscretizeStateWrapper(env::ContinuousStateEnv& env, int n_bins);
//     /**
//      * @param env
//      * @param vec_n_bins vec_n_bins[i] is the number of intervals in the discretization of the i-th each dimension of the state space
//      */
//     DiscretizeStateWrapper(env::ContinuousStateEnv& env, std::vector<int> vec_n_bins);
//     ~DiscretizeStateWrapper(){};

//     /**
//      * all_bins[i] = {x_1, ..., x_n } the points corresponding to the discretization of the i-th dimension of the state space
//      */  
//     utils::vec::vec_2d all_bins;
// };

// } // namespace wrappers
// } // namespace rlly


// #endif