#ifndef __RLLY_DISCRETIZE_STATE_WRAPPER_H___
#define __RLLY_DISCRETIZE_STATE_WRAPPER_H___

#include <vector>
#include <iostream>
#include "basewrapper.h"
#include "utils.h"
#include "space.h"

namespace rlly
{
namespace wrappers
{


class DiscretizeStateWrapper: public Wrapper<std::vector<double>, int>
{
private:
    void check();
public:
    /**
     * @param env
     * @param n_bins number of intervals in the discretization of each dimension of the state space
     */
    DiscretizeStateWrapper(env::Env<std::vector<double>, int>& env, int n_bins);
    /**
     * @param env
     * @param vec_n_bins vec_n_bins[i] is the number of intervals in the discretization of the i-th each dimension of the state space
     */
    DiscretizeStateWrapper(env::Env<std::vector<double>, int>& env, std::vector<int> vec_n_bins);
    ~DiscretizeStateWrapper(){};

    /**
     * all_bins[i] = {x_1, ..., x_n } the points corresponding to the discretization of the i-th dimension of the state space
     */  
    utils::vec::vec_2d all_bins;
    
};

DiscretizeStateWrapper::DiscretizeStateWrapper(env::Env<std::vector<double>, int>& env, int n_bins):  Wrapper<std::vector<double>, int>(env)
{
    check();
}

DiscretizeStateWrapper::DiscretizeStateWrapper(env::Env<std::vector<double>, int>& env, std::vector<int> vec_n_bins):  Wrapper<std::vector<double>, int>(env)
{
    check();
}

void DiscretizeStateWrapper::check()
{
    if ( (*p_env).p_observation_space->name != spaces::box)
    {
        std::cerr << "DiscretizeStateWrapper requires an environment whose observation space is spaces::box" << std::endl;
        throw;
    }
    if ( (*p_env).p_action_space->name != spaces::discrete)
    {
        std::cerr << "DiscretizeStateWrapper requires an environment whose action space is spaces::discrete" << std::endl;
        throw;
    }
}



} // namespace wrappers
} // namespace rlly


#endif