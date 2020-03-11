#ifndef __RLLY_DISCRETIZE_STATE_WRAPPER_H___
#define __RLLY_DISCRETIZE_STATE_WRAPPER_H___

#include <vector>
#include <iostream>
#include "basewrapper.h"
#include "utils.h"
#include "space.h"
#include "env.h"

namespace rlly
{
namespace wrappers
{

template <typename EnvType>
class DiscretizeStateWrapper: public BaseWrapper<EnvType, spaces::Discrete, spaces::Discrete>
{
private:
    void init();
    double tol = 1e-9;
public:
    /**
     * @param env
     * @param n_bins number of intervals in the discretization of each dimension of the state space
     */
    DiscretizeStateWrapper(EnvType& env, int n_bins);
    /**
     * @param env
     * @param vec_n_bins vec_n_bins[i] is the number of intervals in the discretization of the i-th each dimension of the state space
     */
    DiscretizeStateWrapper(EnvType& env, std::vector<int> vec_n_bins);
    ~DiscretizeStateWrapper(){};

    // reset
    int reset() override; 

    // step
    env::StepResult<int> step(int action) override;

    /**
     * all_bins[i] = {x_1, ..., x_n } the points corresponding to the discretization of the i-th dimension of the state space
     */  
    utils::vec::vec_2d all_bins;

    /**
     *  Get discrete representation of a continuous state
     */
    int get_state_index(std::vector<double> state);

};

template <typename EnvType>
void DiscretizeStateWrapper<EnvType>::init()
{
    if(this->p_env.observation_space.name != spaces::box)
    {
        std::cerr << "Error: DiscretizeStateWrapper requires an environment with an observation space of type spaces::Box ." << std::endl;
        throw;
    }
    if(this->p_env.action_space.name != spaces::discrete)
    {
        std::cerr << "Error: DiscretizeStateWrapper requires an environment with an action space of type spaces::Discrete." << std::endl;
        throw;
    }
    this->action_space = this->p_env.action_space;
}

template <typename EnvType>
DiscretizeStateWrapper<EnvType>::DiscretizeStateWrapper(EnvType& env, int n_bins): 
    BaseWrapper<EnvType, spaces::Discrete, spaces::Discrete>(env)
{
    init();

    int num_states = 1;
    unsigned int dim = env.observation_space.low.size();
    for(unsigned int dd = 0; dd < dim; dd++)
    {
        double range = env.observation_space.high[dd] - env.observation_space.low[dd];
        double epsilon = range/n_bins;
        // discretization of dimension dd
        std::vector<double> discretization(n_bins+1);
        for (int bin = 0; bin < n_bins+1; bin++)
        {
            discretization[bin] = env.observation_space.low[dd] + epsilon*bin;
            if (bin == n_bins) discretization[bin] += tol; // "close" the last interval
        } 
        all_bins.push_back(discretization);
        num_states = num_states*n_bins;
    }
    this->observation_space.set_n(num_states);
}

template <typename EnvType>
DiscretizeStateWrapper<EnvType>::DiscretizeStateWrapper(EnvType& env, std::vector<int> vec_n_bins): 
    BaseWrapper<EnvType, spaces::Discrete, spaces::Discrete>(env)
{
    init();

    int num_states = 1;
    unsigned int dim = env.observation_space.low.size();
    if (vec_n_bins.size() != dim)
    {
        std::cerr << "Incompatible dimensions in the constructor of DiscretizeStateWrapper." << std::endl;
        throw;
    }
    for(unsigned int dd = 0; dd < dim; dd++)
    {
        int n_bins = vec_n_bins[dd];
        double range = env.observation_space.high[dd] - env.observation_space.low[dd];
        double epsilon = range/n_bins;
        // discretization of dimension dd
        std::vector<double> discretization(n_bins+1);
        for (int bin = 0; bin < n_bins+1; bin++)
        {
            discretization[bin] = env.observation_space.low[dd] + epsilon*bin;
            if (bin == n_bins) discretization[bin] += tol; // "close" the last interval
        } 
        all_bins.push_back(discretization);
        num_states = num_states*n_bins;
    }
    this->observation_space.set_n(num_states);
}


template <typename EnvType>
int DiscretizeStateWrapper<EnvType>::get_state_index(std::vector<double> state)
{
    unsigned int dim = this->p_env.observation_space.low.size();
    int state_index = 0;
    int aux = 1;
    if (dim != state.size()) throw;
    for(unsigned int dd = 0; dd < dim; dd++)
    {
        int index_dd = utils::binary_search(state[dd], all_bins[dd]);
        if (index_dd == -1) throw;
        state_index += aux*index_dd;
        aux *= (all_bins[dd].size()-1);
    }
    return state_index;
}


template <typename EnvType>
env::StepResult<int> DiscretizeStateWrapper<EnvType>::step(int action)
{
    env::StepResult<int> result;
    env::StepResult<std::vector<double>> wrapped_result = this->p_env.step(action);
    result.next_state = get_state_index(wrapped_result.next_state);
    result.reward     = wrapped_result.reward;
    result.done       = wrapped_result.done;
    return result;
}   

template <typename EnvType>
int DiscretizeStateWrapper<EnvType>::reset()
{
    std::vector<double> wrapped_state = this->p_env.reset();
    return get_state_index(wrapped_state);
}   


} // namespace wrappers
} // namespace rlly


#endif
