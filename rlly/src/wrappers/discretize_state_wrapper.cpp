#include "discretize_state_wrapper.h"

namespace rlly
{
namespace wrappers
{



DiscretizeStateWrapper::DiscretizeStateWrapper(env::ContinuousStateEnv& env, int n_bins):  Wrapper<std::vector<double>, int>(env)
{
    int dim = env.observation_space.low.size();
    for(int dd = 0; dd < dim; dd++)
    {
        double range = env.observation_space.high[dd] - env.observation_space.low[dd];
        double epsilon = range/n_bins;
        // discretization of dimension dd
        std::vector<double> discretization(n_bins+1);
        for (int bin = 0; bin < n_bins+1; bin++)
        {
            discretization[bin] = env.observation_space.low[dd] + epsilon*bin;
        } 
        all_bins.push_back(discretization);
    }
}

DiscretizeStateWrapper::DiscretizeStateWrapper(env::ContinuousStateEnv& env, std::vector<int> vec_n_bins):  Wrapper<std::vector<double>, int>(env)
{
    int dim = env.observation_space.low.size();
    if (vec_n_bins.size() != dim)
    {
        std::cerr << "Incompatible dimensions in the constructor of DiscretizeStateWrapper." << std::endl;
        throw;
    }
    for(int dd = 0; dd < dim; dd++)
    {
        int n_bins = vec_n_bins[dd];
        double range = env.observation_space.high[dd] - env.observation_space.low[dd];
        double epsilon = range/n_bins;
        // discretization of dimension dd
        std::vector<double> discretization(n_bins+1);
        for (int bin = 0; bin < n_bins+1; bin++)
        {
            discretization[bin] = env.observation_space.low[dd] + epsilon*bin;
        } 
        all_bins.push_back(discretization);
    }
}

} // namespace wrappers
} // namespace rlly
