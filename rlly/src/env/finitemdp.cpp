#include <algorithm>
#include <iostream>
#include <string>
#include <cmath>
#include "finitemdp.h"

namespace rlly
{
namespace env
{

FiniteMDP::FiniteMDP(DiscreteReward _reward_function, utils::vec::vec_3d _transitions, int _default_state /* = 0 */, int _seed /* = -1 */)
{
    set_params(_reward_function, _transitions, _default_state, _seed);
}

FiniteMDP::FiniteMDP(DiscreteReward _reward_function, utils::vec::vec_3d _transitions, std::vector<int> _terminal_states, int _default_state /* = 0 */, int _seed /* = -1 */)
{
    set_params(_reward_function, _transitions, _terminal_states, _default_state, _seed);
}

void FiniteMDP::set_params(DiscreteReward _reward_function, utils::vec::vec_3d _transitions, int _default_state /* = 0 */, int _seed /* = -1 */)
{
    reward_function = _reward_function;
    transitions = _transitions;
    id = "FiniteMDP";
    default_state = _default_state;

    check();
    ns = reward_function.mean_rewards.size();
    na = reward_function.mean_rewards[0].size();

    // observation and action spaces
    observation_space.set_n(ns);
    action_space.set_n(na);

    // initialize pointers of base class
    p_observation_space = &observation_space;
    p_action_space      = &action_space;
    set_seed(_seed);
    
    reset();
}

void FiniteMDP::set_params(DiscreteReward _reward_function, utils::vec::vec_3d _transitions, std::vector<int> _terminal_states, int _default_state /* = 0 */, int _seed /* = -1 */)
{
    set_params(_reward_function, _transitions, _default_state, _seed);
    terminal_states = _terminal_states;
}

void FiniteMDP::check()
{
    // Check shape of transitions and rewards
    assert(reward_function.mean_rewards.size() > 0);
    assert(reward_function.mean_rewards[0].size() > 0);
    assert(reward_function.mean_rewards[0][0].size() > 0);
    assert(transitions.size() > 0);
    assert(transitions[0].size() > 0);
    assert(transitions[0][0].size() > 0);

    // Check consistency of number of states
    assert(reward_function.mean_rewards[0][0].size() == reward_function.mean_rewards.size());
    assert(transitions[0][0].size() == transitions.size());
    assert(transitions.size() == reward_function.mean_rewards.size());

    // Check consistency of number of actions
    assert(reward_function.mean_rewards[0].size() == transitions[0].size());

    // Check transition probabilities
    for(int i = 0; i < transitions.size(); i++)
    {
        for(int a = 0; a < transitions[0].size(); a++)
        {
            double sum = 0;
            for(int j = 0; j < transitions[0][0].size(); j++)
            {
                assert(transitions[i][a][j] >= 0.0);
                sum += transitions[i][a][j];
            }
            // std::cout << std::abs(sum - 1.0) << std::endl;
            assert(std::abs(sum - 1.0) <= 1e-12 && "Probabilities must sum to 1");
        }
    }
}

int FiniteMDP::reset()
{
    state = default_state;
    return default_state;
}

bool FiniteMDP::is_terminal(int _state)
{
    return (std::find(terminal_states.begin(), terminal_states.end(), _state) != terminal_states.end());
}

/**
 *  @note done is true if next_state is terminal.
 */
StepResult<int> FiniteMDP::step(int action)
{
    // Sample next state
    int next_state = randgen.choice(transitions[state][action]);
    double reward = reward_function.sample(state, action, next_state, randgen); 
    bool done = is_terminal(next_state);
    StepResult<int> step_result(next_state, reward, done);
    state = step_result.next_state;
    return step_result;
}

}  // namespace env
}  // namespace rlly
