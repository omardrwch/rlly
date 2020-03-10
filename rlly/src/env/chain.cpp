#include <assert.h>
#include "chain.h"
#include "utils.h"


namespace rlly
{
namespace env
{

Chain::Chain(int N, double fail_p)
{
    assert(N > 0 && "Chain needs at least one state");
    utils::vec::vec_3d _rewards = utils::vec::get_zeros_3d(N, 2, N);
    utils::vec::vec_3d _transitions = utils::vec::get_zeros_3d(N, 2, N);
    std::vector<int> _terminal_states = {N-1};

    for(int state = 0; state < N; state++)
    {
        for(int action = 0; action < 2; action++)
        {
            int next_state = -1;
            // First action: go to the right
            if (action == 0)
            {
                next_state = std::min(state + 1, N-1);
            }
            // Second action: go to the left
            else if (action == 1)
            {
                next_state = std::max(state - 1, 0);
            }
            _transitions[state][action][next_state] = 1.0 - fail_p;
            _transitions[state][action][state] += fail_p;
            if (next_state == N-1)
            {
                _rewards[state][action][next_state] = 1.0;
            }
        }
    }

    set_params(_rewards, _transitions, _terminal_states);
    id = "Chain";
}

std::unique_ptr<FiniteEnv> Chain::clone() const
{
    return std::make_unique<Chain>(*this);
}


}  // namespace env
}  // namespace rlly
