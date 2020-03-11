#ifndef __RLLY_CHAIN_H__
#define __RLLY_CHAIN_H__

/**
 * @file
 * @brief Define a N-Chain MDP.
 */

#include <vector>
#include "finitemdp.h"

namespace rlly
{
namespace env
{
    
/**
 * @brief N-Chain environment. States = {0, ..., N-1}, Actions = {0, 1}.
 * @details In state I, when action 0 is taken, the next state is min(I+1, N-1).
 *                      when action 1 is taken, the next state is max(I-1, 0).
 *          A reward of 1 is obtained when the next state is N-1.
 * @param N length of the chain.
 */
class Chain: public FiniteMDP
{
public:
    /**
     * @brief Build chain MDP of length N
     */
    Chain(int N=3, double fail_p=0);
    ~Chain(){};

    /**
     * Clone
     */
    std::unique_ptr<FiniteEnv> clone() const override;
};

} // namespace env
} // namespace rlly


#endif
