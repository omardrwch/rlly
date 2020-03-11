#ifndef __RLLY_BASIC_WRAPPER_H__
#define __RLLY_BASIC_WRAPPER_H__

#include <memory>
#include "abstractenv.h"

namespace rlly
{
namespace wrappers
{

/**
 * @brief Wrapper such that the observation and action spaces of the wrapper environment are not 
 * necessarily the same as in the original environment.
 * @tparam EnvType type of the original environment (can be an abstract class)
 * @tparam S_space type of state space of the wrapper (e.g. spaces::Box, spaces::Discrete)
 * @tparam A_space type of action space of the wrapper (e.g. spaces::Box, spaces::Discrete)
 */
template <typename EnvType, typename S_space, typename A_space>
class BaseWrapper: public rlly::env::Env<S_space, A_space>
{
protected:
    S_space foo_obs_space;
    A_space foo_act_space;
public:
    BaseWrapper(EnvType& env): p_env(env){};
    ~BaseWrapper(){};

    // type of state and action variables
    using S_type = decltype(foo_obs_space.sample());
    using A_type = decltype(foo_act_space.sample());

    /**
     * reset() must be implemented by derived class
     */
    virtual S_type reset()=0;

    /**
     * step() must be implemented by derived class
     */
    virtual env::StepResult<S_type> step(A_type action)=0;

    /**
     *  Reference to the wrapped environment.
     */
    EnvType& p_env;

    /**
     * @brief Returns a null pointer. Prevents the wrapper from being cloned.
     */
    virtual std::unique_ptr<env::Env<S_space, A_space>> clone() const override { return nullptr;};

    /**
     * Set seed
     */
    void set_seed(int _seed)
    {
        this->set_seed(_seed+123);
        p_env.set_seed(_seed);
    };
};


} // namespace wrappers
} // namespace rlly

#endif