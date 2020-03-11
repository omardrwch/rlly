#ifndef __RLLY_ISOMORPHIC_WRAPPER_H__
#define __RLLY_ISOMORPHIC_WRAPPER_H__

#include <memory>
#include <string>
#include "env.h"

namespace rlly
{
namespace wrappers
{

/**
 * @brief Wrapper such that the observation and action spaces of the wrapper environment are the
 * same as the original environment. 
 * @details Useful to define wrappers like time limit.
 * @tparam S_space type of state space (e.g. spaces::Box, spaces::Discrete)
 * @tparam A_space type of action space (e.g. spaces::Box, spaces::Discrete)
 */
template <typename S_space, typename A_space>
class IsomorphicWrapper: public env::Env<S_space, A_space>
{
private:
    S_space foo_obs_space;
    A_space foo_act_space;
public:
    IsomorphicWrapper(env::Env<S_space, A_space>& env);
    ~IsomorphicWrapper(){};

    // type of state and action variables
    using S_type = decltype(foo_obs_space.sample());
    using A_type = decltype(foo_act_space.sample());

    /**
     *  Pointer to the wrapped environment.
     */
    std::unique_ptr<env::Env<S_space, A_space>> p_env;

    // reset 
    virtual S_type reset() override
    {
        return (*p_env).reset();
    };

    // step
    virtual env::StepResult<S_type> step(A_type action) override
    {
        return (*p_env).step(action);
    };

    /**
     * @brief Returns a null pointer. Prevents the wrapper from being cloned.
     */
    virtual std::unique_ptr<env::Env<S_space, A_space>> clone() const override
    {
        return nullptr;
    };

    // Set seed
    void set_seed(int _seed);
};


template <typename S_space, typename A_space>
IsomorphicWrapper<S_space, A_space>::IsomorphicWrapper(env::Env<S_space, A_space>& env)
{
    p_env               = env.clone();
    this->id            = (*p_env).id + "IsomorphicWrapper";
    this->observation_space = (*p_env).observation_space;
    this->action_space      = (*p_env).action_space;
}

template <typename S_space, typename A_space>
void IsomorphicWrapper<S_space, A_space>::set_seed(int _seed)
{
    (*p_env).set_seed(_seed);
    int seed = (*p_env).randgen.get_seed();
    this->observation_space.generator.seed(seed+123);
    this->action_space.generator.seed(seed+456);
}

} // namespace wrappers
} // namespace rlly
#endif