#ifndef __RLLY_ISOMORPHIC_WRAPPER_H__
#define __RLLY_ISOMORPHIC_WRAPPER_H__

#include <iostream>
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
 * @tparam EnvType must NOT be abstract, we need to instantiate it to infer the types of state and action spaces
 */
template <typename EnvType>
class IsomorphicWrapper: public EnvType
{
private:
    EnvType foo_env;
    using S_space = decltype(foo_env.observation_space);
    using A_space = decltype(foo_env.action_space);
    S_space foo_obs_space;
    A_space foo_act_space;
public:
    IsomorphicWrapper(EnvType& env): p_env(env)
    {
        this->id                = p_env.id + "IsomorphicWrapper";
        this->observation_space = p_env.observation_space;
        this->action_space      = p_env.action_space;
    };
    ~IsomorphicWrapper(){};

    // type of state and action variables
    using S_type = decltype(foo_obs_space.sample());
    using A_type = decltype(foo_act_space.sample());

    /**
     *  Reference to the wrapped environment.
     */
    EnvType& p_env;

    // reset 
    virtual S_type reset() override
    {
        return p_env.reset();
    };

    // step
    virtual env::StepResult<S_type> step(A_type action) override
    {
        return p_env.step(action);
    };

    /**
     * @brief Returns a null pointer. Prevents the wrapper from being cloned.
     */
    virtual std::unique_ptr<env::Env<S_space, A_space>> clone() const override
    {
        std::cerr << "Error: trying to clone a wrapper, returning nullptr" << std::endl;
        return nullptr;
    };

    // Set seed
    void set_seed(int _seed);
};

template <typename EnvType>
void IsomorphicWrapper<EnvType>::set_seed(int _seed)
{
    p_env.set_seed(_seed);
    int seed = p_env.randgen.get_seed();
    this->observation_space.generator.seed(seed+123);
    this->action_space.generator.seed(seed+456);
}

} // namespace wrappers
} // namespace rlly
#endif