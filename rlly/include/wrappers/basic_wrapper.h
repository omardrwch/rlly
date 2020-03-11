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
 * @tparam EnvType type of the original environment
 * @tparam S type of state variables of the wrapper  (e.g. int, std::vector<double>)
 * @tparam A type of action variables of the wrapper (e.g. int, std::vector<double>)
 * @tparam S_space type of state space of the wrapper (e.g. spaces::Box, spaces::Discrete)
 * @tparam A_space type of action space of the wrapper (e.g. spaces::Box, spaces::Discrete)
 */
template <typename EnvType, typename S, typename A, typename S_space, typename A_space>
class BasicWrapper: rlly::env::Env<S_space, A_space>
{
public:
    BasicWrapper(EnvType& env)
    {
        p_env = env.clone();
        this->id  = (*p_env).id + "Wrapper";
    };
    ~BasicWrapper(){};

    /**
     *  Pointer to the wrapped environment.
     */
    std::unique_ptr<EnvType> p_env;

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
        p_env->set_seed(_seed);
    };
};



} // namespace wrappers
} // namespace rlly

#endif