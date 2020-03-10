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
 */
template <typename S, typename A, typename S_space, typename A_space>
class IsomorphicWrapper: public env::Env<S, A, S_space, A_space>
{
public:
    IsomorphicWrapper(env::Env<S, A, S_space, A_space>& env);
    ~IsomorphicWrapper(){};

    /**
     *  Pointer to the wrapped environment.
     */
    std::unique_ptr<env::Env<S, A, S_space, A_space>> p_env;

    // reset 
    virtual S reset() override;

    // step
    virtual env::StepResult<S> step(A action) override;

    /**
     * @brief Returns a null pointer. Prevents the wrapper from being cloned.
     */
    virtual std::unique_ptr<env::Env<S, A, S_space, A_space>> clone() const override;

    // Retuns a scene (list of shapes) representing the state
    virtual utils::render::Scene get_scene_for_render2d(S state_var) override;    
    
    // Retuns a scene (list of shapes) representing the background
    virtual utils::render::Scene get_background_for_render2d() override;

    // Set seed
    void set_seed(int _seed);
};


template <typename S, typename A, typename S_space, typename A_space>
IsomorphicWrapper<S, A, S_space, A_space>::IsomorphicWrapper(env::Env<S, A, S_space, A_space>& env)
{
    p_env               = env.clone();
    this->id            = (*p_env).id + "IsomorphicWrapper";
    this->observation_space = (*p_env).observation_space;
    this->action_space      = (*p_env).action_space;

    // rendering parameters
    this->rendering2d_enabled = (*p_env).rendering2d_enabled;
    this->clipping_area_for_render2d = (*p_env).clipping_area_for_render2d;
    this->refresh_interval_for_render2d = (*p_env).refresh_interval_for_render2d;
}

template <typename S, typename A, typename S_space, typename A_space>
S IsomorphicWrapper<S, A, S_space, A_space>::reset()
{
    return (*p_env).reset();
}

template <typename S, typename A, typename S_space, typename A_space>
env::StepResult<S> IsomorphicWrapper<S, A, S_space, A_space>::step(A action)
{
    return (*p_env).step(action);
}


template <typename S, typename A, typename S_space, typename A_space>
utils::render::Scene IsomorphicWrapper<S, A, S_space, A_space>::get_scene_for_render2d(S state_var)
{
    return (*p_env).get_scene_for_render2d(state_var);
}

template <typename S, typename A, typename S_space, typename A_space>
utils::render::Scene IsomorphicWrapper<S, A, S_space, A_space>::get_background_for_render2d()
{
    return (*p_env).get_background_for_render2d();
}


template <typename S, typename A, typename S_space, typename A_space>
std::unique_ptr<env::Env<S, A, S_space, A_space>> IsomorphicWrapper<S, A, S_space, A_space>::clone() const
{
    return nullptr;
}

template <typename S, typename A, typename S_space, typename A_space>
void IsomorphicWrapper<S, A, S_space, A_space>::set_seed(int _seed)
{
    (*p_env).set_seed(_seed);
    int seed = (*p_env).randgen.get_seed();
    this->observation_space.generator.seed(seed+123);
    this->action_space.generator.seed(seed+456);
}

} // namespace wrappers
} // namespace rlly
#endif