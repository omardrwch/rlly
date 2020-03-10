#ifndef __RLLY_BASE_WRAPPER_H__
#define __RLLY_BASE_WRAPPER_H__

#include <memory>
#include <string>
#include "env.h"

namespace rlly
{
namespace wrappers
{

template <typename S, typename A>
class Wrapper: public env::Env<S, A>
{
public:
    Wrapper(env::Env<S, A>& env);
    ~Wrapper(){};

    /**
     *  Pointer to the wrapped environment.
     */
    std::unique_ptr<env::Env<S, A>> p_env;

    // reset 
    virtual S reset() override;
    // step
    virtual env::StepResult<S> step(A action) override;

    /**
     * @brief Returns a clone of the (!) wrapped (!) environment.
     */
    virtual std::unique_ptr<env::Env<S, A>> clone() const override;

    // Retuns a scene (list of shapes) representing the state
    virtual utils::render::Scene get_scene_for_render2d(S state_var) override;    
    
    // Retuns a scene (list of shapes) representing the background
    virtual utils::render::Scene get_background_for_render2d() override;

    // Set seed
    void set_seed(int _seed);

    // Get state
    S get_state();
};


template <typename S, typename A>
Wrapper<S, A>::Wrapper(env::Env<S, A>& env)
{
    p_env               = env.clone();
    this->id            = (*p_env).id + "Wrapper";
    this->p_observation_space = (*p_env).p_observation_space;
    this->p_action_space      = (*p_env).p_action_space;

    // rendering parameters
    this->rendering2d_enabled = (*p_env).rendering2d_enabled;
    this->clipping_area_for_render2d = (*p_env).clipping_area_for_render2d;
    this->refresh_interval_for_render2d = (*p_env).refresh_interval_for_render2d;
}

template <typename S, typename A>
S Wrapper<S, A>::reset()
{
    return (*p_env).reset();
}

template <typename S, typename A>
env::StepResult<S> Wrapper<S, A>::step(A action)
{
    return (*p_env).step(action);
}


template <typename S, typename A>
utils::render::Scene Wrapper<S, A>::get_scene_for_render2d(S state_var)
{
    return (*p_env).get_scene_for_render2d(state_var);
}

template <typename S, typename A>
utils::render::Scene Wrapper<S, A>::get_background_for_render2d()
{
    return (*p_env).get_background_for_render2d();
}


template <typename S, typename A>
std::unique_ptr<env::Env<S, A>> Wrapper<S, A>::clone() const
{
    return (*p_env).clone();
}

template <typename S, typename A>
void Wrapper<S, A>::set_seed(int _seed)
{
    (*p_env).set_seed(_seed);
}

template <typename S, typename A>
S Wrapper<S, A>::get_state()
{
    return (*p_env).get_state();
}

} // namespace wrappers
} // namespace rlly
#endif