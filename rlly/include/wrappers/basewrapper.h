#ifndef __RLLY_BASE_WRAPPER_H__
#define __RLLY_BASE_WRAPPER_H__

#include <memory>
#include <string>
#include "env.h"

namespace rlly
{
namespace wrappers
{

template <typename S, typename A, typename S_space, typename A_space>
class Wrapper: public env::Env<S, A, S_space, A_space>
{
public:
    Wrapper(env::Env<S, A, S_space, A_space>& env);
    ~Wrapper(){};

    /**
     *  Pointer to the wrapped environment.
     */
    std::unique_ptr<env::Env<S, A, S_space, A_space>> p_env;

    // reset 
    virtual S reset() override;
    // step
    virtual env::StepResult<S> step(A action) override;

    /**
     * @brief Returns a clone of the (!) wrapped (!) environment.
     */
    virtual std::unique_ptr<env::Env<S, A, S_space, A_space>> clone() const override;

    // Retuns a scene (list of shapes) representing the state
    virtual utils::render::Scene get_scene_for_render2d(S state_var) override;    
    
    // Retuns a scene (list of shapes) representing the background
    virtual utils::render::Scene get_background_for_render2d() override;

    // Set seed
    void set_seed(int _seed);

    // Get state
    S get_state();
};


template <typename S, typename A, typename S_space, typename A_space>
Wrapper<S, A, S_space, A_space>::Wrapper(env::Env<S, A, S_space, A_space>& env)
{
    p_env               = env.clone();
    this->id            = (*p_env).id + "Wrapper";
    this->observation_space = (*p_env).observation_space;
    this->action_space      = (*p_env).action_space;

    // rendering parameters
    this->rendering2d_enabled = (*p_env).rendering2d_enabled;
    this->clipping_area_for_render2d = (*p_env).clipping_area_for_render2d;
    this->refresh_interval_for_render2d = (*p_env).refresh_interval_for_render2d;
}

template <typename S, typename A, typename S_space, typename A_space>
S Wrapper<S, A, S_space, A_space>::reset()
{
    return (*p_env).reset();
}

template <typename S, typename A, typename S_space, typename A_space>
env::StepResult<S> Wrapper<S, A, S_space, A_space>::step(A action)
{
    return (*p_env).step(action);
}


template <typename S, typename A, typename S_space, typename A_space>
utils::render::Scene Wrapper<S, A, S_space, A_space>::get_scene_for_render2d(S state_var)
{
    return (*p_env).get_scene_for_render2d(state_var);
}

template <typename S, typename A, typename S_space, typename A_space>
utils::render::Scene Wrapper<S, A, S_space, A_space>::get_background_for_render2d()
{
    return (*p_env).get_background_for_render2d();
}


template <typename S, typename A, typename S_space, typename A_space>
std::unique_ptr<env::Env<S, A, S_space, A_space>> Wrapper<S, A, S_space, A_space>::clone() const
{
    return (*p_env).clone();
}

template <typename S, typename A, typename S_space, typename A_space>
void Wrapper<S, A, S_space, A_space>::set_seed(int _seed)
{
    (*p_env).set_seed(_seed);
}

template <typename S, typename A, typename S_space, typename A_space>
S Wrapper<S, A, S_space, A_space>::get_state()
{
    return (*p_env).get_state();
}

} // namespace wrappers
} // namespace rlly
#endif