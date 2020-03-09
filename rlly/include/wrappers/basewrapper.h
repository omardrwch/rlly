#ifndef __RLLY_BASE_WRAPPER_H__
#define __RLLY_BASE_WRAPPER_H__

#include <memory>
#include <string>
#include "env.h"

namespace rlly
{
namespace env
{
namespace wrappers
{

template <typename S, typename A>
class Wrapper
{
public:
    Wrapper(Env<S, A>& env);
    ~Wrapper(){};

    /**
     *  Pointer to the wrapped environment.
     */
    std::unique_ptr<Env<S, A>> p_env;

    // reset 
    virtual S reset();
    // step
    virtual StepResult<S> step(A action);

    // id
    std::string id;
    // pointer to observation space
    spaces::Space<S>* p_observation_space;
    // pointer to action space
    spaces::Space<A>* p_action_space;


    // 2d rendering flag
    bool rendering2d_enabled;

    // Retuns a scene (list of shapes) representing the state
    virtual utils::render::Scene get_scene_for_render2d(S state_var);    
    
    // Retuns a scene (list of shapes) representing the background
    virtual utils::render::Scene get_background_for_render2d();

    // Refresh interval of rendering (in milliseconds)
    int refresh_interval_for_render2d;

    // Clipping area for 2d rendering 
    std::vector<float> clipping_area_for_render2d;

};

template <typename S, typename A>
Wrapper<S, A>::Wrapper(Env<S, A>& env)
{
    p_env               = env.clone();
    id                  = (*p_env).id + "Wrapper";
    p_observation_space = (*p_env).p_observation_space;
    p_action_space      = (*p_env).p_action_space;

    // rendering parameters
    rendering2d_enabled = (*p_env).rendering2d_enabled;
    clipping_area_for_render2d = (*p_env).clipping_area_for_render2d;
    refresh_interval_for_render2d = (*p_env).refresh_interval_for_render2d;
}

template <typename S, typename A>
S Wrapper<S, A>::reset()
{
    return (*p_env).reset();
}

template <typename S, typename A>
StepResult<S> Wrapper<S, A>::step(A action)
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


} // namespace wrappers
} // namespace env
} // namespace rlly
#endif