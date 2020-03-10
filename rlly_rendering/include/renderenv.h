/**
 * @file
 * @brief Contains class for rendering environments
 */


#ifndef __RLLY_RENDER_ENV_H__
#define __RLLY_RENDER_ENV_H__

#include <iostream>
#include <vector>
#include "render2d.h"
#include "utils.h"

namespace rlly
{
namespace render
{

/**
 * @param env  Environment
 * @tparam EnvType an enviroment that implements a rendering interface
 * @tparam S type of state space
 */
template <typename EnvType>
void render_env(EnvType& env)
{
    if (env.rendering_enabled && env.rendering_type == "2d")
    {
        // Background
        auto background = env.get_background_for_render2d();

        // Data
        std::vector<utils::render::Scene2D> data;    
        int n_data = env.state_history_for_rendering.size();
        for(int ii = 0; ii < n_data; ii++)
        {
            utils::render::Scene2D scene = env.get_scene_for_render2d(env.state_history_for_rendering[ii]);
            data.push_back(scene);
        }   

        // Render
        Render2D renderer;
        renderer.set_window_name(env.id);
        renderer.set_refresh_interval(env.refresh_interval_for_render2d);
        renderer.set_clipping_area(env.clipping_area_for_render2d);
        renderer.set_data(data);
        renderer.set_background(background);
        renderer.run_graphics();
    }
    else
    {
        std::cerr << "Error: environement " << env.id << " is not enabled for rendering. Try calling env.enable_rendering()." << std::endl;
    }
    
}



} // namspace render
} // namespace rlly

#endif