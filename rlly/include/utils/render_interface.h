#ifndef __RLLY_RENDER_INTERFACE_H__
#define __RLLY_RENDER_INTERFACE_H__

#include <vector>
#include "render_data.h"

namespace rlly
{
namespace utils
{
namespace render
{

template <typename S>
class RenderInterface2D
{
private:
    /* data */
public:
    RenderInterface2D(){};
    ~RenderInterface2D(){};

    /*

        Methods and attributes used for graph rendering

    */

    /**
     * Set to true to enable rendering.
     */
    bool rendering2d_enabled = false;

    /**
     * Retuns a scene (list of shapes) representing the state
     * @param state_var
     */
    virtual utils::render::Scene2D get_scene_for_render2d(S state_var)=0;    
    
    /**
     * Retuns a scene (list of shapes) representing the background
     */
    virtual utils::render::Scene2D get_background_for_render2d(){return utils::render::Scene2D();};

    /**
     *  Refresh interval of rendering (in milliseconds)
     */
    int refresh_interval_for_render2d = 50;

    /**
     * Clipping area for rendering (left, right, bottom, top). Default = {-1.0, 1.0, -1.0, 1.0}
     */
    std::vector<float> clipping_area_for_render2d = {-1.0, 1.0, -1.0, 1.0};
};




} // namespace render
} // namespace utils
} // namespace rlly

#endif