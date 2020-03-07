/**
 * @file
 * @brief Contains class for rendering lists of Geometric2D objects
 */


#ifndef __RLLY_RENDER_2D_H__
#define __RLLY_RENDER_2D_H__

#include <GL/freeglut.h> 
#include <iostream>
#include <vector>
#include <list>
#include "timegraph2d.h"
#include "utils.h"

namespace rlly
{
namespace render
{

class Render2D
{
private:
    // Window refresh inteval (in milliseconds)
    static const int refresh_interval = 50; 

    // Window size (in pixels)
    static const int window_size = 640;

    // Background color
    static constexpr float background_color[3] = {0.6, 0.75, 1.0}; 

    // Backgroud image 
    static utils::render::Scene background;

    // Data to be rendered (represented by a list of scenes)
    static std::list<utils::render::Scene> data;

    // Time counter 
    static unsigned int time_count;

    // Initialize GL
    static void initGL();

    // Callback function, handler for window re-paint
    static void display();

    // Timer, to call display() periodically (period = refresh_interval)
    static void timer(int value);

    // Draw a 2D shape
    static void draw_geometric2d(utils::render::Geometric2D geom);

public:
    Render2D(){};
    ~Render2D(){};
    
    /**
     * Main function, set up the window and enter the event-processing loop
     */ 
    int run_graphics();

    /**
     * Set scene to be rendered
     */
    void set_data(std::list<utils::render::Scene> _data);

    /**
     * Set background
     */
    void set_background(utils::render::Scene _background);

};


} // namspace render
} // namespace rlly

#endif
