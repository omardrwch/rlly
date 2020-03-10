/**
 * @file
 * @brief Contains class for rendering lists of Geometric2D objects
 */


#ifndef __RLLY_RENDER_2D_H__
#define __RLLY_RENDER_2D_H__

#include <GL/freeglut.h> 
#include <iostream>
#include <vector>
#include <string>
#include <list>
#include "utils.h"

namespace rlly
{
namespace render
{

class Render2D
{
private:
    // Window width (in pixels)
    static int window_width;

     // Window height (in pixels)
    static int window_height;

    // Background color
    static constexpr float background_color[3] = {0.6, 0.75, 1.0}; 

    // Backgroud image 
    static utils::render::Scene2D background;

    // Data to be rendered (represented by a vector of scenes)
    static std::vector<utils::render::Scene2D> data;

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

    // Clipping area. Vector with elements {left, right, bottom, top}
    // Default = {-1.0, 1.0, -1.0, 1.0}
    static std::vector<float> clipping_area;

    // Window name
    static std::string window_name;

    // Window refresh inteval (in milliseconds)
    static int refresh_interval; 

public:
    Render2D();
    ~Render2D(){};
    
    /**
     * Main function, set up the window and enter the event-processing loop
     */ 
    int run_graphics();

    /**
     * Set scene to be rendered
     */
    void set_data(std::vector<utils::render::Scene2D> _data);

    /**
     * Set background
     */
    void set_background(utils::render::Scene2D _background);

    /**
     * Set window name
     */
    void set_window_name(std::string name);

    /**
     * Set refresh interval (in milliseconds)
     */
    void set_refresh_interval(int interval);

    /**
     * Set clipping area. window_width and window_height are adapted 
     * to respect the proportions of the clipping_area
     * @param area vector with elements {left, right, bottom, top}
     */ 
    void set_clipping_area(std::vector<float> area);
};


} // namspace render
} // namespace rlly

#endif
