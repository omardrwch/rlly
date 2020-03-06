/**
 * @file
 * @brief Contains class for rendering objects of type TimeGraph2D
 */


#ifndef __RLLY_GRAPH_RENDER_H__
#define __RLLY_GRAPH_RENDER_H__

#include <GL/freeglut.h> 
#include <iostream>
#include <vector>
#include <list>
#include "timegraph2d.h"
#include "polygon2d.h"

namespace rlly
{
namespace render
{

class GraphRender
{
private:
    // Window refresh inteval (in milliseconds)
    static const int refresh_interval = 50; 

    // Window size (in pixels)
    static const int window_size = 640;

    // Node size
    static constexpr float node_size = 0.02f;

    // Node color 
    static constexpr float node_color[3] = {0.0, 0.0, 0.0}; 

    // Edge color
    static constexpr float edge_color[3] = {0.75, 0.25, 0.25}; 

    // Background color
    static constexpr float background_color[3] = {0.75, 0.75, 0.75}; 

    // Backgroud image (represented by a list of 2D polygons)
    static std::list<Polygon2D> background;

    // Graph to be rendered
    static TimeGraph2D time_graph_2d;

    // Time counter 
    static int time_count;

    // Initialize GL
    static void initGL();

    // Callback function, handler for window re-paint
    static void display();

    // Timer, to call display() periodically (period = refresh_interval)
    static void timer(int value);

    // Draw a node
    static void draw_node(float node_x, float node_y);

    // Draw an edge
    static void draw_edge(float x0, float y0, float x1, float y1);

    // Draw a polygon
    static void draw_polygon(Polygon2D polygon);

public:
    GraphRender(){};
    ~GraphRender(){};
    
    /**
     * Main function, set up the window and enter the event-processing loop
     */ 
    int run_graphics();

    /**
     * Set graph to be rendered
     */
    void set_graph(TimeGraph2D _time_graph_2d);

    /**
     * Set background
     */
    void set_background(std::list<Polygon2D> _background);

};


} // namspace render
} // namespace rlly

#endif
