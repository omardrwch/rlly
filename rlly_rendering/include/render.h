/**
 * @file
 * @brief Class for rendering the environments using freeglut.
 * @details Based on the OpenGL tutorial https://www3.ntu.edu.sg/home/ehchua/programming/opengl/CG_Introduction.html 
 */


#ifndef __RLLY_RENDER_H__
#define __RLLY_RENDER_H__

#include <GL/freeglut.h> 
#include <iostream>
#include <vector>

namespace rlly
{
namespace render
{


/**
 * @brief
 */
class TimeGraph2D
{
private:
public:
    // 2-dimensional vector of double
    typedef std::vector<std::vector<double>> vec_2d;

    TimeGraph2D(){};
    TimeGraph2D(vec_2d x_values, vec_2d y_values);
    ~TimeGraph2D(){};

    void set_data(vec_2d x_values, vec_2d y_values);

    int n_nodes = 0;
    vec_2d x_values;  // shape (n_nodes, time)
    vec_2d y_values;  // shape (n_nodes, time)
};

TimeGraph2D::TimeGraph2D(vec_2d _x_values, vec_2d _y_values)
{
    set_data(_x_values, _y_values);
}

void TimeGraph2D::set_data(vec_2d _x_values, vec_2d _y_values)
{
    /**
     * @todo throw exception if x_values.size() != y_values.size()
     */
    n_nodes = x_values.size(); 
    x_values = _x_values;
    y_values = _y_values;
}


class GraphRender
{
private:
    // Window refresh inteval (in milliseconds)
    static const int refresh_interval = 100; 

    // Window size (in pixels)
    static const int window_size = 640;

    // Node size
    static constexpr float node_size = 0.025f;

    // Node color 
    static constexpr float node_color[3] = {0.0, 0.0, 0.0}; 

    // Background color
    static constexpr float background_color[3] = {0.5, 0.5, 0.5}; 

    // Graph to be rendered
    static TimeGraph2D time_graph_2d;

    // Callback function, handler for window re-paint
    static void display();

    // Draw a node
    static void draw_node(float node_x, float node_y);

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
};

TimeGraph2D GraphRender::time_graph_2d;

void GraphRender::set_graph(TimeGraph2D _time_graph_2d)
{
    time_graph_2d = _time_graph_2d;
}

void GraphRender::draw_node(float node_x, float node_y)
{
    // Draw a square centered at origin
    glBegin(GL_QUADS);              // Each set of 4 vertices form a quad
        glColor3f(node_color[0], node_color[1], node_color[2]); // Red
        glVertex2f( node_x - node_size, node_y - node_size);    // x, y
        glVertex2f( node_x + node_size, node_y - node_size);
        glVertex2f( node_x + node_size, node_y + node_size);
        glVertex2f( node_x - node_size, node_y + node_size);
    glEnd();
}

void GraphRender::display()
{
    float sqr_size = 0.5;
    glClearColor(background_color[0], background_color[1], background_color[2], 1.0f); // Set background color to black and opaque
    glClear(GL_COLOR_BUFFER_BIT);         // Clear the color buffer (background)
 
    // Draw a node
    draw_node(-0.25, 0.5);

    // Check if there is a graph to be shown
    if (time_graph_2d.n_nodes == 0 )
    {
        std::cout << "No graph!" << std::endl;
    }
   
    glFlush();  // Render now
}

int GraphRender::run_graphics()
{
    int argc = 0;
    char **argv;
    glutInit(&argc, argv);                 // Initialize GLUT

    // Continue execution after window is closed
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
                  GLUT_ACTION_GLUTMAINLOOP_RETURNS);

    glutInitWindowSize(window_size, window_size);   // Set the window's initial width & height
    glutInitWindowPosition(50, 50); // Position the window's initial top-left corner

    glutCreateWindow("Render"); // Create a window with the given title
    glutDisplayFunc(display); // Register display callback handler for window re-paint
    glutMainLoop();           // Enter the event-processing loop
    return 0;
}


} // namspace render
} // namespace rlly

#endif
