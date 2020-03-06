/**
 * @file
 * @brief Class for rendering the environments using freeglut.
 * @details Based on the OpenGL tutorial at https://www3.ntu.edu.sg/home/ehchua/programming/opengl/CG_Introduction.html 
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
    TimeGraph2D(vec_2d x_values, vec_2d y_values, vec_2d _edges);
    ~TimeGraph2D(){};

    void set_nodes(vec_2d _x_values, vec_2d _y_values);
    void set_edges(vec_2d _edges);

    int n_nodes = 0;
    int n_edges = 0;
    int time = 0;
    vec_2d x_values;  // shape (n_nodes, time)
    vec_2d y_values;  // shape (n_nodes, time)
    vec_2d edges;     // shape (n_edges, 2)
};

TimeGraph2D::TimeGraph2D(vec_2d _x_values, vec_2d _y_values, vec_2d _edges)
{
    set_nodes(_x_values, _y_values);
}

void TimeGraph2D::set_nodes(vec_2d _x_values, vec_2d _y_values)
{
    /**
     * @todo throw exception if x_values.size() != y_values.size()
     */
    x_values = _x_values;
    y_values = _y_values;
    n_nodes = _x_values.size(); 
    if (n_nodes > 0)
    {
        time = _x_values[0].size();
    }
}

void TimeGraph2D::set_edges(vec_2d _edges)
{
    edges = _edges;
    n_edges = _edges.size();
}


class GraphRender
{
private:
    // Window refresh inteval (in milliseconds)
    static const int refresh_interval = 100; 

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


void GraphRender::set_graph(TimeGraph2D _time_graph_2d)
{
    time_graph_2d = _time_graph_2d;
}

void GraphRender::draw_edge(float x0, float y0, float x1, float y1)
{
    glBegin(GL_LINES); 
        glColor3f(edge_color[0], edge_color[1], edge_color[2]); 
        glVertex2f( x0,  y0);
        glVertex2f( x1,  y1);
    glEnd();
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


void GraphRender::initGL()
{
    // Nothing implemented yet    
}

void GraphRender::timer(int value)
{
    glutPostRedisplay();
    glutTimerFunc(refresh_interval, timer, 0);
}

void GraphRender::display()
{
    // Set background color (clear background)
    glClearColor(background_color[0], background_color[1], background_color[2], 1.0f); 
    glClear(GL_COLOR_BUFFER_BIT);    

    // Display graph
    if (time_graph_2d.n_nodes != 0 )
    {
        int time = 0;
        if (time_graph_2d.time > 0) time = time_count % time_graph_2d.time;

        // Edges
        if (time_graph_2d.n_edges != 0)
        {
            for(int ee = 0; ee < time_graph_2d.n_edges; ee++)
            {
                float x0 = time_graph_2d.x_values[time_graph_2d.edges[ee][0]][time];
                float y0 = time_graph_2d.y_values[time_graph_2d.edges[ee][0]][time];

                float x1 = time_graph_2d.x_values[time_graph_2d.edges[ee][1]][time];
                float y1 = time_graph_2d.y_values[time_graph_2d.edges[ee][1]][time];

                draw_edge(x0, y0, x1, y1);
            }
        }
        // Nodes
        for(int nn = 0; nn < time_graph_2d.n_nodes; nn++)
        {
            draw_node( time_graph_2d.x_values[nn][time], time_graph_2d.y_values[nn][time]);
        }
    }
   
    time_count += 1; // Increment time 
    glFlush();       // Render now
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
    glutTimerFunc(0, timer, 0);     // First timer call immediately
    initGL();
    glutMainLoop();           // Enter the event-processing loop
    return 0;
}


} // namspace render
} // namespace rlly

#endif
