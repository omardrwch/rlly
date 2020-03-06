#include "graphrender.h"

namespace rlly
{
namespace render
{

TimeGraph2D GraphRender::time_graph_2d;

//

std::list<Polygon2D> GraphRender::background;

// 

int GraphRender::time_count = 0;

// 

void GraphRender::set_graph(TimeGraph2D _time_graph_2d)
{
    time_graph_2d = _time_graph_2d;
}

//

void GraphRender::set_background(std::list<Polygon2D> _background)
{
    background = _background;
}

// 

void GraphRender::draw_polygon(Polygon2D polygon)
{
    int n_vertices = polygon.vertices.size();
    glBegin(GL_POLYGON);
        glColor3f(polygon.color[0], polygon.color[1], polygon.color[2]);
        for(int ii = 0; ii < n_vertices; ii++)
        {
            glVertex2f( polygon.vertices[ii][0], polygon.vertices[ii][1]);
        }
    glEnd();
}

//

void GraphRender::draw_edge(float x0, float y0, float x1, float y1)
{
    glBegin(GL_LINES); 
        glColor3f(edge_color[0], edge_color[1], edge_color[2]); 
        glVertex2f( x0,  y0);
        glVertex2f( x1,  y1);
    glEnd();
}

//

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

//

void GraphRender::initGL()
{
    // Nothing implemented yet    
}

//

void GraphRender::timer(int value)
{
    glutPostRedisplay();
    glutTimerFunc(refresh_interval, timer, 0);
}

// 

void GraphRender::display()
{
    // Set background color (clear background)
    glClearColor(background_color[0], background_color[1], background_color[2], 1.0f); 
    glClear(GL_COLOR_BUFFER_BIT);    

    // Draw background
    for(auto p_polygon = background.begin(); 
             p_polygon != background.end();
             ++p_polygon)
    {
        draw_polygon(*p_polygon);
    }

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

// 

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

}
}