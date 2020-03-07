#include <iostream>
#include "render2d.h"
#include "utils.h"

namespace rlly
{
namespace render
{

std::vector<utils::render::Scene> Render2D::data;

//

utils::render::Scene Render2D::background;

// 

unsigned int Render2D::time_count = 0;

//

void Render2D::set_data(std::vector<utils::render::Scene> _data)
{
    data = _data;
}

void Render2D::set_background(utils::render::Scene _background)
{
    background = _background;
}


//

void Render2D::initGL()
{
    // Nothing implemented yet    
}

//

void Render2D::timer(int value)
{
    glutPostRedisplay();
    glutTimerFunc(refresh_interval, timer, 0);
}

// 

void Render2D::display()
{
    // Set background color (clear background)
    glClearColor(background_color[0], background_color[1], background_color[2], 1.0f); 
    glClear(GL_COLOR_BUFFER_BIT);    

    // Display background
    for(auto p_shape = background.shapes.begin(); p_shape != background.shapes.end(); ++p_shape)
        draw_geometric2d(*p_shape);
    
    // Display objects
    if (data.size() > 0)
    {
        int idx = time_count % data.size();
        for(auto p_shape = data[idx].shapes.begin(); p_shape != data[idx].shapes.end(); ++ p_shape)
            draw_geometric2d(*p_shape);
    }
    time_count += 1; // Increment time 
    glFlush();       // Render now
}

// 

int Render2D::run_graphics()
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

//

void Render2D::draw_geometric2d(utils::render::Geometric2D geom)
{
    // Begin according to geometric primitive
    if      (geom.type == "GL_POINTS")         glBegin(GL_POINTS);
    else if (geom.type == "GL_LINES")          glBegin(GL_LINES);
    else if (geom.type == "GL_LINE_STRIP")     glBegin(GL_LINE_STRIP);
    else if (geom.type == "GL_LINE_LOOP")      glBegin(GL_LINE_LOOP);
    else if (geom.type == "GL_POLYGON")        glBegin(GL_POLYGON);
    else if (geom.type == "GL_TRIANGLES")      glBegin(GL_TRIANGLES);
    else if (geom.type == "GL_TRIANGLE_STRIP") glBegin(GL_TRIANGLE_STRIP);
    else if (geom.type == "GL_TRIANGLE_FAN")   glBegin(GL_TRIANGLE_FAN);
    else if (geom.type == "GL_QUADS")          glBegin(GL_QUADS);
    else if (geom.type == "GL_QUAD_STRIP")     glBegin(GL_QUAD_STRIP);
    else std::cerr << "Error in Render2D::draw_geometric2d: invatid primitive type!" << std::endl;
    
    // Set color
    glColor3f(geom.color[0], geom.color[1], geom.color[2]); 
    
    // Create vertices
    int n_vertices = geom.vertices.size();
    for(int ii = 0; ii < n_vertices; ii++)
    {
        float x = geom.vertices[ii][0];
        float y = geom.vertices[ii][1];
        glVertex2f(x, y);
    }

    //
    glEnd();
}

}
}