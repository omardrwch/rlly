#include <iostream>
#include <cmath>
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

int Render2D::refresh_interval = 50;
unsigned int Render2D::time_count = 0;
std::string Render2D::window_name = "render";
std::vector<float> Render2D::clipping_area;
int Render2D::window_width = 640;
int Render2D::window_height = 640;


//

Render2D::Render2D()
{
    // setting some defaults
    clipping_area.push_back(-1.0);
    clipping_area.push_back( 1.0);
    clipping_area.push_back(-1.0);
    clipping_area.push_back( 1.0);
}


//

void Render2D::set_window_name(std::string name)
{
    window_name = name;
}

void Render2D::set_refresh_interval(int interval)
{
    refresh_interval = interval;
}

void Render2D::set_clipping_area(std::vector<float> area)
{
    clipping_area = area; 
    int base_size = std::max(window_width, window_height);
    float width_range  = area[1] - area[0];
    float height_range = area[3] - area[2];
    float base_range   = std::max(width_range, height_range);
    width_range /= base_range;
    height_range /= base_range;
    // update window width and height
    window_width  = (int) (base_size*width_range);
    window_height = (int) (base_size*height_range);
}

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
    // set clipping area
    glMatrixMode(GL_PROJECTION);  // To operate on the Projection matrix
    glLoadIdentity();  
    gluOrtho2D(clipping_area[0], clipping_area[1], clipping_area[2], clipping_area[3]); 
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
    char **argv = nullptr;
    glutInit(&argc, argv);                 // Initialize GLUT

    // Continue execution after window is closed
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
                  GLUT_ACTION_GLUTMAINLOOP_RETURNS);

    glutInitWindowSize(window_width, window_height);   // Set the window's initial width & height
    glutInitWindowPosition(50, 50); // Position the window's initial top-left corner

    glutCreateWindow(window_name.c_str()); // Create a window with the given title
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