/*
    To run this example:
    $ bash scripts/compile.sh render_example && ./build/examples/render_example
*/

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include "render.h"


int main()
{
    // render object
    rlly::render::Render2D render2D;
    render2D.set_refresh_interval(2000); // in milliseconds

    // Create a background
    std::vector<rlly::utils::render::Scene2D> data;
    rlly::utils::render::Scene2D scene1, scene2;
    rlly::utils::render::Geometric2D shape1, shape2, shape3, shape4;

    // Define 4 polygons
    shape1.type = "GL_POLYGON";
    shape1.add_vertex(0.0, 0.0);
    shape1.add_vertex(1.0, 0.0);
    shape1.add_vertex(1.0, 1.0);
    shape1.add_vertex(0.0, 1.0);
    shape1.set_color(0.5, 0.0, 0.0);

    shape2.type = "GL_POLYGON";
    shape2.add_vertex(0.0, 0.0);
    shape2.add_vertex(-1.0, 0.0);
    shape2.add_vertex(-1.0, 1.0);
    shape2.add_vertex(0.0, 1.0);
    shape2.set_color(0.0, 0.5, 0.0);


    shape3.type = "GL_POLYGON";
    shape3.add_vertex(0.0, 0.0);
    shape3.add_vertex(-1.0, 0.0);
    shape3.add_vertex(-1.0, -1.0);
    shape3.add_vertex(0.0, -1.0);
    shape3.set_color(0.0, 0.0, 0.5);


    shape4.type = "GL_POLYGON";
    shape4.add_vertex(0.0, 0.0);
    shape4.add_vertex(1.0, 0.0);
    shape4.add_vertex(1.0, -1.0);
    shape4.add_vertex(0.0, -1.0);

    // scene1 consists of shape1 and shape3
    scene1.add_shape(shape1);
    scene1.add_shape(shape3);

    // scene2 consists of shape3 and shape4
    scene2.add_shape(shape2);
    scene2.add_shape(shape4);

    // data is a vector of scenes
    data.push_back(scene1);
    data.push_back(scene2);

    // Render data
    render2D.set_data(data);
    render2D.run_graphics();
    return 0;
}