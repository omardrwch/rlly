/*
    To run this example:
    $ bash scripts/compile.sh render_example && ./build/examples/render_example
*/

#include <iostream>
#include <vector>
#include <string>
#include "rlly.hpp"
#include "render.h"
#include "GL/freeglut.h"


int main()
{
    rlly::render::GraphRender grender;
    rlly::render::TimeGraph2D graph;

    /**
     * Define graph
     */ 
    int n_objects = 5;
    rlly::utils::vec::vec_2d x_values = rlly::utils::vec::get_zeros_2d(n_objects, 1);
    rlly::utils::vec::vec_2d y_values = rlly::utils::vec::get_zeros_2d(n_objects, 1);
    rlly::utils::vec::vec_2d edges = {{0, 1}, {1, 2}, {2, 3}, {3, 4}};


    for(int ii = 0; ii < n_objects; ii++)
    {
        x_values[ii][0] = (1.0*ii)/n_objects;
        y_values[ii][0] = (1.0*ii)/n_objects;
    }
    std::cout << "size = " << x_values.size() << std::endl;
    graph.set_nodes(x_values, y_values);
    graph.set_edges(edges);
    grender.set_graph(graph);

    /**
     * Render graph
     */
    grender.run_graphics();

    std::cout << "Back to execution!" << std::endl;

    std::cout << "hereeeee" << std::endl;
    return 0;
}