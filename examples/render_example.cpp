/*
    To run this example:
    $ bash scripts/compile.sh render_example && ./build/examples/render_example
*/

#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include "env.h"
#include "space.h"
#include "utils.h"

// #include "rlly.hpp"

#include "render.h"


int main()
{

    rlly::render::Render2D render2D;

    // Create a background
    std::vector<rlly::utils::render::Scene> data;
    rlly::utils::render::Scene scene1, scene2;
    rlly::utils::render::Geometric2D geom1, geom2;

    geom1.type = "GL_TRIANGLES";
    geom1.add_vertex(0.0, 0.0);
    geom1.add_vertex(1.0, 1.0);
    geom1.add_vertex(1.0, 0.0);
    scene1.add_shape(geom1);

    geom2.type = "GL_TRIANGLES";
    geom2.add_vertex(0.0, 0.0);
    geom2.add_vertex(-1.0, -1.0);
    geom2.add_vertex(-1.0, 0.0);
    scene2.add_shape(geom2);

    data.push_back(scene1);
    data.push_back(scene2);

    // Render
    render2D.set_data(data);
    render2D.run_graphics();


    // =======================================================================


    // rlly::env::MountainCar env;
    // int horizon = 50;
    
    // std::vector<std::vector<double>> states;
    // env.set_seed(76345);
    // for(int hh = 1; hh < horizon; hh++)
    // {
    //     auto action = env.action_space.sample();
    //     auto step_result = env.step(action);
    //     std::cout << "action  " << action << ", angle = " << step_result.next_state[0] <<std::endl;
    //     states.push_back(step_result.next_state);
    // }

    // rlly::render::render_env(states, env);

    // =======================================================================

    // rlly::render::GraphRender grender;
    // rlly::render::TimeGraph2D graph;

    // /**
    //  * Define graph
    //  */ 
    // int n_objects = 5;
    // int time = 50;
    // rlly::utils::vec::vec_2d x_values = rlly::utils::vec::get_zeros_2d(n_objects, time);
    // rlly::utils::vec::vec_2d y_values = rlly::utils::vec::get_zeros_2d(n_objects, time);
    // rlly::utils::vec::vec_2d edges = {{0, 1}, {1, 2}, {2, 3}, {3, 4}};


    // for(int tt = 0; tt < time; tt ++)
    // {
    //     for(int ii = 0; ii < n_objects; ii++)
    //     {
    //         float theta = 3.1415*ii/6.0 + tt/5.0;
    //         // x_values[ii][tt] = 0.5*std::cos(theta); 
    //         x_values[ii][tt] = 2*tt*1.0/time -1 ; 
    //         y_values[ii][tt] = 0.5*std::sin(theta);
    //     }
    // }
    // std::cout << "size = " << x_values.size() << std::endl;
    // graph.set_nodes(x_values, y_values);
    
    // graph.set_edges(edges);
    // grender.set_graph(graph);

    // /**
    //  * Define background
    // */ 
    // std::vector<std::vector<float>> vertices1 = {{0.5, 0.5}, 
    //                                              {0.5, -0.5},
    //                                              {-0.25, -0.25},
    //                                              {-0.25, 0.25}};
    // std::vector<float> color1 = {0.5, 0.75, 0.5};
    // Polygon2D polygon1 = {vertices1, color1};

    // std::list<Polygon2D> background = {polygon1};

    // grender.set_background(background);

    // /**
    //  * Render graph
    //  */
    // grender.run_graphics();

    // std::cout << "Back to execution!" << std::endl;

    // std::cout << "hereeeee" << std::endl;
    return 0;
}