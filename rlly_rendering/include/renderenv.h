/**
 * @file
 * @brief Contains class for rendering environments
 */


#ifndef __RLLY_RENDER_ENV_H__
#define __RLLY_RENDER_ENV_H__

#include <vector>
#include "graphrender.h"
#include "timegraph2d.h"

namespace rlly
{
namespace render
{

/**
 * @param states list of states to render
 * @param env    environment
 * @tparam EnvType represents Env<S, A> (see abstractenv.h)
 * @tparam S type of state space
 */
template <typename EnvType, typename S>
void render_env(std::vector<S> states, EnvType& env)
{
    if (env.graph_rendering_enabled)
    {
        int n_data = states.size();
        int n_nodes = env.graph_rendering_n_nodes;
        std::vector<std::vector<double>> states_x(n_nodes);  // shape (n_nodes, n_data)
        std::vector<std::vector<double>> states_y(n_nodes);  // shape (n_nodes, n_data)

        // Get 2d representation of states
        for(int nn = 0; nn < n_data; nn++)
        {
            std::vector<std::vector<float>> nodes_xy = env.get_nodes_for_graph_render(states[nn]); // shape (n_nodes, 2)
            for(int node = 0; node < n_nodes; node++)
            {
                states_x[node].push_back(nodes_xy[node][0]);
                states_y[node].push_back(nodes_xy[node][1]);
            }
        }

        // Background
        auto background = env.get_background_for_render();

        // Build graph
        rlly::render::TimeGraph2D graph;
        graph.set_nodes(states_x, states_y);

        // Render
        GraphRender renderer;
        renderer.set_graph(graph);
        renderer.set_background(background);
        renderer.run_graphics();
        std::cout << "graph_rendering_enabled for " << env.id << std::endl;
    }
}



} // namspace render
} // namespace rlly

#endif