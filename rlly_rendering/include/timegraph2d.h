/**
 * @file
 * @brief  Contains a class to define a graph whose nodes are points in 2D that can change with time.
 */

#ifndef __RLLY_TIMEGRAPH2D_H__
#define __RLLY_TIMEGRAPH2D_H__

#include <vector>

namespace rlly
{
namespace render
{

/**
 * @brief Class that defines a graph whose nodes are represented points in [-1, 1]^2 that can 
 * change with time.
 */
class TimeGraph2D
{
private:
public:
    // 2-dimensional vector of double
    typedef std::vector<std::vector<double>> vec_2d;
    typedef std::vector<std::vector<int>> ivec_2d;


    TimeGraph2D(){};
    TimeGraph2D(vec_2d x_values, vec_2d y_values, ivec_2d _edges);
    ~TimeGraph2D(){};
    
    /**
     * Set coordinates of the nodes in the graph
     * @param _x_values vector of double, size (n_nodes, time)
     * @param _y_values vector of double, size (n_nodes, time)
     */
    void set_nodes(vec_2d _x_values, vec_2d _y_values);
    void set_edges(ivec_2d _edges);

    /**
     * Number of nodes in the graph
     */
    int n_nodes = 0;

    /**
     *  Number of edges
     */ 
    int n_edges = 0;


    /**
     * Time counter
     */
    int time = 0;

    /**
     * 2d vector of dimension (n_nodes, time) containing 
     * the x-coordinate of the each node as a function of time.
     */
    vec_2d x_values;  // shape (n_nodes, time)

    /**
     * 2d vector of dimension (n_nodes, time) containing 
     * the y-coordinate of the each node as a function of time.
     */
    vec_2d y_values;  // shape (n_nodes, time)

    /**
     * 2d vector of dimension (n_edges, 2) such that edges[i][0] is a source 
     * node and edge[i][1] is the destination node.
     */
    ivec_2d edges;     // shape (n_edges, 2)
};

} // namspace render
} // namespace rlly


#endif