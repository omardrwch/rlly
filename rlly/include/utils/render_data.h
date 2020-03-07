/**
 * @file
 * @brief Data structures used for rendering.
 */

#ifndef __RLLY_RENDER_DATA_H__
#define __RLLY_RENDER_DATA_H__

#include <vector>
#include <string>

namespace rlly
{
namespace utils
{
/**
 * Rendering utils
 */
namespace render
{


/**
 * Data representing an OpenGL geometric primitive in 2D
 */ 
struct Geometric2D
{
    /**
     * Primitive type (GL_LINE_STRIP by defaut)
     * Possibilities:
     *      GL_POINTS
     *      GL_LINES
     *      GL_LINE_STRIP
     *      GL_LINE_LOOP
     *      GL_POLYGON
     *      GL_TRIANGLES
     *      GL_TRIANGLE_STRIP
     *      GL_TRIANGLE_FAN
     *      GL_QUADS
     *      GL_QUAD_STRIP
     */
    std::string type = "GL_LINE_STRIP";

    /**
     * vector with 3 elements, contaning the color of the shape
     * gray by default
     */
    std::vector<float> color = {0.25f, 0.25f, 0.25f};   

    /**
     * 2d vector of shape (n_vertices, 2)
     * vertices[i][j] = j-th cordinnate of vertex i
     */
    std::vector<std::vector<float>> vertices; 

    /**
     * Add vertex 
     */ 
    void add_vertex(std::vector<float> vertex) { vertices.push_back(vertex); }; 
};

/**
 * Data representing a scene, which is a vector of Geometric2D objects
 */
struct Scene
{
    /**
     * Vector of 2D shapes represeting the scene
     */
    std::vector<Geometric2D> shapes;
};


struct Polygon2D
{
    /**
     * 2d vector of shape (n_vertices, 2)
     * vertices[i][j] = j-th cordinnate of vertex i
     */
    std::vector<std::vector<float>> vertices; 
    /**
     * vector with 3 elements, contaning the color of the polygon
     */
    std::vector<float> color;   
};


}
}
}

#endif