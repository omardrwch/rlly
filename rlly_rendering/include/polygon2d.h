#ifndef __RLLY_POLYGON_2D_H__
#define __RLLY_POLYGON_2D_H__

#include <vector>

struct Polygon2D
{
    /**
     * 2d vector of shape (n_vertices, 2)
     * vertices[i][j] = j-th cordinnate of vertex i
     */
    std::vector<std::vector<float>> vertices; 
    /**
     * vector with 3 elements, contaning the color of the polygon
     * colors[i] = RGB color of vertex i
     */
    std::vector<float> color;   
};


#endif