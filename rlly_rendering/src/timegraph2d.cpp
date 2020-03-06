#include "timegraph2d.h"

namespace rlly
{
namespace render
{

TimeGraph2D::TimeGraph2D(vec_2d _x_values, vec_2d _y_values, vec_2d _edges)
{
    set_nodes(_x_values, _y_values);
}

void TimeGraph2D::set_nodes(vec_2d _x_values, vec_2d _y_values)
{
    /**
     * @todo throw exception if x_values.size() != y_values.size()
     */
    x_values = _x_values;
    y_values = _y_values;
    n_nodes = _x_values.size(); 
    if (n_nodes > 0)
    {
        time = _x_values[0].size();
    }
}

void TimeGraph2D::set_edges(vec_2d _edges)
{
    edges = _edges;
    n_edges = _edges.size();
}


}
}