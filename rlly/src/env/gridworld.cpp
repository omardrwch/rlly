#include <assert.h>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include "gridworld.h"
#include "utils.h"
#include "discrete_reward.h"

namespace rlly
{
namespace env
{

GridWorld::GridWorld(int _nrows, int _ncols, double fail_p /* = 0 */, double reward_smoothness /* = 0 */, double reward_sigma /* = 0 */)
{
    nrows = _nrows;
    ncols = _ncols;
    assert(nrows > 1 && "Invalid number of rows");
    assert(ncols > 1 && "Invalid number of columns");
    assert(reward_smoothness >= 0);
    assert(fail_p >= 0.0 && fail_p <= 1.0);

    // Number of states and actions
    int S = ncols*nrows;
    int A = 4;

    // Terminal state
    std::vector<double> goal_coord = { (double) nrows - 1, (double) ncols - 1};
    std::vector<int> _terminal_states = {S - 1};

    // Initialize vectors
    utils::vec::vec_3d _rewards = utils::vec::get_zeros_3d(S, A, S);
    utils::vec::vec_3d _transitions = utils::vec::get_zeros_3d(S, A, S);

    // Build maps between coordinates and indices
    int index = 0;
    for(int rr = 0; rr < nrows; rr++)
    {
        for(int cc = 0; cc < ncols; cc++)
        {
            std::vector<int> coord = {rr, cc};
            coord2index[coord] = index;
            index2coord[index] = coord; 
            index++;
        }
    }

    // Build rewards
    for(int jj = 0; jj < S; jj++)
    {
        std::vector<int>& next_state_coord = index2coord[jj];
        double squared_distance = std::pow( (1.0*next_state_coord[0]-1.0*goal_coord[0])/(nrows-1) , 2)
                                    + std::pow( (1.0*next_state_coord[1]-1.0*goal_coord[1])/(ncols-1), 2);
        double reward = 0;
        if (reward_smoothness > 0)
        {
            reward = std::exp( -squared_distance/ (2*std::pow(reward_smoothness, 2))  );
        }
        else 
        {
            reward = 1.0*(squared_distance == 0);
        }

        for(int ii = 0; ii < S; ii++)
        {
            for(int aa = 0; aa < A; aa++)
            {
                // reward depends on the distance between the next state and the goal state.
                _rewards[ii][aa][jj] = reward;
            }
        }
    }

    // Build transitions
    for(int ii = 0; ii < S; ii++)
    {
        std::vector<int>& state_coord = index2coord[ii];
        for(int aa = 0; aa < A; aa++)
        {
            // Coordinates of the next state
            std::vector<int> next_state_coord = get_neighbor(state_coord, aa);
            int next_state_index = coord2index[next_state_coord];
            _transitions[ii][aa][next_state_index] = 1.0;

            /*
                Handle the failure case.

                With probability fail_p, go to another neighbor!
            */
            if (fail_p > 0)
            {
                for(int bb = 0; bb < A; bb++)
                {
                    if (bb == aa) continue; 
                    std::vector<int> perturbed_next_state_coord = get_neighbor(state_coord, bb);
                    int perturbed_next_state_index = coord2index[perturbed_next_state_coord];
                    _transitions[ii][aa][next_state_index] -= fail_p/4.0;
                    _transitions[ii][aa][perturbed_next_state_index] += fail_p/4.0;
                }  
            }             
        }
    }
    // Initialize base class (FiniteMDP)
    if (reward_sigma == 0)
        set_params(_rewards, _transitions, _terminal_states);
    else
    {
        std::vector<double> noise_params;
        noise_params.push_back(reward_sigma);
        DiscreteReward _reward_function(_rewards, "gaussian", noise_params);
        set_params(_reward_function, _transitions, _terminal_states);
    }
        
    id = "GridWorld";

    // 2D rendering parameters
    refresh_interval_for_render2d = 1000; // 1 second between frames
    clipping_area_for_render2d[0] = 0.0;
    clipping_area_for_render2d[1] = 1.0*ncols;
    clipping_area_for_render2d[2] = 0.0;
    clipping_area_for_render2d[3] = 1.0*nrows;
}

std::vector<int> GridWorld::get_neighbor(std::vector<int> state_coord, int action)
{
    int neighbor_row = state_coord[0];
    int neighbor_col = state_coord[1];      
    switch(action) 
    {
        // Left
        case 0:
            neighbor_col = std::max(0, state_coord[1] - 1);
            break;
        // Right
        case 1:
            neighbor_col = std::min(ncols-1, state_coord[1] + 1);
            break;
        // Up
        case 2:
            neighbor_row = std::max(0, state_coord[0]-1);
            break;
            // Down
        case 3:
            neighbor_row = std::min(nrows-1, state_coord[0]+1);
            break;
    }
    std::vector<int> neighbor_coord = {neighbor_row, neighbor_col};
    return neighbor_coord;
}

void GridWorld::render()
{
    // std::cout<< "GridWorld" << std::endl;
    for(int ii = 0; ii < ncols; ii ++) std::cout<<"----";
    std::cout << std::endl;
    for (auto const& cell : index2coord) // key = cell.first, value = cell.second
    {
        std::string cell_str = "";
        
        // If state index (cell.first) is in terminal states
        if (std::find(terminal_states.begin(), terminal_states.end(), cell.first) != terminal_states.end())
            cell_str = " x  ";
        
        // If current state
        else if (cell.first == state)
            cell_str = " A  ";
        
        // 
        else 
            cell_str = " o  ";
        
        // Display
        std::cout << cell_str;
        if (cell.second[1] == ncols - 1) 
            std::cout << std::endl;
    }
    for(int ii = 0; ii < ncols; ii ++) std::cout<<"----";
    std::cout << std::endl;
}

void GridWorld::render_values(std::vector<double> values)
{
    for(int ii = 0; ii < ncols; ii ++) std::cout<<"------";
    std::cout << std::endl;
    for (auto const& cell : index2coord) // key = cell.first, value = cell.second
    {
        // Round value
        double value = values[cell.first];
        int ivalue = (int) (100*value);
        value = ivalue/100.0;
        std::cout << std::setw (6)<< value;   
        if (cell.second[1] == ncols - 1) 
            std::cout << std::endl;
    }
    for(int ii = 0; ii < ncols; ii ++) std::cout<<"------";
    std::cout << std::endl;       
}

void GridWorld::set_state(int state)
{
    this->state = state;
}

StepResult<int> GridWorld::step(int action)
{
    // for rendering
    if (rendering_enabled) append_state_for_rendering(state);
    
    // Sample next state
    int next_state = randgen.choice(transitions[state][action]);
    double reward = reward_function.sample(state, action, next_state, randgen); 
    bool done = is_terminal(next_state);
    StepResult<int> step_result(next_state, reward, done);
    state = step_result.next_state;
    return step_result;
}

std::unique_ptr<FiniteEnv> GridWorld::clone() const
{
    return std::make_unique<GridWorld>(*this);
}

utils::render::Scene2D GridWorld::get_scene_for_render2d(int state_var)
{
    utils::render::Scene2D scene; 
    utils::render::Geometric2D agent_shape;
    agent_shape.type = "GL_QUADS";
    agent_shape.set_color(0.0, 0.0, 0.5);
    std::vector<int>& state_coord = index2coord[state_var];
    
    // Getting (x, y) representation of the state
    float x_delta = 1.0;
    float y_delta = 1.0;

    float x = state_coord[1]*1.0;  
    float y = state_coord[0]*1.0;  
    x = x + x_delta/2.0;  // centering
    y = y + y_delta/2.0;  // centering

    // 
    float x_size = x_delta/4.0;
    float y_size = y_delta/4.0;

    agent_shape.add_vertex( x - x_size, y - y_size );
    agent_shape.add_vertex( x + x_size, y - y_size );
    agent_shape.add_vertex( x + x_size, y + y_size );
    agent_shape.add_vertex( x - x_size, y + y_size );

    scene.add_shape(agent_shape);
    return scene;
}

utils::render::Scene2D GridWorld::get_background_for_render2d()
{
    utils::render::Scene2D scene; 

    // Getting (x, y) representation of the state
    float x_size = 1.0;
    float y_size = 1.0;

    bool color = true;
    for(int cc = 0; cc < ncols; cc++)
    {
        for(int rr = 0; rr < nrows; rr++)
        {
            utils::render::Geometric2D shape;
            shape.type = "GL_QUADS";
            if ((rr+color) % 2 == 0) shape.set_color(0.35, 0.35, 0.35);
            else shape.set_color(0.5, 0.5, 0.5);
            if ( rr == nrows - 1  && cc == ncols - 1 ) shape.set_color(0.0, 0.5, 0.0);
            float x = 1.0*cc;
            float y = 1.0*rr;
            shape.add_vertex(x, y);
            shape.add_vertex(x+x_size, y);
            shape.add_vertex(x+x_size, y+y_size);
            shape.add_vertex(x, y+y_size);
            scene.add_shape(shape);
        }
        color = !color;
    }
    return scene;
}

} // namespace env 
} // namespace rlly
