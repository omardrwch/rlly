#ifndef __RLLY_GRIDWORLD_H__
#define __RLLY_GRIDWORLD_H__

/**
 * @file
 * @brief Define a simple and finite grid world. No walls!
 */

#include <map>
#include <vector>
#include "finitemdp.h"
#include "utils.h"

namespace rlly
{
namespace env
{
/**
 * Define a GridWorld environment: a nrows x ncols grid in which an agent can take 4 actions:
 * 'left', 'right', 'up' and 'down' 
 * 
 * @details
 *   Actions:
 *           0: left    (col -> col - 1)
 *           1: right   (col -> col + 1)
 *           2: up      (row -> row - 1)
 *           3: down    (row -> row + 1)
 * 
 *   Fail probability:
 * 
 *      With probability fail_p, a random action will be taken instead of the chosen action. Note that, even in
 *      the case of failure, the chosen action can be chosen by chance.
 */
class GridWorld: public FiniteMDP
{
public:
    /**
     * Number of rows.
     */
    int nrows;
    /**
     * Number of columns.
     */ 
    int ncols;

    /**
     * Map state indices to 2d coordinates
     */
    std::map<int, std::vector<int>> index2coord;

    /**
     * Map 2d coordinates to state indices
     */
    std::map<std::vector<int>, int> coord2index;

    /**
     * Get coordinates of next state given the coordinates of a state and an action
     */
    std::vector<int> get_neighbor(std::vector<int> state_coord, int action);

    /**
     * Render (ASCII)
     */ 
    void render();

    /**
     * Visualize values on the grid
     * @param values vector containing values to be shown on the grid (e.g., value functions)
     */
    void render_values(std::vector<double> values);


    /**
     * Clone 
     */
    std::unique_ptr<FiniteEnv> clone() const override;


    /**
     * Generate 2D representation (Scene) of a given state.
     */
    utils::render::Scene get_scene_for_render2d(int state_var) override;

    /**
     * Background for rendering
     */
    utils::render::Scene get_background_for_render2d() override;

private:
    /* data */

protected:
    /**
     * Default constructor
     */
    GridWorld();

public:
    /**
     * @param _nrows number of rows
     * @param _ncols number of columns
     * @param fail_p failure probability (default = 0)
     * @param reward_smoothness reward parameter. default = 0, mean_reward[s, a, s'] = exp(-(distance(s', goal_state)/reward_smoothness)^2  )
     * @param reward_sigma standard deviation of the reward noise. reward(s, a, s') = mean_reward(s, a, s') + reward_sigma*standard_gaussian_noise
     */ 
    GridWorld(int _nrows, int _ncols, double fail_p = 0, double reward_smoothness = 0, double reward_sigma = 0);
    ~GridWorld(){};
};   
} // namespace env
} // namespace rlly

#endif
