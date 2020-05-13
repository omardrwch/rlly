#ifndef __RLLY_CHASING_BLOBS_H__
#define __RLLY_CHASING_BLOBS_H__

#include <vector>
#include "env_typedefs.h"
#include "utils.h"

/**
 * @file 
 * @brief Contains a class for the ChasingBlobs environment.
 */

namespace rlly
{
namespace env
{

/**
 * @brief ChasingBlobs environment with states in [0, 1]^2 and 4 actions 
 * @details 
 *      The agent starts at (start_x, start_y) and, in each state, it can take for actions (0 to 3) representing a
 *      displacement of (-d, 0), (d, 0), (0, -d) and (0, d), respectively.
 *          
 *      The immediate reward received in each state s = (s_x, s_y) is, for any action a,
 *          r(s, a) =  \sum_i c_i *  exp( - ((s_x-blob_i_x)^2 + (s_y-blob_i_y)^2)/(2*reward_smoothness^2)  )
 * 
 *      Every <period> episodes there is a change in the blob configuration.
 *   
 */
class ChasingBlobs: public ContinuousStateEnv, public rlly::utils::render::RenderInterface2D<std::vector<double>>
{
private:
    // Coordinates of start position
    double start_x = 0.5;
    double start_y = 0.5;

    // Index of the current configuration (from 0 to 3)
    int current_configuration = 0;

    // Number of configurations
    int n_configurations = 4;

    // Blob reward multipliers in each configuration
    std::vector<std::vector<double>> reward_multipliers = { 
                                                            {0.25, 0.00, 0.00, 0.00},
                                                            {0.25, 0.50, 0.00, 0.00},
                                                            {0.25, 0.50, 0.75, 0.00},
                                                            {0.25, 0.50, 0.75, 1.00}
                                                          };


    // Coordinates of the goal position in each configuration
    std::vector<double> blob_x_vec = {0.9, 0.9, 0.1, 0.1};
    std::vector<double> blob_y_vec = {0.9, 0.1, 0.1, 0.9};

    // Action displacement
    double displacement = 0.1;

    // Reward smoothness
    double reward_smoothness = 0.05;

    // Standard dev of reward noise (gaussian)
    double reward_noise_stdev = 0.01;

    // Standard dev of transition noise (gaussian) 
    double transition_noise_stdev = 0.01;

    // Clip to domain
    void clip_to_domain(double &xx, double &yy);

public:
    ChasingBlobs(int _period);
    ~ChasingBlobs(){};

    // period of changes in the environment
    int period;

    // current episode, increased every time reset() is called
    int current_episode = -1;

    std::unique_ptr<ContinuousStateEnv> clone() const override;
    std::vector<double> reset() override;
    env::StepResult<std::vector<double>> step(int action) override;

    utils::render::Scene2D get_scene_for_render2d(std::vector<double> state_var) override;    
    utils::render::Scene2D get_background_for_render2d();

};

}  // namespace env
}  // namespace rlly


#endif