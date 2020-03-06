#ifndef __RLLY_ABSTRACTMDP_H__
#define __RLLY_ABSTRACTMDP_H__

#include <string>
#include <random>
#include <iostream>
#include "space.h"
#include "stepresult.h"
#include "utils.h"

/**
 * @file
 * @brief Contains abstract class for reinforcement learning environments
 */


namespace rlly
{
namespace env
{

/**
 * @brief Abstract class for reinforcement learning environments
 */
template <typename S, typename A>
class Env
{
public:
    Env(/* args */) {};
    ~Env() {};

    /**
     * @brief Put environment in default state
     * @return Default state
     */
    virtual S reset()=0;

    /**
     * @brief Take a step in the MDP
     * @param action
     * @return An instance of mdp::StepResult containing the next state,
     * the reward and the done flag.
     */
    virtual StepResult<S> step(A action)=0;

    /**
     * Current state
     */
    S state;

    /**
     *  Environment identifier
     */
    std::string id;

    /**
     * Pointer to observation space
     */
    spaces::Space<S>* p_observation_space;

    /**
     * Pointer to action space
     */   
    spaces::Space<A>* p_action_space;

    /**
    * For random number generation
    */
    utils::rand::Random randgen;

    /**
     * Set the seed of randgen and seed of action space and observation space
     * The seed of randgen is set to _seed, the seed of action space is set to _seed+123
     * and the seed of observation space is set to _seed+456
     * Note: If _seed < 1,  we set _seed = std::rand()
     * @param _seed
     */
    void set_seed(int _seed)
    {
        if (_seed < 1) 
        {
            _seed = std::rand();
            // std::cout << _seed << std::endl;
        }

        randgen.set_seed(_seed);
        // seeds for spaces
        if ( p_observation_space != nullptr && p_action_space != nullptr) 
        { 
            (*p_observation_space).generator.seed(_seed+123);
            (*p_action_space).generator.seed(_seed+456);
        }
        else
        {
            std::cerr << "Warning (rlly::Env), trying to set seed of not initialized observation or action space." << std::endl;
        }
        
    }; 
    
    /*

        Methods and attributes used for graph rendering

    */

    /**
     * Set to true if the environment supports graph rendering.
     * To support graph rendering, the derived class must:
     *     - set graph_rendering_enabled to true
     *     - define the number of nodes n_nodes
     *     - implement the method get_nodes_for_graph_render()
     *     - implement the method get_edges_for_graph_render()
     */
    bool graph_rendering_enabled = false;

    /**
     * Number of nodes for graph rendering.
     * Set to -1 if not applicable
     */
    int graph_rendering_n_nodes = -1;
    
    /**
     * Return [(x_1, y_1), ..., (x_n, y_n)] representation of the state, where x_i and y_i are in [0, 1]
     * and n is the number of nodes representing the state
     */
    virtual std::vector<std::vector<float>> get_nodes_for_graph_render(S state_var) {return std::vector<std::vector<float>>();};
    virtual std::vector<float> get_edges_for_graph_render(){return std::vector<float>();};
}; 
}  // namespace env
}  // namespace rlly

#endif
