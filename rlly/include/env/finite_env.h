#ifndef __FINITE_ENV_H__
#define __FINITE_ENV_H__

#include <vector>
#include "abstractenv.h"
#include "space.h"

/**
 * @file
 * @brief Base class for  with finite states and finite actions.
 */


namespace rlly
{
namespace env
{

class FiniteEnv: public Env<int, int>
{
private:
    /* data */
public:
    FiniteEnv()
    {
        p_observation_space = &observation_space;
        p_action_space      = &action_space;
    };
    ~FiniteEnv(){};

    /**
     * Observation space
     */
    spaces::Discrete observation_space;

    /**
     * Action space
     */
    spaces::Discrete action_space;
};

}  // namespace env
}  // namespace rlly


#endif