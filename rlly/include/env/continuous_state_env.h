#ifndef __RLLY_CONTINUOUS_STATE_ENV_H__
#define __RLLY_CONTINUOUS_STATE_ENV_H__

#include <vector>
#include "abstractenv.h"
#include "space.h"

/**
 * @file
 * @brief Base class for continuous-state environments with finite actions.
 */


namespace rlly
{
namespace env
{

class ContinuousStateEnv: public Env<std::vector<double>, int>
{
private:
    /* data */
public:
    ContinuousStateEnv()
    {
        p_observation_space = &observation_space;
        p_action_space      = &action_space;
    };
    ~ContinuousStateEnv(){};

    /**
     * Observation space
     */
    spaces::Box observation_space;

    /**
     * Action space
     */
    spaces::Discrete action_space;
};

}  // namespace env
}  // namespace rlly


#endif