#ifndef __RLLY_LQR_H__
#define __RLLY_LQR_H__

#include <vector>
#include "env_typedefs.h"
#include "utils.h"

namespace rlly
{
namespace env
{

/**
 * @brief Base class for LQR environments
 * @details Defined by the matrices A, B, Q and R such that:
 * 
 *  Transitions: x_{k+1} =  A x_k + B u_k + fixed_disturb
 *  Cost       : c_k     = x_k' Q x_k + u_k' R u_k
 * 
 *  where x_k and u_k are the state and the action at time k, respectively.
 */
class LQR: public ContinuousEnv
{
public:
    LQR(){};
    ~LQR(){};
    utils::vec::vec_2d A;
    utils::vec::vec_2d B;
    utils::vec::vec_2d Q;
    utils::vec::vec_2d R;
};


} // namespace env
} // namespace rlly

#endif