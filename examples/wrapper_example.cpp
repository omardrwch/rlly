/*
    To run this example:
    $ bash scripts/compile.sh wrapper_example && ./build/examples/wrapper_example
*/

#include <iostream>
#include "env.h"
#include "render.h"
#include "wrappers.h"
#include "space.h"

typedef rlly::wrappers::ContinuousStateEnvWrapper ContinuousStateWrapper;
typedef rlly::env::ContinuousStateEnv ContinuousStateEnv;

class WrapperExample: public ContinuousStateWrapper
{
private:
    /* data */
public:
    WrapperExample(ContinuousStateEnv& env);
    ~WrapperExample(){};

    // override step function
    rlly::env::StepResult<std::vector<double>> step(int action) override;
};

WrapperExample::WrapperExample(ContinuousStateEnv& env): ContinuousStateWrapper(env)
{
    // do something
}

rlly::env::StepResult<std::vector<double>> WrapperExample::step(int action)
{
    auto step_result = (*p_env).step(action);
    // do something
    step_result.next_state[0] = 0; // setting x always to 0, for example
    return step_result;
}


int main()
{
    rlly::env::CartPole cartpole;

    // Wrapper
    WrapperExample env(cartpole);

    std::vector<double> state = env.reset();
    rlly::utils::vec::vec_2d states;

    int horizon = 200;
    for(int ii = 0; ii < horizon; ii++)
    {
        // auto action = env.action_space.sample();
        auto action = env.action_space.sample();
        auto step_result = env.step(action);
        states.push_back(step_result.next_state);
        std::cout << "state = "; rlly::utils::vec::printvec(step_result.next_state);
        std::cout << "reward = " << step_result.reward << std::endl;
        if (step_result.done) break;
    }

    // rlly::render::render_env(states, env);

    return 0;
}