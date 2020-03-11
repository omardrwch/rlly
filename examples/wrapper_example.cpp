/*
    To run this example:
    $ bash scripts/compile.sh wrapper_example && ./build/examples/wrapper_example
*/

#include <iostream>
#include "env.h"
#include "render.h"
#include "wrappers.h"
#include "space.h"

using namespace rlly::wrappers;

template <typename EnvType>
class WrapperExample: public rlly::wrappers::IsomorphicWrapper<EnvType>
{
private:
    /* data */
public:
    WrapperExample(EnvType& env);
    ~WrapperExample(){};

    // override step function
    rlly::env::StepResult<std::vector<double>> step(int action) override;
};

template <typename EnvType>
WrapperExample<EnvType>::WrapperExample(EnvType& env): IsomorphicWrapper<EnvType>(env)
{
    // do something
}

template <typename EnvType>
rlly::env::StepResult<std::vector<double>> WrapperExample<EnvType>::step(int action)
{
    auto step_result = this->p_env.step(action);
    // do something
    step_result.next_state[0] = 0; // setting x always to 0, for example
    return step_result;
}


int main()
{
    rlly::env::CartPole cartpole;
    cartpole.enable_rendering();

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
    // Rendering is done with the original enrivonment
    rlly::render::render_env(cartpole);
    return 0;
}