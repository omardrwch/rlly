#include "movingball.h"
#include <cmath>

namespace rlly
{
namespace env
{

MovingBall::MovingBall()
{
    // Set seed
    set_seed(-1);

    // observation and action spaces
    std::vector<double> _state_low  = {-1.0, -1.0, -250.0, -250.0};
    std::vector<double> _state_high = { 1.0,  1.0,  250.0,  250.0};
    observation_space.set_bounds(_state_low, _state_high);

    std::vector<double> _action_low  = {-1.5, -1.5};
    std::vector<double> _action_high = { 1.5,  1.5};
    action_space.set_bounds(_action_low, _action_high);

    // Initialize LQR matrices
    A = utils::vec::get_zeros_2d(4, 4);
    B = utils::vec::get_zeros_2d(4, 2);
    Q = utils::vec::get_zeros_2d(4, 4);
    R = utils::vec::get_zeros_2d(2, 2);

    A[0][0] = 1.0; A[1][1] = 1.0; A[2][2] = 1.0; A[3][3] = 1.0;
    A[0][2] = delta_t;  A[1][3] = delta_t;
    B[2][0] = delta_t/ball_mass; B[3][1] = delta_t/ball_mass;
    
    Q[0][0] = 1.0; Q[1][1] = 1.0; Q[2][2] = 1.0/100.0; Q[3][3] = 1.0/100.0;
    R[0][0] = 0.25; R[1][1] = 0.25;


    // Initialize state
    for (int ii = 0; ii < 4; ii++) state.push_back(0);
    reset();

    id = "MovingBall";
}


std::vector<double> MovingBall::reset()
{
    state[0] =  randgen.sample_real_uniform(-1.0, 1.0);
    state[1] =  randgen.sample_real_uniform(-1.0, 1.0);
    state[2] =  0.0;
    state[3] =  0.0;
    return state;
}

StepResult<std::vector<double>> MovingBall::step(std::vector<double> action)
{
    // for rendering
    if (rendering_enabled) append_state_for_rendering(state);
    //

    if (!action_space.contains(action)) throw;
    double x  = state[0];
    double y  = state[1];
    double vx = state[2];
    double vy = state[3];

    double Fx = action[0];
    double Fy = action[1];

    // compute reward 
    double cost = x*x + y*y + (vx*vx + vy*vy)/(100.0) + 0.25*(Fx*Fx + Fy*Fy);
    double reward = -1.0*cost;
    bool   done = false;

    // compute next state
    x  = x + delta_t*vx; 
    y  = y + delta_t*vy;
    vx = vx + (delta_t/ball_mass)*Fx; 
    vy = vy + (delta_t/ball_mass)*Fy;
    
    // clipping
    x  = utils::clamp( x  , -1.0,   1.0);
    y  = utils::clamp( y,   -1.0,   1.0);
    vx = utils::clamp(vx, -250.0, 250.0);
    vy = utils::clamp(vy, -250.0, 250.0);

    // if a wall is hit, set velocity to zero and the episode is over
    if (x == -1.0 || x == 1.0 || y == -1.0 || y == 1.0)
    {
        vx = 0; vy = 0;
        done = true;
    }

    // update state 
    state[0] = x;  state[1] = y;
    state[2] = vx; state[3] = vy; 
    StepResult<std::vector<double>> step_result(state, reward, done);
    return step_result;
}


std::unique_ptr<ContinuousEnv> MovingBall::clone() const
{
    return std::make_unique<MovingBall>(*this);
}


utils::render::Scene2D MovingBall::get_scene_for_render2d(std::vector<double> state_var)
{
    float x = state_var[0];
    float y = state_var[1];
    
    utils::render::Scene2D ball_scene;
    utils::render::Geometric2D ball;
    ball.type = "GL_POLYGON";
    ball.set_color(0.5f, 0.0f, 0.5f);
    
    float radius = 0.05;
    int n_points = 25;
    for(int ii = 0; ii < n_points; ii++)
    {
        float angle = 2.0*3.141592*ii/n_points;
        float xcirc = x + radius*std::cos(angle);
        float ycirc = y + radius*std::sin(angle);
        ball.add_vertex(xcirc, ycirc);    
    }

    ball_scene.add_shape(ball);
    return ball_scene;
}

utils::render::Scene2D MovingBall::get_background_for_render2d()
{
    utils::render::Scene2D background;
    utils::render::Geometric2D goal;
    goal.type = "GL_LINE_STRIP";
    float size = 0.075;
    float x = 0.0;
    float y = 0.0;
    goal.set_color(0.0f, 0.5f, 0.0f);
    goal.add_vertex(x - size, y - size);
    goal.add_vertex(x + size, y - size);
    goal.add_vertex(x + size, y + size);
    goal.add_vertex(x - size, y + size);
    goal.add_vertex(x - size, y - size);
    background.add_shape(goal);
    return background;
}


} // namespace env
} // namespace rlly
