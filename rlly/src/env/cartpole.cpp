#include "cartpole.h"

namespace rlly
{
namespace env
{

CartPole::CartPole()
{
    // Set seed
    int _seed = std::rand();
    set_seed(_seed);

    // Allocate memory for state
    for(int ii = 0; ii < 4; ii ++) state.push_back(0.0);

    // observation and action spaces
    double inf = std::numeric_limits<double>::infinity();
    double angle_lim_rad = 2.0*theta_threshold_radians;
    std::vector<double> _low = {-4.8, -inf, -angle_lim_rad, -inf};
    std::vector<double> _high = {4.8,  inf, angle_lim_rad,  inf};
    observation_space.set_bounds(_low, _high);
    action_space.set_n(2);


    // id
    id = "CartPole";

    // 2D rendering is enabled for CartPole
    rendering2d_enabled = true;

    clipping_area_for_render2d[0] = -2.4;
    clipping_area_for_render2d[1] =  2.4;
    clipping_area_for_render2d[2] = -0.5;
    clipping_area_for_render2d[3] =  1.5;
}


StepResult<std::vector<double>> CartPole::step(int action)
{
    // get state variables
    double x = state[0]; 
    double x_dot = state[1];
    double theta = state[2];
    double theta_dot = state[3];

    // compute force
    double force = 0;
    if(action == 1) force =  force_magnitude;
    else            force = -force_magnitude;

    // quantities used to compute next state
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);
    double temp = (force + pole_mass_times_length * theta_dot * theta_dot * sin_theta) / total_mass;

    double thetaacc = (gravity * sin_theta - cos_theta* temp) / (half_pole_length * (4.0/3.0 - mass_pole * cos_theta * cos_theta / total_mass));
    double xacc  = temp - pole_mass_times_length * thetaacc * cos_theta / total_mass;

    // compute next state
    x         = x         + delta_t*x_dot;
    x_dot     = x_dot     + delta_t*xacc;
    theta     = theta     + delta_t*theta_dot;
    theta_dot = theta_dot + delta_t*thetaacc;
    
    // store next state
    state[0] = x; 
    state[1] = x_dot;
    state[2] = theta;
    state[3] = theta_dot;

    // check if done
    bool done = (x < -x_threshold) || (x> x_threshold) ||
                (theta < -theta_threshold_radians) || (theta > theta_threshold_radians);
    
    // compute reward
    double reward = 0.0;
    if (!done) reward = 1.0;
    else if (steps_beyond_done == -1)
    {
        // pole just fell
        steps_beyond_done = 0;
        reward = 1.0;
    }
    else
    {
        if (steps_beyond_done == 0)
            std::cerr << "Warning (CartPole): undefined behaviour: calling step() after done = True." << std::endl;
        steps_beyond_done += 1;
        reward = 0.0;
    }

    // return
    StepResult<std::vector<double>> step_result(state, reward, done);
    return step_result;
}

std::vector<double> CartPole::reset()
{
    for(int ii = 0; ii < 4; ii++)
    {
        state[ii] = randgen.sample_real_uniform(-0.05, 0.05);
    }
    return state; 
}

std::unique_ptr<Env<std::vector<double>, int>> CartPole::clone() const
{
    return std::make_unique<CartPole>(*this);
}

utils::render::Scene CartPole::get_scene_for_render2d(std::vector<double> state_var)
{
    // Compute cart and pole positions
    float pole_length = 2.0*half_pole_length; 
    float theta = state_var[2];
    float cart_y = 0;
    float cart_x = state_var[0];

    float pole_x0 = cart_x;
    float pole_y0 = cart_y;
    float pole_x1 = cart_x + pole_length*std::sin(theta);
    float pole_y1 = cart_y + pole_length*std::cos(theta);

    std::vector<float> pole_vec, u_vec;
    pole_vec.push_back(pole_x1-pole_x0);
    pole_vec.push_back(pole_y1-pole_y0);
    if (std::abs(pole_vec[0]) < 1e-4)
    {
        u_vec.push_back(-1); u_vec.push_back(0);
    }
    else
    {
        u_vec.push_back(-pole_vec[1]/pole_vec[0]);
        u_vec.push_back(1.0);
    }
    float norm = std::sqrt( u_vec[0]*u_vec[0]
                           +u_vec[1]*u_vec[1]);
    u_vec[0] /= norm;
    u_vec[1] /= norm;

    u_vec[0] /= 50.0;
    u_vec[1] /= 50.0;



    utils::render::Scene cartpole_scene;
    utils::render::Geometric2D cart, pole;
    cart.type = "GL_QUADS";
    cart.set_color(0.0f, 0.0f, 0.0f);
    
    float size = 0.075;
    cart.add_vertex(cart_x - size, cart_y - size);
    cart.add_vertex(cart_x + size, cart_y - size);
    cart.add_vertex(cart_x + size, cart_y + size);
    cart.add_vertex(cart_x - size, cart_y + size);

    pole.type = "GL_QUADS";
    pole.add_vertex(pole_x0 + u_vec[0], pole_y0 + u_vec[1]);
    pole.add_vertex(pole_x0 - u_vec[0], pole_y0 - u_vec[1]);
    pole.add_vertex(pole_x1 - u_vec[0], pole_y1 - u_vec[1]);
    pole.add_vertex(pole_x1 + u_vec[0], pole_y1 + u_vec[1]);
    pole.set_color(0.4f, 0.0f, 0.0f);


    cartpole_scene.add_shape(pole);
    cartpole_scene.add_shape(cart);
    return cartpole_scene;
}

utils::render::Scene CartPole::get_background_for_render2d()
{
    utils::render::Scene background;
    utils::render::Geometric2D base;
    base.type = "GL_QUADS";
    base.set_color(0.6, 0.3, 0.0);
    
    float y = 0;
    float size = 0.0125;
    base.add_vertex(-2.4, y - size);
    base.add_vertex(-2.4, y + size);
    base.add_vertex( 2.4, y + size);
    base.add_vertex( 2.4, y - size);

    background.add_shape(base);
    return background;
}


} // namespace env
} // namespace rlly
