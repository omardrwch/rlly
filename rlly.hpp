#include <algorithm>
#include <assert.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <random>
#include <stdlib.h>
#include <string>
#include <vector>
#ifndef __RLLY_ENVS_H__
#define __RLLY_ENVS_H__ 
#ifndef __RLLY_ABSTRACT_SPACE_H__
#define __RLLY_ABSTRACT_SPACE_H__

/**
 * @file
 * @brief Class for definining observation and action spaces. 
 */

namespace rlly
{
namespace spaces
{

/**
 * @brief Possible space names.
 */
enum spc_name {undefined, discrete, box}; 

/**
 * @brief Base class for observation and action spaces.
 * 
 */
template <typename T> 
class Space
{
public:
    Space(/* args */){};
    ~Space(){};
    /**
     * @brief Sample a value of the space with a uniform distribution
     */
    virtual T sample() = 0;

    /**
     * @brief Returns true if x belongs to the space, and false otherwise.
     * @param x 
     */
    virtual bool contains(T x) = 0;

    /**
     * Name of the space (discrete, box, etc.)
     */
    spc_name name = undefined; 

    /**
     * Random number generator
     */
    std::mt19937 generator;

    /**
     * Size of the space (-1 for infinity or undefined )
     */
    int n = -1;
};

} // namespace spaces
} // namespace rlly

#endif
#ifndef __RLLY_BOX_SPACE_H__
#define __RLLY_BOX_SPACE_H__

/**
 * @file
 * @brief Class for definining box observation and action spaces.
 */

namespace rlly
{
namespace spaces
{

/**
 * @brief Class for box spaces in R^n
 */
class Box: public Space<std::vector<double>>
{
public:

    /**
     * @brief Default constructor
     */
    Box();

    /**
     * @param _low: array contaning the lower bounds of the box
     * @param _high: array containing the upper bounds of the box
     * @param _seed: seed for random number generation (default = 42)
     */
    Box(std::vector<double> _low, std::vector<double> _high, unsigned _seed = 42);
    ~Box(){};

    // Methods
    /**
     * @brief Define the lower and upper bounds of the box
     * @param _low: array contaning the lower bounds of the box
     * @param _high: array containing the upper bounds of the box
     */ 
    void set_bounds(std::vector<double> _low, std::vector<double> _high);

    // Methods of base class
    std::vector<double> sample();
    bool contains(std::vector<double> x);

    // Attributes

    /**
     * Size of the fox
     */
    unsigned int size; 

    /**
     * lower bounds of the box
     */
    std::vector<double> low;

    /**
     * upper bounds of the box
     */
    std::vector<double> high;
};

}
}

#endif
#ifndef __RLLY_DISCRETE_SPACE_H__
#define __RLLY_DISCRETE_SPACE_H__

/**
 * @file
 * @brief Class for definining discrete observation and action spaces.
 */

namespace rlly
{
namespace spaces
{

/**
 * @brief Class for discrete spaces.
 * Set of possible values = {0, ..., n-1}
 */
class Discrete: public Space<int>
{
public:

    /**
     * @brief Default constructor
     */
    Discrete();

    /**
     * @param _n: Value of n
     * @param _seed: seed for random number generation (default = 42)
     */
    Discrete(int _n, unsigned _seed = 42);
    ~Discrete(){};

    // Methods

    /**
     * @brief Set the value of n, that defines the discrete space {0, ..., n-1}
     */
    void set_n(int _n);

    // Methods of base class
    int sample();
    bool contains(int x);
};

}
}

#endif
#ifndef __RLLY_SPACE_H__
#define __RLLY_SPACE_H__

/**
 * @file 
 * All space headers.
 */

namespace rlly
{
/**
 * @brief Definitions for observation and action spaces.
 */
namespace spaces{}
}

#endif
#ifndef __RLLY_STEP_RESULT_H__
#define __RLLY_STEP_RESULT_H__

/**
 * @file
 * @brief Contains class for storing the results of env.step()
 */

namespace rlly
{
namespace env
{
    
/**
 * @brief Class to represent an object returned by Env::step()
 */
template<typename S>
class StepResult
{
public:
    StepResult(); // default constructor
    /**
     * @brief Initialize object with data
     * @param _next_state
     * @param _reward
     * @param _done
     */
    StepResult(S _next_state, double _reward, bool _done);
    ~StepResult() {};

    /**
     * @brief Next state
     */
    S next_state;

    /**
     * @brief Value of the reward
     */
    double reward;

    /**
     * @brief Flag that is true if a terminal state is reached
     */
    bool done;
};

template<typename S>
StepResult<S>::StepResult()
{
    next_state = -1;
    reward = 0;
    done = false;
}

template<typename S>
StepResult<S>::StepResult(S _next_state, double _reward, bool _done)
{
    next_state = _next_state;
    reward = _reward;
    done = _done;
}

}  // namespace env
}  // namespace rlly

#endif
#ifndef __RLLY_VECTOR_OP__
#define __RLLY_VECTOR_OP__

/**
 * @file
 * @brief Common vector operations.
 */

namespace rlly
{
namespace utils
{

/**
 * Utils for common vector operations.
 */
namespace vec
{
    /**
     * @brief Type for 2d vector (double)
     */
    typedef std::vector<std::vector<double>> vec_2d;

    /**
     * @brief Type for 3d vector (double)
     */
    typedef std::vector<std::vector<std::vector<double>>> vec_3d;

    /**
     * @brief Type for 4d vector (double)
     */
    typedef std::vector<std::vector<std::vector<std::vector<double>>>> vec_4d;

    /**
     * @brief Type for 2d vector (integer)
     */
    typedef std::vector<std::vector<int>> ivec_2d;

    /**
     * @brief Type for 3d vector (integer)
     */
    typedef std::vector<std::vector<std::vector<int>>> ivec_3d;

    /**
     * @brief Type for 4d vector (integer)
     */
    typedef std::vector<std::vector<std::vector<std::vector<int>>>> ivec_4d;

    /**
     * @brief Computes the mean of a vector.
     * @param vec
     * @return mean of vec
     */
    double mean(std::vector<double> vec);

    /**
     * @brief Computes the standard deviation of a vector.
     * @param vec
     * @return standard deviation of vec
     */
    double stdev(std::vector<double> vec);

    /**
     * @brief Computes the inner product between vec1 and vec2
     * @param vec1
     * @param vec2
     * @return inner product
     */
    double inner_prod(std::vector<double> vec1, std::vector<double> vec2);

    /**
     * @brief Print vector
     * @param vec
     */
    template <typename T>
    void printvec(std::vector<T> vec)
    {
        int n = vec.size();
        std::cout << "{";
        for(int i = 0; i < n; i++)
        {
            std::cout << vec[i];
            if (i < n-1){ std::cout << ", ";}
        }
        std::cout << "}" << std::endl;
    }

    /**
     * @brief Print vector of double
     */
    template void printvec<double>(std::vector<double>);
    /**
     * @brief Print vector of int
     */
    template void printvec<int>(std::vector<int>);

    /**
     * @brief Get 2d vector of doubles of dimensions (dim1, dim2) filled with zeros
     * @param dim1
     * @param dim2
     * @return vec_2d with dimensions (dim1, dim2)
     */
    vec_2d get_zeros_2d(int dim1, int dim2);

    /**
     * @brief Get 2d vector of integers of dimensions (dim1, dim2) filled with zeros
     * @param dim1
     * @param dim2
     * @return ivec_2d with dimensions (dim1, dim2)
     */
    ivec_2d get_zeros_i2d(int dim1, int dim2);

    /**
     * @brief Get 3d vector of doubles of dimensions (dim1, dim2, dim3) filled with zeros
     * @param dim1
     * @param dim2
     * @param dim3
     * @return vec_3d with dimensions (dim1, dim2, dim3)
     */
    vec_3d get_zeros_3d(int dim1, int dim2, int dim3);

    /**
     * @brief Get 3d vector of integers of dimensions (dim1, dim2, dim3) filled with zeros
     * @param dim1
     * @param dim2
     * @param dim3
     * @return ivec_3d with dimensions (dim1, dim2, dim3)
     */
    ivec_3d get_zeros_i3d(int dim1, int dim2, int dim3);

    /**
     * @brief Get 4d vector of integers of dimensions (dim1, dim2, dim3, dim4) filled with zeros
     * @param dim1
     * @param dim2
     * @param dim3
     * @param dim4
     * @return ivec_4d with dimensions (dim1, dim2, dim3, dim4)
     */
    ivec_4d get_zeros_i4d(int dim1, int dim2, int dim3, int dim4);

    /**
     * @brief Get 4d vector of double of dimensions (dim1, dim2, dim3, dim4) filled with zeros
     * @param dim1
     * @param dim2
     * @param dim3
     * @param dim4
     * @return vec_4d with dimensions (dim1, dim2, dim3, dim4)
     */
    vec_4d get_zeros_4d(int dim1, int dim2, int dim3, int dim4);
}

} // namespace utils
} // namespace rlly
#endif
#ifndef __RLLY_RANDOM_H__
#define __RLLY_RANDOM_H__

/**
 * @file
 * @brief Contains class for random number generation.
 */

namespace rlly
{
namespace utils
{
    /**
     * Utils for random number generation.
     */
    namespace rand
    {
        /**
         * @brief Class for random number generation.
         */
        class Random
        {
        
        private:
            /**
             * Random number generator
             */
            std::mt19937 generator;
            /**
             * continuous uniform distribution in (0, 1)
             */ 
            std::uniform_real_distribution<double> real_unif_dist;       
            /**
             *  standard normal distribution
             */
            std::normal_distribution<double> gaussian_dist;
            /**
             * Seed for the std::mt19937 generator.
             */
            unsigned seed;

        public:
            /**
             * @brief Initializes object with given seed.
             * @param _seed
             */
            Random(unsigned _seed = 42);
            ~Random(){};

            /**
             * @brief Set seed for random number generator
             * @param _seed
             */
            void set_seed(unsigned _seed);

            /**
             * @brief Sample according to probability vector.
             * @details The parameter prob is passed by reference to avoid copying. It is not changed by the algorithm.
             * @param prob probability vector 
             * @param u (optional) sample from a real uniform distribution in (0, 1)
             * @return integer between 0 and prob.size()-1 according to 
             * the probabilities in prob.
             */
            int choice(std::vector<double>& prob, double u = -1);

            /**
             * @brief Sample from (continuous) uniform distribution in (a, b)
             * @param a 
             * @param b
             * @return sample
             */
            double sample_real_uniform(double a, double b);

            /**
             * @brief Sample from a gaussian distribution with mean mu and variance sigma^2 
             * @param mu mean
             * @param sigma standard deviation
             * @return sample
             */
            double sample_gaussian(double mu, double sigma);

            /**
             * @brief returns seed value
             */
            int get_seed();
        };     
    }
}  // namespace utils
}  // namespace rlly
#endif
#ifndef __RLLY_MISC_H__
#define __RLLY_MISC_H__

/**
 * @file 
 * Other utils.
 */

namespace rlly
{
namespace utils
{

    /**
     * @brief Clamp a value between an upper and lower bound.
     * @param v value to be clampled
     * @param lo lower bound
     * @param hi upper bound
     * @tparam T type of v, lo and hi
     */
    template<class T>
    constexpr const T& clamp( const T& v, const T& lo, const T& hi )
    {
        assert( !(hi < lo) );
        return (v < lo) ? lo : (hi < v) ? hi : v;
    }

    /**
     *  Map a value x in [x0, x1] linearly to the range [y1, y2]
     */
    double linear_map(double x, double x1, double x2, double y1, double y2);

    /**
     * @brief Binary search in a sorted vector with increasing values.
     * @details if vec = {x_0, x_1, ..., x_n}, with x_0 <= x_1 <= ... <= x_n,
     * returns the value i such that vec[i] <= val < vec[i+1]. If there is no
     * such value, returns -1.
     * @param val value to be searched
     * @param vec vector in which to search the interval where val is.
     * @param l   index where to start the search (default = 0)
     * @param r   index where to end the search (default = -1). If -1, it is set to vec.size()-1
     */
    int binary_search(double val, std::vector<double> vec, int l = 0, int r = -1);

    /**
     * @brief Binary search in d dimensions. Returns flat index. 
     * @param d_val vector to be searched
     * @param bins 2d array such that bins[i] represents the intervals where to search for d_val[i]. Represents
     * a set of hypercubes in R^d
     * @return flat index in column-major order corresponding to the hypercube where d_val lives
     */
    int binary_search_nd(std::vector<double> d_val, std::vector<std::vector<double>> bins);
}
}  

#endif
/**
 * @file
 * @brief Data structures used for rendering.
 */

#ifndef __RLLY_RENDER_DATA_H__
#define __RLLY_RENDER_DATA_H__

namespace rlly
{
namespace utils
{
/**
 * Rendering utils
 */
namespace render
{

/**
 * Data representing an OpenGL geometric primitive in 2D
 */ 
struct Geometric2D
{
    /**
     * Primitive type (GL_LINE_LOOP by defaut)
     * Possibilities:
     *      GL_POINTS
     *      GL_LINES
     *      GL_LINE_STRIP
     *      GL_LINE_LOOP
     *      GL_POLYGON
     *      GL_TRIANGLES
     *      GL_TRIANGLE_STRIP
     *      GL_TRIANGLE_FAN
     *      GL_QUADS
     *      GL_QUAD_STRIP
     */
    std::string type = "GL_LINE_LOOP";

    /**
     * vector with 3 elements, contaning the color of the shape
     * gray by default
     */
    std::vector<float> color = {0.25f, 0.25f, 0.25f};   

    /**
     * 2d vector of shape (n_vertices, 2)
     * vertices[i][j] = j-th cordinnate of vertex i
     */
    std::vector<std::vector<float>> vertices; 

    /**
     * Add vertex 
     */ 
    void add_vertex(std::vector<float> vertex) { vertices.push_back(vertex); }; 
    void add_vertex(float x, float y) 
    { 
        std::vector<float> vertex;
        vertex.push_back(x);
        vertex.push_back(y);
        vertices.push_back(vertex); 
    }; 

    /**
     * Set color
     */
    void set_color(float r, float g, float b)
    {
        color[0] = r; color[1] = g; color[2] = b; 
    };

};

/**
 * Data representing a scene, which is a vector of Geometric2D objects
 */
struct Scene2D
{
    /**
     * Vector of 2D shapes represeting the scene
     */
    std::vector<Geometric2D> shapes;

    /**
     * Include new shape
     */
    void add_shape(Geometric2D shape){ shapes.push_back(shape);};
};

struct Polygon2D
{
    /**
     * 2d vector of shape (n_vertices, 2)
     * vertices[i][j] = j-th cordinnate of vertex i
     */
    std::vector<std::vector<float>> vertices; 
    /**
     * vector with 3 elements, contaning the color of the polygon
     */
    std::vector<float> color;   
};

}
}
}

#endif
#ifndef __RLLY_RENDER_INTERFACE_H__
#define __RLLY_RENDER_INTERFACE_H__

namespace rlly
{
namespace utils
{
namespace render
{

template <typename S>
class RenderInterface2D
{
private:
    /* data */
public:
    RenderInterface2D(){};
    ~RenderInterface2D(){};

    /*

        Methods and attributes used for graph rendering

    */

    /**
     * Flag to say that rendering is enabled
     */
    bool rendering_enabled = false;

    /**
     * Rendering type
     */ 
    const std::string rendering_type = "2d";

    /**
     * Enable rendering
     */
    void enable_rendering() {rendering_enabled = true; };

    /**
     * Disable rendering
     */
    void disable_rendering() {rendering_enabled = false; };

    /**
     * Retuns a scene (list of shapes) representing the state
     * @param state_var
     */
    virtual utils::render::Scene2D get_scene_for_render2d(S state_var)=0;    
    
    /**
     * Retuns a scene (list of shapes) representing the background
     */
    virtual utils::render::Scene2D get_background_for_render2d(){return utils::render::Scene2D();};

    /**
     * List of states to be rendered
     */
    std::vector<S> state_history_for_rendering;

    /**
     * Clear rendering buffer 
     */
    void clear_render_buffer() { state_history_for_rendering.clear(); };

    /**
     * Add state to rendering buffer
     */ 
    void append_state_for_rendering(S state) {state_history_for_rendering.push_back(state); };

    /**
     *  Refresh interval of rendering (in milliseconds)
     */
    int refresh_interval_for_render2d = 50;

    /**
     * Clipping area for rendering (left, right, bottom, top). Default = {-1.0, 1.0, -1.0, 1.0}
     */
    std::vector<float> clipping_area_for_render2d = {-1.0, 1.0, -1.0, 1.0};
};

} // namespace render
} // namespace utils
} // namespace rlly

#endif
#ifndef __RLLY_UTILS_H__
#define __RLLY_UTILS_H__

/**
 * @file 
 * All utils headers.
 */

namespace rlly
{
/**
 * @brief Useful definitions (random number generation, vector operations etc.)
 */
namespace utils{}
}

#endif
#ifndef __RLLY_ABSTRACTMDP_H__
#define __RLLY_ABSTRACTMDP_H__

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
 * @tparam S_space type of state space  (e.g. spaces::Box, spaces::Discrete)
 * @tparam A_space type of action space (e.g. spaces::Box, spaces::Discrete)
 */
template <typename S_space, typename A_space>
class Env
{
public:
    Env() {};
    ~Env() {};

    /**
     * Observation space
     */
    S_space observation_space;

    /**
     * Action space
     */   
    A_space action_space;

    /**
     * type of state variables
     */
    using S_type = decltype(observation_space.sample());

    /**
     *  type of action variables
     */
    using A_type = decltype(action_space.sample());

    /**
     * @brief Put environment in default state
     * @return Default state
     */
    virtual S_type reset()=0;

    /**
     * @brief Take a step in the MDP
     * @param action
     * @return An instance of mdp::StepResult containing the next state,
     * the reward and the done flag.
     */
    virtual StepResult<S_type> step(A_type action)=0;

    /**
     *  Environment identifier
     */
    std::string id;

    /**
    * For random number generation
    */
    utils::rand::Random randgen;

    /**
     * Function to clone the environment
     */
    virtual std::unique_ptr<Env<S_space, A_space>> clone() const = 0;

    /**
     * Set the seed of randgen and seed of action space and observation space
     * The seed of randgen is set to _seed, the seed of action space is set to _seed+123
     * and the seed of observation space is set to _seed+456
     * Note: If _seed < 1,  we set _seed = std::rand()
     * @param _seed
     */
    virtual void set_seed(int _seed);
protected:
    /**
     * Current state
     */
    S_type state;

}; 

template <typename S_space, typename A_space>
void Env<S_space, A_space>::set_seed(int _seed)
{
    if (_seed < 1) 
    {
        _seed = std::rand();
    }
    randgen.set_seed(_seed);
    observation_space.generator.seed(_seed+123);
    action_space.generator.seed(_seed+456);
}; 

}  // namespace env
}  // namespace rlly

#endif
#ifndef __RLLY_ENV_TYPEDEFS_H__
#define __RLLY_ENV_TYPEDEFS_H__

/**
 * @file
 * @brief Useful type definitions
 */

namespace rlly
{
namespace env
{

/**
 * @brief Base class for environments with finite states and finite actions.
 */
typedef Env<spaces::Discrete, spaces::Discrete> FiniteEnv;

/**
 * @brief Base class for continuous-state environments with finite actions.
 */
typedef Env<spaces::Box, spaces::Discrete> ContinuousStateEnv;

/**
 * @brief Base class for continuous-state environments with continuous actions.
 */
typedef Env<spaces::Box, spaces::Box> ContinuousEnv;

/**
 * @brief Base class for discrete-state environments with continuous actions.
 */
typedef Env<spaces::Discrete, spaces::Box> ContinuousActionEnv;

}  // namespace env
}  // namespace rlly

#endif
#ifndef __RLLY_SQUAREWORLD_H__
#define __RLLY_SQUAREWORLD_H__

/**
 * @file 
 * @brief Contains a class for the SquareWorld environment.
 */

namespace rlly
{
namespace env
{

/**
 * @brief SquareWorld environment with states in [0, 1]^2 and 4 actions 
 * @details 
 *      The agent starts at (start_x, start_y) and, in each state, it can take for actions (0 to 3) representing a
 *      displacement of (-d, 0), (d, 0), (0, -d) and (0, d), respectively.
 *          
 *      The immediate reward received in each state s = (s_x, s_y) is, for any action a,
 *          r(s, a) = exp( - ((s_x-goal_x)^2 + (s_y-goal_y)^2)/(2*reward_smoothness^2)  )
 */
class SquareWorld: public ContinuousStateEnv, public rlly::utils::render::RenderInterface2D<std::vector<double>>
{
private:
    // Coordinates of start position
    double start_x = 0.1;
    double start_y = 0.1;

    // Coordinates of goal position (where reward is max)
    double goal_x = 0.75;
    double goal_y = 0.75;

    // Action displacement
    double displacement = 0.1;

    // Reward smoothness
    double reward_smoothness = 0.1;

    // Standard dev of reward noise (gaussian)
    double reward_noise_stdev = 0.01;

    // Standard dev of transition noise (gaussian) 
    double transition_noise_stdev = 0.01;

public:
    SquareWorld();
    ~SquareWorld(){};

    std::unique_ptr<ContinuousStateEnv> clone() const override;
    std::vector<double> reset() override;
    env::StepResult<std::vector<double>> step(int action) override;

    utils::render::Scene2D get_scene_for_render2d(std::vector<double> state_var) override;    
    utils::render::Scene2D get_background_for_render2d();

};

}  // namespace env
}  // namespace rlly

#endif
#ifndef __RLLY_DISCRETE_REWARD_H__
#define __RLLY_DISCRETE_REWARD_H__

/**
 * @file
 * @brief Contains class for defining rewards in finite MDPs.
 */

namespace rlly
{
namespace env
{
    
/**
 * Class for defining reward functions in finite MDPs.
 */ 
class DiscreteReward
{

public:
    /**
     * Default constructor
     */
    DiscreteReward();
    /**
     * Create "reward without noise" object
     * @param _mean_rewards
     */
    DiscreteReward(utils::vec::vec_3d _mean_rewards);
    /**
     * Create "reward with noise" object
     * @param _mean_rewards
     * @param _noise_type 
     * @param _noise_params
     */
    DiscreteReward(utils::vec::vec_3d _mean_rewards, std::string _noise_type, std::vector<double> _noise_params);
    ~DiscreteReward(){};

    /**
    * 3d vector such that mean_rewards[s][a][s'] is the mean reward obtained when the
    * state s' is reached by taking action a in state s.
    */
    utils::vec::vec_3d mean_rewards;

    /**
     * String describing the type of noise
     * "none": zero noise
     * "gaussian": zero-mean Gaussian distribution with variance given in noise_params
     */
    std::string noise_type;

    /**
    * Vector of noise parameters
    * - Gaussian noise: noise_params = [variance]
    */
    std::vector<double> noise_params;

    /**
     * Get a reward sample at (state, action, next_state)
     * @param state
     * @param action
     * @param next_state
     * @param randgen random number generator for sampling the noise
     */
    double sample(int state, int action, int next_state, utils::rand::Random randgen);
};

} // namespace env
} // namespace rlly

#endif
#ifndef __RLLY_FINITEMDP_H__
#define __RLLY_FINITEMDP_H__

/**
 * @file
 * @brief Base class for finite MDPs.
 */

namespace rlly
{
namespace env
{
/**
 * Base class for Finite Markov Decision Processes with __known__ transitions and rewards.
 */ 
class FiniteMDP: public FiniteEnv
{

public:
    /**
     * @param _reward_function object of type DiscreteReward representing the reward function
     * @param _transitions
     * @param _default_state index of the default state
     * @param _seed random seed
     */
    FiniteMDP(DiscreteReward _reward_function, utils::vec::vec_3d _transitions, int _default_state = 0, int _seed = -1);

    /**
     * @param _reward_function object of type DiscreteReward representing the reward function
     * @param _transitions
     * @param _terminal_states vector containing the indices of the terminal states
     * @param _default_state index of the default state
     * @param _seed random seed
     */
    FiniteMDP(DiscreteReward _reward_function, utils::vec::vec_3d _transitions, std::vector<int> _terminal_states, int _default_state = 0, int _seed = -1);

    ~FiniteMDP(){};

    /**
     * @brief Set MDP to default_state
     * @return default_state
     */
    int reset();

    /**
     * @brief take a step in the MDP
     * @param action action to take
     * @return StepResult object, contaning next state, reward and 'done' flag
     */
    virtual StepResult<int> step(int action) override;

    /**
     * @brief Check if _state is terminal
     * @param _state
     * @return true if _state is terminal, false otherwise
     */
    bool is_terminal(int _state);

protected:
    /**
     * @brief Default constructor. Returns a undefined MDP.
     */
    FiniteMDP(){};

    /**
     * @brief Constructor *without* terminal states.
     * @param _reward_function object of type DiscreteReward representing the reward function
     * @param _transitions
     * @param _default_state index of the default state
     * @param _seed random seed. If seed < 1, a random seed is selected by calling std::rand().
     */
    void set_params(DiscreteReward _reward_function, utils::vec::vec_3d _transitions, int _default_state = 0, int _seed = -1);

    /**
     * @brief Constructor *with* terminal states.
     * @param _reward_function object of type DiscreteReward representing the reward function
     * @param _transitions
     * @param _terminal_states vector containing the indices of the terminal states
     * @param _default_state index of the default state
     * @param _seed random seed. If seed < 1, a random seed is selected by calling std::rand().
     */
    void set_params(DiscreteReward _reward_function, utils::vec::vec_3d _transitions, std::vector<int> _terminal_states, int _default_state = 0, int _seed = -1);

    /**
     * @brief check if attributes are well defined.
     */
    void check();
public:
    /**
     * DiscreteReward representing the reward function.
     */
    DiscreteReward reward_function;

    /**
     * 3d vector such that transitions[s][a][s'] is the probability of reaching
     * state s' by taking action a in state s.
     */
    utils::vec::vec_3d transitions;

    /**
     * Default state
     */
    int default_state;

    /**
     * Number of states
     */
    int ns;

    /**
     * Number of actions;
     */
    int na;

    /**
     * Vector of terminal states
     */
    std::vector<int> terminal_states;

};
} // namespace env
} // namespace rlly

#endif
#ifndef __RLLY_CHAIN_H__
#define __RLLY_CHAIN_H__

/**
 * @file
 * @brief Define a N-Chain MDP.
 */

namespace rlly
{
namespace env
{
    
/**
 * @brief N-Chain environment. States = {0, ..., N-1}, Actions = {0, 1}.
 * @details In state I, when action 0 is taken, the next state is min(I+1, N-1).
 *                      when action 1 is taken, the next state is max(I-1, 0).
 *          A reward of 1 is obtained when the next state is N-1.
 * @param N length of the chain.
 */
class Chain: public FiniteMDP
{
public:
    /**
     * @brief Build chain MDP of length N
     */
    Chain(int N=3, double fail_p=0);
    ~Chain(){};

    /**
     * Clone
     */
    std::unique_ptr<FiniteEnv> clone() const override;
};

} // namespace env
} // namespace rlly

#endif
#ifndef __RLLY_MOUNTAINCAR_H__
#define __RLLY_MOUNTAINCAR_H__

namespace rlly
{
namespace env
{
/**
 * @brief 1d Mountain car environment
 * @details
 *    State space = (position, velocity)
 *                   position: value in [-1.2, 0.6]
 *                   velocity: value in [-0.07, 0.07]
 *    Action space: Discrete(3)
 * 
 *    The initial position is a random number (in the position range). 
 *    The initial velocity is 0.
 * 
 *   Action 0: negative force
 *   Action 1: do nothing
 *   Action 2: positive force
 * 
 *   The terminal state is (goal_position, goal_velocity)
 * 
 *   A reward of 0 is obtained everywhere, except for the terminal state, where the reward is 1.
 */
class MountainCar: public ContinuousStateEnv, public rlly::utils::render::RenderInterface2D<std::vector<double>>
{
    
public:
    /**
     * Indices of position and velocity in the state vector.
     */
    enum StateLabel
    {
        position = 0, velocity = 1
    };

    MountainCar();
    std::vector<double> reset();
    StepResult<std::vector<double>> step(int action) override;

    /**
     * Get scene representing a given state
     */
    utils::render::Scene2D get_scene_for_render2d(std::vector<double> state_var) override;

    /**
     * Returns background for rendering 
     */
    utils::render::Scene2D get_background_for_render2d() override;

    /**
     * Clone 
     */
    std::unique_ptr<ContinuousStateEnv> clone() const override;

protected:
    /**
     * @brief Returns true if the state is terminal.
     */
    bool is_terminal(std::vector<double> state);
    /**
     * Position at the terminal state
     */
    double goal_position;
    /**
     * Velocity at the terminal state
     */
    double goal_velocity;

private:
    /**
     * Force magnitude.
     */
    static constexpr double force = 0.001;
    /**
     * Gravity.
     */
    static constexpr double gravity = 0.0025;

};
} // namespace env
} // namespace rlly

#endif
#ifndef __RLLY_GRIDWORLD_H__
#define __RLLY_GRIDWORLD_H__

/**
 * @file
 * @brief Define a simple and finite grid world. No walls!
 */

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
class GridWorld: public FiniteMDP, public rlly::utils::render::RenderInterface2D<int>
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
     * Step function (overriden for storing states for rendering)
     */ 
    StepResult<int> step(int action) override;

    /**
     * Clone 
     */
    std::unique_ptr<FiniteEnv> clone() const override;

    /**
     * Generate 2D representation (Scene) of a given state.
     */
    utils::render::Scene2D get_scene_for_render2d(int state_var) override;

    /**
     * Background for rendering
     */
    utils::render::Scene2D get_background_for_render2d() override;

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
#ifndef __RLLY_CARTPOLE_H__
#define __RLLY_CARTPOLE_H__

namespace rlly
{
namespace env
{

/**
 * CartPole environment, as in https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
 */
class CartPole: public ContinuousStateEnv, public rlly::utils::render::RenderInterface2D<std::vector<double>>
{
private:
    
    const float pi = std::atan(1.0)*4.0;
    const float gravity = 9.8;
    const float mass_cart = 1.0;
    const float mass_pole = 0.1;
    const float total_mass = mass_cart + mass_pole;
    const float half_pole_length = 0.5;
    const float pole_mass_times_length = half_pole_length*mass_pole;
    const float force_magnitude = 10.0;
    const float delta_t = 0.02;

    // angle threshold
    const float theta_threshold_radians = 12.0*pi/180.0;
    // position threshold
    const float x_threshold = 2.4;

    int steps_beyond_done = -1;

public:
    CartPole();
    ~CartPole(){};

    std::vector<double> reset();
    StepResult<std::vector<double>> step(int action) override;

    /**
     *  Clone the object
     */
    std::unique_ptr<ContinuousStateEnv> clone() const override;

    /**
     * Get scene representing a given state
     */
    utils::render::Scene2D get_scene_for_render2d(std::vector<double> state_var) override;

    /**
     * Returns background for rendering 
     */
    utils::render::Scene2D get_background_for_render2d() override;
};

} // namespace env
} // namespace rlly

#endif
#ifndef __RLLY_ENV_H__
#define __RLLY_ENV_H__

/**
 * @file 
 * All env headers.
 */

namespace rlly
{
/**
 * @brief Definitions for reinforcement learning environments
 */
namespace env{}
}

#endif
#ifndef __RLLY_ISOMORPHIC_WRAPPER_H__
#define __RLLY_ISOMORPHIC_WRAPPER_H__

namespace rlly
{
namespace wrappers
{

/**
 * @brief Wrapper such that the observation and action spaces of the wrapper environment are the
 * same as the original environment. 
 * @details Useful to define wrappers like time limit.
 * @tparam EnvType must NOT be abstract, we need to instantiate it to infer the types of state and action spaces
 */
template <typename EnvType>
class IsomorphicWrapper: public EnvType
{
private:
    EnvType foo_env;
    using S_space = decltype(foo_env.observation_space);
    using A_space = decltype(foo_env.action_space);
    S_space foo_obs_space;
    A_space foo_act_space;
public:
    IsomorphicWrapper(EnvType& env): p_env(env)
    {
        this->id                = p_env.id + "IsomorphicWrapper";
        this->observation_space = p_env.observation_space;
        this->action_space      = p_env.action_space;
    };
    ~IsomorphicWrapper(){};

    // type of state and action variables
    using S_type = decltype(foo_obs_space.sample());
    using A_type = decltype(foo_act_space.sample());

    /**
     *  Reference to the wrapped environment.
     */
    EnvType& p_env;

    // reset 
    virtual S_type reset() override
    {
        return p_env.reset();
    };

    // step
    virtual env::StepResult<S_type> step(A_type action) override
    {
        return p_env.step(action);
    };

    /**
     * @brief Returns a null pointer. Prevents the wrapper from being cloned.
     */
    virtual std::unique_ptr<env::Env<S_space, A_space>> clone() const override
    {
        std::cerr << "Error: trying to clone a wrapper, returning nullptr" << std::endl;
        return nullptr;
    };

    // Set seed
    void set_seed(int _seed);
};

template <typename EnvType>
void IsomorphicWrapper<EnvType>::set_seed(int _seed)
{
    p_env.set_seed(_seed);
    int seed = p_env.randgen.get_seed();
    this->observation_space.generator.seed(seed+123);
    this->action_space.generator.seed(seed+456);
}

} // namespace wrappers
} // namespace rlly
#endif
#ifndef __RLLY_BASIC_WRAPPER_H__
#define __RLLY_BASIC_WRAPPER_H__

namespace rlly
{
namespace wrappers
{

/**
 * @brief Wrapper such that the observation and action spaces of the wrapper environment are not 
 * necessarily the same as in the original environment.
 * @tparam EnvType type of the original environment (can be an abstract class)
 * @tparam S_space type of state space of the wrapper (e.g. spaces::Box, spaces::Discrete)
 * @tparam A_space type of action space of the wrapper (e.g. spaces::Box, spaces::Discrete)
 */
template <typename EnvType, typename S_space, typename A_space>
class BaseWrapper: public rlly::env::Env<S_space, A_space>
{
protected:
    S_space foo_obs_space;
    A_space foo_act_space;
public:
    BaseWrapper(EnvType& env): p_env(env){};
    ~BaseWrapper(){};

    // type of state and action variables
    using S_type = decltype(foo_obs_space.sample());
    using A_type = decltype(foo_act_space.sample());

    /**
     * reset() must be implemented by derived class
     */
    virtual S_type reset()=0;

    /**
     * step() must be implemented by derived class
     */
    virtual env::StepResult<S_type> step(A_type action)=0;

    /**
     *  Reference to the wrapped environment.
     */
    EnvType& p_env;

    /**
     * @brief Returns a null pointer. Prevents the wrapper from being cloned.
     */
    virtual std::unique_ptr<env::Env<S_space, A_space>> clone() const override { return nullptr;};

    /**
     * Set seed
     */
    void set_seed(int _seed)
    {
        this->set_seed(_seed+123);
        p_env.set_seed(_seed);
    };
};

} // namespace wrappers
} // namespace rlly

#endif
#ifndef __RLLY_DISCRETIZE_STATE_WRAPPER_H___
#define __RLLY_DISCRETIZE_STATE_WRAPPER_H___

namespace rlly
{
namespace wrappers
{

template <typename EnvType>
class DiscretizeStateWrapper: public BaseWrapper<EnvType, spaces::Discrete, spaces::Discrete>
{
private:
    void init();
    double tol = 1e-9;
public:
    /**
     * @param env
     * @param n_bins number of intervals in the discretization of each dimension of the state space
     */
    DiscretizeStateWrapper(EnvType& env, int n_bins);
    /**
     * @param env
     * @param vec_n_bins vec_n_bins[i] is the number of intervals in the discretization of the i-th each dimension of the state space
     */
    DiscretizeStateWrapper(EnvType& env, std::vector<int> vec_n_bins);
    ~DiscretizeStateWrapper(){};

    // reset
    int reset() override; 

    // step
    env::StepResult<int> step(int action) override;

    /**
     * all_bins[i] = {x_1, ..., x_n } the points corresponding to the discretization of the i-th dimension of the state space
     */  
    utils::vec::vec_2d all_bins;

    /**
     *  Get discrete representation of a continuous state
     */
    int get_state_index(std::vector<double> state);
};

template <typename EnvType>
void DiscretizeStateWrapper<EnvType>::init()
{
    if(this->p_env.observation_space.name != spaces::box)
    {
        std::cerr << "Error: DiscretizeStateWrapper requires an environment with an observation space of type spaces::Box ." << std::endl;
        throw;
    }
    if(this->p_env.action_space.name != spaces::discrete)
    {
        std::cerr << "Error: DiscretizeStateWrapper requires an environment with an action space of type spaces::Discrete." << std::endl;
        throw;
    }
    this->action_space = this->p_env.action_space;
}

template <typename EnvType>
DiscretizeStateWrapper<EnvType>::DiscretizeStateWrapper(EnvType& env, int n_bins): 
    BaseWrapper<EnvType, spaces::Discrete, spaces::Discrete>(env)
{
    init();

    int num_states = 1;
    unsigned int dim = env.observation_space.low.size();
    for(unsigned int dd = 0; dd < dim; dd++)
    {
        double range = env.observation_space.high[dd] - env.observation_space.low[dd];
        double epsilon = range/n_bins;
        // discretization of dimension dd
        std::vector<double> discretization(n_bins+1);
        for (int bin = 0; bin < n_bins+1; bin++)
        {
            discretization[bin] = env.observation_space.low[dd] + epsilon*bin;
            if (bin == n_bins) discretization[bin] += tol; // "close" the last interval
        } 
        all_bins.push_back(discretization);
        num_states = num_states*n_bins;
    }
    this->observation_space.set_n(num_states);
}

template <typename EnvType>
DiscretizeStateWrapper<EnvType>::DiscretizeStateWrapper(EnvType& env, std::vector<int> vec_n_bins): 
    BaseWrapper<EnvType, spaces::Discrete, spaces::Discrete>(env)
{
    init();

    int num_states = 1;
    unsigned int dim = env.observation_space.low.size();
    if (vec_n_bins.size() != dim)
    {
        std::cerr << "Incompatible dimensions in the constructor of DiscretizeStateWrapper." << std::endl;
        throw;
    }
    for(unsigned int dd = 0; dd < dim; dd++)
    {
        int n_bins = vec_n_bins[dd];
        double range = env.observation_space.high[dd] - env.observation_space.low[dd];
        double epsilon = range/n_bins;
        // discretization of dimension dd
        std::vector<double> discretization(n_bins+1);
        for (int bin = 0; bin < n_bins+1; bin++)
        {
            discretization[bin] = env.observation_space.low[dd] + epsilon*bin;
            if (bin == n_bins) discretization[bin] += tol; // "close" the last interval
        } 
        all_bins.push_back(discretization);
        num_states = num_states*n_bins;
    }
    this->observation_space.set_n(num_states);
}

template <typename EnvType>
int DiscretizeStateWrapper<EnvType>::get_state_index(std::vector<double> state)
{
    return utils::binary_search_nd(state, all_bins);
}

template <typename EnvType>
env::StepResult<int> DiscretizeStateWrapper<EnvType>::step(int action)
{
    env::StepResult<int> result;
    env::StepResult<std::vector<double>> wrapped_result = this->p_env.step(action);
    result.next_state = get_state_index(wrapped_result.next_state);
    result.reward     = wrapped_result.reward;
    result.done       = wrapped_result.done;
    return result;
}   

template <typename EnvType>
int DiscretizeStateWrapper<EnvType>::reset()
{
    std::vector<double> wrapped_state = this->p_env.reset();
    return get_state_index(wrapped_state);
}   

} // namespace wrappers
} // namespace rlly

#endif
#ifndef __RLLY_WRAPPERS_H__
#define __RLLY_WRAPPERS_H__

/**
 * @file 
 * All headers for wrappers.
 */

namespace rlly
{

/**
 * @brief Wrappers for the environments
 */
namespace wrappers{}

}
#endif
namespace rlly
{
namespace env
{

GridWorld::GridWorld(int _nrows, int _ncols, double fail_p /* = 0 */, double reward_smoothness /* = 0 */, double reward_sigma /* = 0 */)
{
    nrows = _nrows;
    ncols = _ncols;
    assert(nrows > 1 && "Invalid number of rows");
    assert(ncols > 1 && "Invalid number of columns");
    assert(reward_smoothness >= 0);
    assert(fail_p >= 0.0 && fail_p <= 1.0);

    // Number of states and actions
    int S = ncols*nrows;
    int A = 4;

    // Terminal state
    std::vector<double> goal_coord = { (double) nrows - 1, (double) ncols - 1};
    std::vector<int> _terminal_states = {S - 1};

    // Initialize vectors
    utils::vec::vec_3d _rewards = utils::vec::get_zeros_3d(S, A, S);
    utils::vec::vec_3d _transitions = utils::vec::get_zeros_3d(S, A, S);

    // Build maps between coordinates and indices
    int index = 0;
    for(int rr = 0; rr < nrows; rr++)
    {
        for(int cc = 0; cc < ncols; cc++)
        {
            std::vector<int> coord = {rr, cc};
            coord2index[coord] = index;
            index2coord[index] = coord; 
            index++;
        }
    }

    // Build rewards
    for(int jj = 0; jj < S; jj++)
    {
        std::vector<int>& next_state_coord = index2coord[jj];
        double squared_distance = std::pow( (1.0*next_state_coord[0]-1.0*goal_coord[0])/(nrows-1) , 2)
                                    + std::pow( (1.0*next_state_coord[1]-1.0*goal_coord[1])/(ncols-1), 2);
        double reward = 0;
        if (reward_smoothness > 0)
        {
            reward = std::exp( -squared_distance/ (2*std::pow(reward_smoothness, 2))  );
        }
        else 
        {
            reward = 1.0*(squared_distance == 0);
        }

        for(int ii = 0; ii < S; ii++)
        {
            for(int aa = 0; aa < A; aa++)
            {
                // reward depends on the distance between the next state and the goal state.
                _rewards[ii][aa][jj] = reward;
            }
        }
    }

    // Build transitions
    for(int ii = 0; ii < S; ii++)
    {
        std::vector<int>& state_coord = index2coord[ii];
        for(int aa = 0; aa < A; aa++)
        {
            // Coordinates of the next state
            std::vector<int> next_state_coord = get_neighbor(state_coord, aa);
            int next_state_index = coord2index[next_state_coord];
            _transitions[ii][aa][next_state_index] = 1.0;

            /*
                Handle the failure case.

                With probability fail_p, go to another neighbor!
            */
            if (fail_p > 0)
            {
                for(int bb = 0; bb < A; bb++)
                {
                    if (bb == aa) continue; 
                    std::vector<int> perturbed_next_state_coord = get_neighbor(state_coord, bb);
                    int perturbed_next_state_index = coord2index[perturbed_next_state_coord];
                    _transitions[ii][aa][next_state_index] -= fail_p/4.0;
                    _transitions[ii][aa][perturbed_next_state_index] += fail_p/4.0;
                }  
            }             
        }
    }
    // Initialize base class (FiniteMDP)
    if (reward_sigma == 0)
        set_params(_rewards, _transitions, _terminal_states);
    else
    {
        std::vector<double> noise_params;
        noise_params.push_back(reward_sigma);
        DiscreteReward _reward_function(_rewards, "gaussian", noise_params);
        set_params(_reward_function, _transitions, _terminal_states);
    }
        
    id = "GridWorld";

    // 2D rendering parameters
    refresh_interval_for_render2d = 1000; // 1 second between frames
    clipping_area_for_render2d[0] = 0.0;
    clipping_area_for_render2d[1] = 1.0*ncols;
    clipping_area_for_render2d[2] = 0.0;
    clipping_area_for_render2d[3] = 1.0*nrows;
}

std::vector<int> GridWorld::get_neighbor(std::vector<int> state_coord, int action)
{
    int neighbor_row = state_coord[0];
    int neighbor_col = state_coord[1];      
    switch(action) 
    {
        // Left
        case 0:
            neighbor_col = std::max(0, state_coord[1] - 1);
            break;
        // Right
        case 1:
            neighbor_col = std::min(ncols-1, state_coord[1] + 1);
            break;
        // Up
        case 2:
            neighbor_row = std::max(0, state_coord[0]-1);
            break;
            // Down
        case 3:
            neighbor_row = std::min(nrows-1, state_coord[0]+1);
            break;
    }
    std::vector<int> neighbor_coord = {neighbor_row, neighbor_col};
    return neighbor_coord;
}

void GridWorld::render()
{
    // std::cout<< "GridWorld" << std::endl;
    for(int ii = 0; ii < ncols; ii ++) std::cout<<"----";
    std::cout << std::endl;
    for (auto const& cell : index2coord) // key = cell.first, value = cell.second
    {
        std::string cell_str = "";
        
        // If state index (cell.first) is in terminal states
        if (std::find(terminal_states.begin(), terminal_states.end(), cell.first) != terminal_states.end())
            cell_str = " x  ";
        
        // If current state
        else if (cell.first == state)
            cell_str = " A  ";
        
        // 
        else 
            cell_str = " o  ";
        
        // Display
        std::cout << cell_str;
        if (cell.second[1] == ncols - 1) 
            std::cout << std::endl;
    }
    for(int ii = 0; ii < ncols; ii ++) std::cout<<"----";
    std::cout << std::endl;
}

void GridWorld::render_values(std::vector<double> values)
{
    for(int ii = 0; ii < ncols; ii ++) std::cout<<"------";
    std::cout << std::endl;
    for (auto const& cell : index2coord) // key = cell.first, value = cell.second
    {
        // Round value
        double value = values[cell.first];
        int ivalue = (int) (100*value);
        value = ivalue/100.0;
        std::cout << std::setw (6)<< value;   
        if (cell.second[1] == ncols - 1) 
            std::cout << std::endl;
    }
    for(int ii = 0; ii < ncols; ii ++) std::cout<<"------";
    std::cout << std::endl;       
}

StepResult<int> GridWorld::step(int action)
{
    // for rendering
    if (rendering_enabled) append_state_for_rendering(state);
    
    // Sample next state
    int next_state = randgen.choice(transitions[state][action]);
    double reward = reward_function.sample(state, action, next_state, randgen); 
    bool done = is_terminal(next_state);
    StepResult<int> step_result(next_state, reward, done);
    state = step_result.next_state;
    return step_result;
}

std::unique_ptr<FiniteEnv> GridWorld::clone() const
{
    return std::make_unique<GridWorld>(*this);
}

utils::render::Scene2D GridWorld::get_scene_for_render2d(int state_var)
{
    utils::render::Scene2D scene; 
    utils::render::Geometric2D agent_shape;
    agent_shape.type = "GL_QUADS";
    agent_shape.set_color(0.0, 0.0, 0.5);
    std::vector<int>& state_coord = index2coord[state_var];
    
    // Getting (x, y) representation of the state
    float x_delta = 1.0;
    float y_delta = 1.0;

    float x = state_coord[1]*1.0;  
    float y = state_coord[0]*1.0;  
    x = x + x_delta/2.0;  // centering
    y = y + y_delta/2.0;  // centering

    // 
    float x_size = x_delta/4.0;
    float y_size = y_delta/4.0;

    agent_shape.add_vertex( x - x_size, y - y_size );
    agent_shape.add_vertex( x + x_size, y - y_size );
    agent_shape.add_vertex( x + x_size, y + y_size );
    agent_shape.add_vertex( x - x_size, y + y_size );

    scene.add_shape(agent_shape);
    return scene;
}

utils::render::Scene2D GridWorld::get_background_for_render2d()
{
    utils::render::Scene2D scene; 

    // Getting (x, y) representation of the state
    float x_size = 1.0;
    float y_size = 1.0;

    bool color = true;
    for(int cc = 0; cc < ncols; cc++)
    {
        for(int rr = 0; rr < nrows; rr++)
        {
            utils::render::Geometric2D shape;
            shape.type = "GL_QUADS";
            if ((rr+color) % 2 == 0) shape.set_color(0.35, 0.35, 0.35);
            else shape.set_color(0.5, 0.5, 0.5);
            if ( rr == nrows - 1  && cc == ncols - 1 ) shape.set_color(0.0, 0.5, 0.0);
            float x = 1.0*cc;
            float y = 1.0*rr;
            shape.add_vertex(x, y);
            shape.add_vertex(x+x_size, y);
            shape.add_vertex(x+x_size, y+y_size);
            shape.add_vertex(x, y+y_size);
            scene.add_shape(shape);
        }
        color = !color;
    }
    return scene;
}

} // namespace env 
} // namespace rlly
namespace rlly
{
namespace env
{

Chain::Chain(int N, double fail_p)
{
    assert(N > 0 && "Chain needs at least one state");
    utils::vec::vec_3d _rewards = utils::vec::get_zeros_3d(N, 2, N);
    utils::vec::vec_3d _transitions = utils::vec::get_zeros_3d(N, 2, N);
    std::vector<int> _terminal_states = {N-1};

    for(int state = 0; state < N; state++)
    {
        for(int action = 0; action < 2; action++)
        {
            int next_state = -1;
            // First action: go to the right
            if (action == 0)
            {
                next_state = std::min(state + 1, N-1);
            }
            // Second action: go to the left
            else if (action == 1)
            {
                next_state = std::max(state - 1, 0);
            }
            _transitions[state][action][next_state] = 1.0 - fail_p;
            _transitions[state][action][state] += fail_p;
            if (next_state == N-1)
            {
                _rewards[state][action][next_state] = 1.0;
            }
        }
    }

    set_params(_rewards, _transitions, _terminal_states);
    id = "Chain";
}

std::unique_ptr<FiniteEnv> Chain::clone() const
{
    return std::make_unique<Chain>(*this);
}

}  // namespace env
}  // namespace rlly
namespace rlly
{
namespace env
{

FiniteMDP::FiniteMDP(DiscreteReward _reward_function, utils::vec::vec_3d _transitions, int _default_state /* = 0 */, int _seed /* = -1 */)
{
    set_params(_reward_function, _transitions, _default_state, _seed);
}

FiniteMDP::FiniteMDP(DiscreteReward _reward_function, utils::vec::vec_3d _transitions, std::vector<int> _terminal_states, int _default_state /* = 0 */, int _seed /* = -1 */)
{
    set_params(_reward_function, _transitions, _terminal_states, _default_state, _seed);
}

void FiniteMDP::set_params(DiscreteReward _reward_function, utils::vec::vec_3d _transitions, int _default_state /* = 0 */, int _seed /* = -1 */)
{
    reward_function = _reward_function;
    transitions = _transitions;
    id = "FiniteMDP";
    default_state = _default_state;

    check();
    ns = reward_function.mean_rewards.size();
    na = reward_function.mean_rewards[0].size();

    // observation and action spaces
    observation_space.set_n(ns);
    action_space.set_n(na);
    set_seed(_seed);
    
    reset();
}

void FiniteMDP::set_params(DiscreteReward _reward_function, utils::vec::vec_3d _transitions, std::vector<int> _terminal_states, int _default_state /* = 0 */, int _seed /* = -1 */)
{
    set_params(_reward_function, _transitions, _default_state, _seed);
    terminal_states = _terminal_states;
}

void FiniteMDP::check()
{
    // Check shape of transitions and rewards
    assert(reward_function.mean_rewards.size() > 0);
    assert(reward_function.mean_rewards[0].size() > 0);
    assert(reward_function.mean_rewards[0][0].size() > 0);
    assert(transitions.size() > 0);
    assert(transitions[0].size() > 0);
    assert(transitions[0][0].size() > 0);

    // Check consistency of number of states
    assert(reward_function.mean_rewards[0][0].size() == reward_function.mean_rewards.size());
    assert(transitions[0][0].size() == transitions.size());
    assert(transitions.size() == reward_function.mean_rewards.size());

    // Check consistency of number of actions
    assert(reward_function.mean_rewards[0].size() == transitions[0].size());

    // Check transition probabilities
    for(unsigned int i = 0; i < transitions.size(); i++)
    {
        for(unsigned int a = 0; a < transitions[0].size(); a++)
        {
            double sum = 0;
            for(unsigned int j = 0; j < transitions[0][0].size(); j++)
            {
                assert(transitions[i][a][j] >= 0.0);
                sum += transitions[i][a][j];
            }
            // std::cout << std::abs(sum - 1.0) << std::endl;
            assert(std::abs(sum - 1.0) <= 1e-12 && "Probabilities must sum to 1");
        }
    }
}

int FiniteMDP::reset()
{
    state = default_state;
    return default_state;
}

bool FiniteMDP::is_terminal(int _state)
{
    return (std::find(terminal_states.begin(), terminal_states.end(), _state) != terminal_states.end());
}

/**
 *  @note done is true if next_state is terminal.
 */
StepResult<int> FiniteMDP::step(int action)
{
    // Sample next state
    int next_state = randgen.choice(transitions[state][action]);
    double reward = reward_function.sample(state, action, next_state, randgen); 
    bool done = is_terminal(next_state);
    StepResult<int> step_result(next_state, reward, done);
    state = step_result.next_state;
    return step_result;
}

}  // namespace env
}  // namespace rlly
namespace rlly
{
namespace env
{

MountainCar::MountainCar()
{
    // Set seed
    int _seed = std::rand();
    set_seed(_seed);

    // observation and action spaces
    std::vector<double> _low = {-1.2, -0.07};
    std::vector<double> _high = {0.6, 0.07};
    observation_space.set_bounds(_low, _high);
    action_space.set_n(3);

    goal_position = 0.5;
    goal_velocity = 0;

    state.push_back(0);
    state.push_back(0);

    id = "MountainCar";

    // 2D rendering is enabled for MountainCar
    clipping_area_for_render2d[0] = -1.2;
    clipping_area_for_render2d[1] =  0.6;
    clipping_area_for_render2d[2] = -0.2;
    clipping_area_for_render2d[3] =  1.1;
}

std::vector<double> MountainCar::reset()
{
    state[position] = randgen.sample_real_uniform(observation_space.low[position], observation_space.high[position]);
    state[velocity] = 0;
    return state;
}

StepResult<std::vector<double>> MountainCar::step(int action)
{
    assert(action_space.contains(action));

    // for rendering
    if (rendering_enabled) append_state_for_rendering(state);
    //
    
    std::vector<double>& lo = observation_space.low;
    std::vector<double>& hi = observation_space.high;

    double p = state[position];
    double v = state[velocity];

    v += (action-1)*force + std::cos(3*p)*(-gravity);
    v = utils::clamp(v, lo[velocity], hi[velocity]);
    p += v;
    p = utils::clamp(p, lo[position], hi[position]);
    if ((std::abs(p-lo[position])<1e-10) && (v<0))
    { 
        v = 0;
    }
    bool done = is_terminal(state);
    double reward = 0.0;
    if (done) reward = 1.0;

    state[position] = p;
    state[velocity] = v;

    StepResult<std::vector<double>> step_result(state, reward, done);
    return step_result;
}

bool MountainCar::is_terminal(std::vector<double> state)
{
    return ((state[position] >= goal_position) && (state[velocity]>=goal_velocity));
}

std::unique_ptr<ContinuousStateEnv> MountainCar::clone() const
{
    return std::make_unique<MountainCar>(*this);
}

utils::render::Scene2D MountainCar::get_scene_for_render2d(std::vector<double> state_var)
{
    float y = std::sin(3*state_var[position])*0.45 + 0.55;
    float x = state_var[position];

    utils::render::Scene2D car_scene;
    utils::render::Geometric2D car;
    car.type = "GL_POLYGON";
    car.set_color(0.0f, 0.0f, 0.0f);
    
    float size = 0.025;
    car.add_vertex(x - size, y - size);
    car.add_vertex(x + size, y - size);
    car.add_vertex(x + size, y + size);
    car.add_vertex(x - size, y + size);

    car_scene.add_shape(car);
    return car_scene;
}

utils::render::Scene2D MountainCar::get_background_for_render2d()
{
    utils::render::Scene2D background;
    utils::render::Geometric2D mountain;
    utils::render::Geometric2D flag;
    mountain.type = "GL_TRIANGLE_FAN";
    mountain.set_color(0.6, 0.3, 0.0);
    flag.type     = "GL_TRIANGLES";
    flag.set_color(0.0, 0.5, 0.0);

    std::vector<std::vector<float>> vertices1 = {{-1.0, -1.0}};

    // Mountain
    mountain.add_vertex( -0.3f, -1.0f);
    mountain.add_vertex(  0.6f, -1.0f);

    int n_points = 100;
    double range = observation_space.high[0] - observation_space.low[0];
    double eps = range/(n_points-1.0);
    for(int ii = n_points-1; ii >= 0; ii--)
    {
        double x = observation_space.low[0] + ii*eps;
        double y = std::sin(3*x)*0.45 + 0.55;
        mountain.add_vertex(x, y);
    }
    mountain.add_vertex(-1.2f, -1.0f);

    // Flag
    float goal_x = goal_position;
    float goal_y = std::sin(3*goal_position)*0.45 + 0.55;
    flag.add_vertex(goal_x, goal_y);
    flag.add_vertex(goal_x+0.025f, goal_y+0.075f);
    flag.add_vertex(goal_x-0.025f, goal_y+0.075f);

    background.add_shape(mountain);
    background.add_shape(flag);
    return background;
}

}  // namespace env
}  // namespace rlly
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

    // 2D rendering parameters
    clipping_area_for_render2d[0] = -2.4;
    clipping_area_for_render2d[1] =  2.4;
    clipping_area_for_render2d[2] = -0.5;
    clipping_area_for_render2d[3] =  1.5;
}

StepResult<std::vector<double>> CartPole::step(int action)
{
    // for rendering
    if (rendering_enabled) append_state_for_rendering(state);
    //
    
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

std::unique_ptr<ContinuousStateEnv> CartPole::clone() const
{
    return std::make_unique<CartPole>(*this);
}

utils::render::Scene2D CartPole::get_scene_for_render2d(std::vector<double> state_var)
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

    utils::render::Scene2D cartpole_scene;
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

utils::render::Scene2D CartPole::get_background_for_render2d()
{
    utils::render::Scene2D background;
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
namespace rlly
{
namespace utils
{
namespace vec
{
    
double mean(std::vector<double> vec)
{
    double result = 0;
    int n = vec.size();
    if (n == 0) {std::cerr << "Warning: calling mean() on empty vector." <<std::endl;}
    for(int i = 0; i < n; i++)
    {
        result += vec[i];
    }
    return result/((double) n);
}

double stdev(std::vector<double> vec)
{
    int n = vec.size();
    if (n == 0) {std::cerr << "Warning: calling stdev() on empty vector." <<std::endl;}
    double mu = mean(vec);
    std::vector<double> aux(n);
    for(int i = 0; i < n; i++)
    {
        aux[i] = std::pow(vec[i]-mu, 2.0);
    }
    return std::sqrt(mean(aux));
}

double inner_prod(std::vector<double> vec1, std::vector<double> vec2)
{
    unsigned int n = vec1.size();
    assert( n == vec2.size() && "vec1 and vec2 must have the same size.");
    if (n == 0) {std::cerr << "Warning: calling inner_prod() on empty vectors." <<std::endl;}
    double result = 0.0;
    for(unsigned int i = 0; i < n; i++)
    {
        result += vec1[i]*vec2[i];
    }
    return result;
}

ivec_2d get_zeros_i2d(int dim1, int dim2)
{
    utils::vec::ivec_2d vector;
    for(int ii = 0; ii < dim1; ii++)
    {
        vector.push_back(std::vector<int>());
        for(int jj = 0; jj < dim2; jj++) vector[ii].push_back(0);
    }
    return vector;
}

ivec_3d get_zeros_i3d(int dim1, int dim2, int dim3)
{
    utils::vec::ivec_3d vector;
    for(int ii = 0; ii < dim1; ii++)
    {
        vector.push_back(std::vector<std::vector<int>>());
        for(int jj = 0; jj < dim2; jj++)
        {
            vector[ii].push_back(std::vector<int>());
            for(int kk = 0; kk < dim3; kk++) vector[ii][jj].push_back(0);
        }
    }
    return vector;
}

ivec_4d get_zeros_i4d(int dim1, int dim2, int dim3, int dim4)
{
    utils::vec::ivec_4d vector;
    for(int ii = 0; ii < dim1; ii++)
    {
        vector.push_back(std::vector<std::vector<std::vector<int>>>());
        for(int jj = 0; jj < dim2; jj++)
        {
            vector[ii].push_back(std::vector<std::vector<int>>());
            for(int kk = 0; kk < dim3; kk++)
            {
                vector[ii][jj].push_back(std::vector<int>());
                for(int ll = 0; ll < dim4; ll++) vector[ii][jj][kk].push_back(0);
            }
        }
    }
    return vector;
}

vec_2d get_zeros_2d(int dim1, int dim2)
{
    utils::vec::vec_2d vector;
    for(int ii = 0; ii < dim1; ii++)
    {
        vector.push_back(std::vector<double>());
        for(int jj = 0; jj < dim2; jj++) vector[ii].push_back(0.0);
    }
    return vector;
}

vec_3d get_zeros_3d(int dim1, int dim2, int dim3)
{
    utils::vec::vec_3d vector;
    for(int ii = 0; ii < dim1; ii++)
    {
        vector.push_back(std::vector<std::vector<double>>());
        for(int jj = 0; jj < dim2; jj++)
        {
            vector[ii].push_back(std::vector<double>());
            for(int kk = 0; kk < dim3; kk++) vector[ii][jj].push_back(0.0);
        }
    }
    return vector;
}

vec_4d get_zeros_4d(int dim1, int dim2, int dim3, int dim4)
{
    utils::vec::vec_4d vector;
    for(int ii = 0; ii < dim1; ii++)
    {
        vector.push_back(std::vector<std::vector<std::vector<double>>>());
        for(int jj = 0; jj < dim2; jj++)
        {
            vector[ii].push_back(std::vector<std::vector<double>>());
            for(int kk = 0; kk < dim3; kk++)
            {
                vector[ii][jj].push_back(std::vector<double>());
                for(int ll = 0; ll < dim4; ll++) vector[ii][jj][kk].push_back(0.0);
            }
        }
    }
    return vector;
}

} // namespace vec
} // namespace utils
} // namespace rlly
namespace rlly
{
namespace env
{

SquareWorld::SquareWorld(/* args */)
{
    id = "SquareWorld";
    state.push_back(start_x);
    state.push_back(start_y);

    // observation and action spaces
    std::vector<double> _low  = {0.0, 0.0};
    std::vector<double> _high = {1.0, 1.0};
    observation_space.set_bounds(_low, _high);
    action_space.set_n(4);

    // set seed
    int _seed = std::rand();
    set_seed(_seed);

    // SquareWorld supports 2d rendering
    refresh_interval_for_render2d = 500;
    clipping_area_for_render2d[0] = 0.0;
    clipping_area_for_render2d[1] = 1.0;
    clipping_area_for_render2d[2] = 0.0;
    clipping_area_for_render2d[3] = 1.0;
}

env::StepResult<std::vector<double>> SquareWorld::step(int action)
{
    // for rendering
    if (rendering_enabled) append_state_for_rendering(state);

    //
    bool done = false; 
    double reward = std::exp( -0.5*(std::pow(state[0]-goal_x, 2) + std::pow(state[1]-goal_y, 2))/(std::pow(reward_smoothness, 2)));
    reward += randgen.sample_gaussian(0, reward_noise_stdev);

    double noise_x = randgen.sample_gaussian(0, transition_noise_stdev);
    double noise_y = randgen.sample_gaussian(0, transition_noise_stdev);

    state[0] = std::min(1.0, std::max(0.0, state[0] + noise_x));
    state[1] = std::min(1.0, std::max(0.0, state[1] + noise_y)); 

    if      (action == 0) state[0] = std::max(0.0, state[0] - displacement);
    else if (action == 1) state[0] = std::min(1.0, state[0] + displacement);
    else if (action == 2) state[1] = std::max(0.0, state[1] - displacement);
    else if (action == 3) state[1] = std::min(1.0, state[1] + displacement);

    env::StepResult<std::vector<double>> result(state, reward, done);
    return result;
}

std::vector<double> SquareWorld::reset()
{
    std::vector<double> initial_state {start_x, start_y};
    state = initial_state;
    return initial_state; 
}

std::unique_ptr<ContinuousStateEnv> SquareWorld::clone() const
{
    return std::make_unique<SquareWorld>(*this);
}

utils::render::Scene2D SquareWorld::get_scene_for_render2d(std::vector<double> state_var)
{
    utils::render::Scene2D agent_scene;
    utils::render::Geometric2D agent;
    agent.type = "GL_QUADS";
    agent.set_color(0.75, 0.0, 0.5);

    float size = 0.025;
    float x = state_var[0];
    float y = state_var[1];
    agent.add_vertex(x-size/4.0, y-size);
    agent.add_vertex(x+size/4.0, y-size);
    agent.add_vertex(x+size/4.0, y+size);
    agent.add_vertex(x-size/4.0, y+size);

    agent.add_vertex(x-size, y-size/4.0);
    agent.add_vertex(x+size, y-size/4.0);
    agent.add_vertex(x+size, y+size/4.0);
    agent.add_vertex(x-size, y+size/4.0);

    agent_scene.add_shape(agent);
    return agent_scene;
}

utils::render::Scene2D SquareWorld::get_background_for_render2d()
{
    utils::render::Scene2D background;
    
    float epsilon = 0.01;
    float x = 0.0;
    while (x < 1.0)
    {
        float y = 0.0;
        while (y < 1.0)
        {
            utils::render::Geometric2D shape;
            shape.type = "GL_QUADS";
            float reward = std::exp( -0.5*(std::pow(x-goal_x, 2) + std::pow(y-goal_y, 2))/(std::pow(reward_smoothness, 2)));

            shape.set_color(0.1, 0.9*reward + 0.1, 0.1);
            shape.add_vertex(x-epsilon, y-epsilon);
            shape.add_vertex(x+epsilon, y-epsilon);
            shape.add_vertex(x+epsilon, y+epsilon);
            shape.add_vertex(x-epsilon, y+epsilon);
            background.add_shape(shape);
            y += epsilon;
        }
        x += epsilon;
    }
    
    return background;
}

}  // namespace env
}  // namespace rlly
namespace rlly
{
namespace env
{

DiscreteReward::DiscreteReward()
{
    noise_type = "none";
}

DiscreteReward::DiscreteReward(utils::vec::vec_3d _mean_rewards)
{
    mean_rewards = _mean_rewards;
    noise_type = "none";
}

DiscreteReward::DiscreteReward(utils::vec::vec_3d _mean_rewards, std::string _noise_type, std::vector<double> _noise_params)
{
    mean_rewards = _mean_rewards;
    noise_type = _noise_type;
    noise_params = _noise_params;
}

double DiscreteReward::sample(int state, int action, int next_state, utils::rand::Random randgen)
{
    double mean_r = mean_rewards[state][action][next_state];
    double noise;
    if (noise_type == "none")
        noise = 0;
    else if(noise_type == "gaussian")
    {
        assert(noise_params.size() == 1 && "noise type and noise params are not compatible");
        noise = randgen.sample_gaussian(0, noise_params[0]);
    }
    else
    {
        std::cerr << "Invalid noise type in DiscreteReward" << std::endl;
    }        
    return mean_r + noise;
}

}  // namespace env
}  // namespace rlly
namespace rlly
{
namespace spaces
{

/*
Members of Box
*/
Box::Box()
{
    // Do nothing. low and high are empty vectors.
    name = box;
}

Box::Box(std::vector<double> _low, std::vector<double> _high, unsigned _seed /* = 42 */)
{
    low = _low;
    high = _high;
    size = _low.size();
    generator.seed(_seed);
    assert(size == _high.size() && "The size of _low and _high must be the same.");
    name = box;
}    

void Box::set_bounds(std::vector<double> _low, std::vector<double> _high)
{
    low = _low; 
    high = _high;
    size = _low.size();
    assert(size == _high.size() && "The size of _low and _high must be the same.");
    name = box;

}

bool Box::contains(std::vector<double> x)
{
    bool contains = true;
    if (x.size() != size)
    {
        contains = false;
    }
    for(unsigned int i = 0; i < x.size(); i++)
    {
        contains = contains && (x[i] >= low[i] && x[i] <= high[i]);
    }
    return contains;
}

std::vector<double> Box::sample()
{
    // uniform real distribution
    std::uniform_real_distribution<double> distribution(0.0,1.0);

    std::vector<double> sampled_state(size);
    for(unsigned int i = 0; i < size; i++)
    {
        double a;
        double b;
        a = low[i];
        b = high[i];
        sampled_state[i] = a + (b-a)*distribution(generator);
    } 
    return sampled_state;
}

}
}namespace rlly
{
namespace wrappers
{

} // namespace wrappers
} // namespace rlly
namespace rlly
{
namespace spaces
{

/*
   Members of Discrete
*/ 

Discrete::Discrete()
{
    n = 0;
    name = discrete;
}

Discrete::Discrete(int _n, unsigned _seed /* = 42 */) 
{
    name = discrete;
    n = _n;
    generator.seed(_seed);
}

void Discrete::set_n(int _n)
{
    name = discrete;
    n = _n;
}

bool Discrete::contains(int x)
{
    return (x >= 0 && x < n);
}

int Discrete::sample()
{
    std::uniform_int_distribution<int> distribution(0,n-1);
    return distribution(generator);
}

}
}namespace rlly
{
namespace utils
{
namespace rand
{

Random::Random(unsigned _seed /* = 42 */)
{
    seed = _seed;
    generator.seed(_seed);
}

void Random::set_seed(unsigned _seed)
{
    seed = _seed;
    generator.seed(_seed);
}

int Random::choice(std::vector<double>& prob, double u /* = -1 */)
{
    int n = prob.size();
    if (n == 0)
    {
        std::cerr << "Calling Random::choice with empty probability vector! Returning -1." << std::endl;
        return -1;
    }
    std::vector<double> cumul(n);

    // Compute cumulative distribution function 
    cumul[0] = prob[0];
    for(int i = 1; i < n; i++)
    {
        cumul[i] = cumul[i-1] + prob[i];
    }
    // Get sample 
    double unif_sample;
    if (u == -1){ unif_sample = real_unif_dist(generator); }
    else {unif_sample = u;}

    for(int i = 0; i < n; i++)
    {
        if (unif_sample <= cumul[i])
        {
            return i;
        }
    }
    return -1;  // in case of error
}

double Random::sample_real_uniform(double a, double b)
{
    assert( b >= a && "b must be greater than a");
    double unif_sample = real_unif_dist(generator);
    return (b - a)*unif_sample + a;
}

double Random::sample_gaussian(double mu, double sigma)
{
    assert ( sigma > 0  && "Standard deviation must be positive.");
    double standard_sample = gaussian_dist(generator);
    return mu + sigma*standard_sample;
}

int Random::get_seed()
{
    return seed;
}

} // namespace rand
} // namesmape utils
} // namespace rlly
namespace rlly
{
namespace utils
{

/**
 *  Map a value x in [x0, x1] linearly to the range [y1, y2]
 */
double linear_map(double x, double x1, double x2, double y1, double y2)
{
    if (x1 == x2 || y1 == y2) return 0.0;
    double a = (y2 - y1)/(x2 - x1);
    double b = y1 - a*x1;
    return a*x + b;
}

int binary_search(double val, std::vector<double> vec, int l /*= 0*/, int r /*= -1*/) 
{ 
    if (r == - 1) r = vec.size()-1;

    if (r > l) 
    { 
        int mid = l + (r - l) / 2; 
        if (vec[mid] <= val && vec[mid+1] > val) 
            return mid;
        if (val >= vec[mid+1])
            return binary_search(val, vec, mid+1, r);
        if (val < vec[mid])
            return binary_search(val, vec, l, mid);
    } 
    return -1; 
} 

int binary_search_nd(std::vector<double> d_val, std::vector<std::vector<double>> bins)
{
    unsigned int dim = bins.size();
    int flat_index = 0;
    int aux = 1;
    if (dim != d_val.size()) throw;
    for(unsigned int dd = 0; dd < dim; dd++)
    {
        int index_dd = binary_search(d_val[dd], bins[dd]);
        if (index_dd == -1) throw;
        flat_index += aux*index_dd;
        aux *= (bins[dd].size()-1);
    }
    return flat_index;
}

}
}

 #endif
