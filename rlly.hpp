#include <algorithm>
#include <assert.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
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
    virtual T sample() {T foo; return foo;};

    /**
     * @brief Returns true if x belongs to the space, and false otherwise.
     * @param x 
     */
    virtual bool contains(T x) {return false;};

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

    /**
     * Name of the space
     */
    spc_name name = box;

    // Attributes

    /**
     * Size of the fox
     */
    int size; 

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

    // this should be defined in C++17
    /**
     * @brief Clamp a value between an upper and lower bound. Requires C++17.
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
}
}  

#endif
#ifndef __SPACE_H__
#define __SPACE_H__

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
        };     
    }
}  // namespace utils
}  // namespace rlly
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
 * Base class for Finite Markov Decision Processes.
 */ 
class FiniteMDP: public Env<int, int>
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
    StepResult<int> step(int action);

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

    /**
     * State (observation) space
     */
    spaces::Discrete observation_space;

    /**
     *  Action space
     */
    spaces::Discrete action_space;

    // Members of base class

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
    Chain(int N, double fail_p=0);
    ~Chain(){};
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
class MountainCar: public Env<std::vector<double>, int>
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
    StepResult<std::vector<double>> step(int action);

    /**
    * State (observation) space
    */
    spaces::Box observation_space;

    /**
    *  Action space
    */
    spaces::Discrete action_space;

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
class GridWorld: public FiniteMDP
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

    // initialize pointers of base class
    p_observation_space = &observation_space;
    p_action_space      = &action_space;
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
    for(int i = 0; i < transitions.size(); i++)
    {
        for(int a = 0; a < transitions[0].size(); a++)
        {
            double sum = 0;
            for(int j = 0; j < transitions[0][0].size(); j++)
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
    // Initialize pointers in the base class
    p_action_space = &action_space;
    p_observation_space = &observation_space;

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

    std::vector<double>& lo = observation_space.low;
    std::vector<double>& hi = observation_space.high;

    double p = state[position];
    double v = state[velocity];

    v += (action-1)*force + std::cos(3*p)*(-gravity);
    v = utils::clamp(v, lo[velocity], hi[velocity]);
    p += v;
    p = utils::clamp(p, lo[position], hi[position]);
    if ((abs(p-lo[position])<1e-10) && (v<0)) v = 0;

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

}  // namespace env
}  // namespace rlly
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
    int n = vec1.size();
    assert( n == vec2.size() && "vec1 and vec2 must have the same size.");
    if (n == 0) {std::cerr << "Warning: calling inner_prod() on empty vectors." <<std::endl;}
    double result = 0.0;
    for(int i = 0; i < n; i++)
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
}

Box::Box(std::vector<double> _low, std::vector<double> _high, unsigned _seed /* = 42 */)
{
    low = _low;
    high = _high;
    size = _low.size();
    generator.seed(_seed);
    assert(size == _high.size() && "The size of _low and _high must be the same.");
}    

void Box::set_bounds(std::vector<double> _low, std::vector<double> _high)
{
    low = _low; 
    high = _high;
}

bool Box::contains(std::vector<double> x)
{
    bool contains = true;
    if (x.size() != size)
    {
        contains = false;
    }
    for(int i = 0; i < x.size(); i++)
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
    for(int i = 0; i < size; i++)
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

    int sample = 0;
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

} // namespace rand
} // namesmape utils
} // namespace rlly

 #endif
