#ifndef __RLLY_RANDOM_H__
#define __RLLY_RANDOM_H__

#include <random>
#include <vector>

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
