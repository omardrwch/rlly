#ifndef __RLLY_PARAMS_UTIL_H__
#define __RLLY_PARAMS_UTIL_H__

#include <map>
#include <string>

/**
 * @file
 * @brief Contains class for defining parameters.
 * 
 * @todo Implement I/O for parameters files.
 */

namespace rlly
{
namespace utils
{
namespace params
{


class Params
{
private:
    /* data */
public:
    Params(){};
    ~Params(){};

    // int parameters
    std::map<std::string, int>          int_params;
    void append(std::string param_name, int param_val)
    {int_params[param_name] = param_val;}; 

    // double parameters
    std::map<std::string, double>       double_params;
    void append(std::string param_name, double param_val)
    {double_params[param_name] = param_val;};

    // string parameters
    std::map<std::string, std::string>  string_params;
    void append(std::string param_name, std::string param_val)
    {string_params[param_name] = param_val;};

    // Check is parameter is defined 
    bool is_defined(std::string param_name, std::string param_type = "");

    // Print function
    void print();

    // Save to .txt file
    int save(std::string filename);
};


} // namespace params
} // namespace utils
} // namespace rlly
#endif
