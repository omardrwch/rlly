#include "params.h"
#include <iostream>
#include <fstream>
#include <vector>

namespace rlly
{
namespace utils
{
namespace params
{

int Params::save(std::string filename)
{
    std::ofstream myfile(filename);
    if (myfile.is_open())
    {
        myfile << "(string)" << std::endl;
        for(auto it = string_params.begin(); it != string_params.end(); ++it)
        {
            myfile << "    " << it->first << " = " << it->second << std::endl;
        }
        myfile << "(int)" << std::endl;
        for(auto it = int_params.begin(); it != int_params.end(); ++it)
        {
            myfile << "    " << it->first << " = " << it->second << std::endl;
        }
        myfile << "(double)" << std::endl;
        for(auto it = double_params.begin(); it != double_params.end(); ++it)
        {
            myfile << "    " << it->first << " = " << it->second << std::endl;
        }
        myfile.close();
        return 0;
    }
    std::cerr << "(!) Params error: unable to open file " <<  filename << std::endl;
    return 1;
}


void Params::print()
{

    std::cout << "(string)" << std::endl;
    for(auto it = string_params.begin(); it != string_params.end(); ++it)
    {
        std::cout << "    " << it->first << " = " << it->second << std::endl;
    }
    std::cout << std::endl;


    std::cout << "(int) " << std::endl;
    for(auto it = int_params.begin(); it != int_params.end(); ++it)
    {
        std::cout << "    " << it->first << " = " << it->second << std::endl;
    }
    std::cout << std::endl;


    std::cout << "(double)" << std::endl;
    for(auto it = double_params.begin(); it != double_params.end(); ++it)
    {
        std::cout << "    " << it->first << " = " << it->second << std::endl;
    }
    std::cout << std::endl;
}


bool Params::is_defined(std::string param_name, std::string param_type /*= ""*/)
{
    // Search in string params
    bool string_param_found = ( string_params.find(param_name) != string_params.end() );
    if(param_type == "string") return string_param_found;

    // Search in int params
    bool int_param_found = ( int_params.find(param_name) != int_params.end() );
    if (param_type == "int") return int_param_found;

    // Search in double params
    bool double_param_found = ( double_params.find(param_name) != double_params.end() );
    if (param_type == "double") return double_param_found;

    return (string_param_found || int_param_found || double_param_found);
}

} // namespace params
} // namespace utils
} // namespace rlly
