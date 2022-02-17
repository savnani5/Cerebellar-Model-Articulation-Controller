#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <boost/tuple/tuple.hpp>
#include "gnuplot-iostream.h"

void write_to_file(std::string file, std::vector<std::pair<float, float>> data)
{
    std::ofstream myfile;
    myfile.open(file);
    for (int i = 0; i < data.size(); i++)
    {
        myfile << data[i].first << " " << data[i].second << std::endl;
    }
    myfile.close();
}

void plot(std::vector<std::pair<float, float>> data, std::vector<std::pair<float, float>> predicted_data, char type)
{
    Gnuplot gp("\"F:\\MEngg Robotics\\ENPM690\\HW2\\gnuplot\\bin\\gnuplot.exe\"");

    if (type == 'd')
    {
        gp << "set title 'Discrete CMAC Fitting (Function x*sin(x))'" << std::endl;
        gp << "plot 'discrete_data.txt' using 1:2 title 'data' with lines smooth csplines, \ 'discrete_predicted_data.txt' using 1:2 title 'predicted data' with lines" << std::endl;
    }
    else
    {
        gp << "set title 'Continous CMAC Fitting (Function x*sin(x))'" << std::endl;
        gp << "plot 'continous_data.txt' using 1:2 title 'data' with lines smooth csplines, \ 'continous_predicted_data.txt' using 1:2 title 'predicted data' with lines" << std::endl;
    }

    std::cin.get();
}
