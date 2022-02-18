/**
 * Copyright (c) 2022 Paras Savnani (savnani5@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the Software
 * is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <boost/tuple/tuple.hpp>
#include "gnuplot-iostream.h"

/**
 * @brief Write plot data to a text file (usage requirement for gnuplot)
 *
 * @param file File path
 * @param data Continer of the input and output sample data  
 */
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

/**
 * @brief Plot function to plot the actual and predicted data for visualization
 *
 * @param data Continer of the input and output data  
 * @param predicted_data Continer of the input and output predicted data 
 * @param type To distinguish between discrete and continous cmac
 */
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
