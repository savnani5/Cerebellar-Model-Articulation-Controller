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

#include <chrono>
#include <ctime>
#include <random>
#include <algorithm>
#include "cmac.h"
#include "utils.h"


int main()
{
    int points = 100;
    int gen_factor = 2;
    int num_weights = 35;

    std::vector<std::pair<float, float>> data;
    data.push_back({ 0, 0 * sin(0) });
    float increment = 2 * PI / points;

    for (int i = 1; i < points; i++)
        data.push_back({ i * increment, (i * increment) * sin(i * increment) });

    // Random shuffling of the data
    unsigned seed = 0;
    shuffle(data.begin(), data.end(), std::default_random_engine(seed));

    // Train data
    std::vector<std::pair<float, float>> train(data.begin(), data.begin() + 70);
    std::vector<std::pair<float, float>> test(data.begin() + 70, data.end());

    float lowerlimit = 0;
    float upperlimit = 2 * PI;
    int epochs = 2000;
    float lr = 0.01;
    float convergenceThreshold = 0.00000000001;
    float accuracy = 1.0;
    std::vector<std::pair<float, float>> predicted_data_dicrete;
    std::vector<std::pair<float, float>> predicted_data_continous;

    // discrete cmac
    DiscreteCMAC discrete_cmac(gen_factor, num_weights);

    //Training
    auto dt_start = std::chrono::high_resolution_clock::now();
    discrete_cmac.train(train, lowerlimit, upperlimit, epochs, lr, convergenceThreshold);
    auto dt_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms_d = std::chrono::duration<double, std::milli>(dt_end - dt_start).count();
    
    std::cout << "DiscreteCMAC: " << " Generalization Factor : " << gen_factor << " Convergence Time : " << elapsed_time_ms_d << std::endl;

    predicted_data_dicrete = discrete_cmac.predict(test, lowerlimit, upperlimit, accuracy, false);
    sort(predicted_data_dicrete.begin(), predicted_data_dicrete.end());


    std::cout << std::endl << "----------------------------------------------------------------------------" << std::endl;

    // continous cmac
    accuracy = 0.0;
    ContinousCMAC continous_cmac(gen_factor, num_weights);

    // Training
    auto ct_start = std::chrono::high_resolution_clock::now();
    continous_cmac.train(train, lowerlimit, upperlimit, epochs, lr, convergenceThreshold);
    auto ct_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms_c = std::chrono::duration<double, std::milli>(ct_end - ct_start).count();
    std::cout << "ContinousCMAC: " << " Generalization Factor: " << gen_factor << " Convergence Time: " << elapsed_time_ms_c << std::endl;


    predicted_data_continous = continous_cmac.predict(test, lowerlimit, upperlimit, accuracy, false);
    sort(predicted_data_continous.begin(), predicted_data_continous.end());


    // sorting the original data for plotting
    sort(data.begin(), data.end());

    //plotting the discrete cmac
    write_to_file("discrete_data.txt", data);
    write_to_file("discrete_predicted_data.txt", predicted_data_dicrete);
    plot(data, predicted_data_dicrete, 'd');

    std::cin.get();
    //plotting the continous cmac
    write_to_file("continous_data.txt", data);
    write_to_file("continous_predicted_data.txt", predicted_data_continous);
    plot(data, predicted_data_continous, 'c');

    //----------------------------------------------------------------------
   
    //Analysis of Generalization Factors and Convergence times

    //DiscreteCMAC discrete_cmac(gen_factor, num_weights);

    //float time;
    //std::vector<int> gen_factors;
    //for (int i = 1; i <= num_weights; i++)
    //    gen_factors.push_back(i);
    //
    //for (auto gf : gen_factors)
    //{
    //    accuracy = 1.0;
    //    discrete_cmac.setGenFactor(gf);
    //    
    //    auto t_start = std::chrono::high_resolution_clock::now();
    //    discrete_cmac.train(train, lowerlimit, upperlimit, epochs, lr, convergenceThreshold);
    //    auto t_end = std::chrono::high_resolution_clock::now();
    //    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    //    predicted_data_dicrete = discrete_cmac.predict(test, lowerlimit, upperlimit, accuracy, false);
    //     
    //    std::cout << "Generalization Factor: " << gf << " Convergence Time(ms): " << elapsed_time_ms << std::endl;
    //}
}

