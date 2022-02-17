#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <math.h>
#include <vector>
#include <unordered_map>
# define PI 3.141592  // pi 

// Cerebellar Motor Articulation Controller (CMAC)
class CMAC
{
private:
    int gen_factor;
    int num_weights;
    std::vector<float> wt_vector;
    int associated_vec_size;
    std::unordered_map<float, int> association_map;

public:
    CMAC(int gen_factor, int num_weights);
    void setGenFactor(int genFactor);
    int getGenFactor();
    int getAssociatedVecSize();
    std::vector<float> getWtVector();
    void setWtVector(int start_index, float correction);
    int getAssociationMapValue(float key);
    void setAssociationMapValue(float key, int value);
    float calculateError(std::vector<std::pair<float, float>> data, std::vector<std::pair<float, float>> predicted_data);
    void generateAssociationMap(std::vector<std::pair<float, float>> data, float lowerlimit, float upperlimit);
    virtual void train(std::vector<std::pair<float, float>> data, float lowerlimit, float upperlimit, int epochs, float lr, float convergenceThreshold) = 0;
    virtual std::vector<std::pair<float, float>> predict(std::vector<std::pair<float, float>> data, float lowerlimit, float upperlimit, float& accuracy, bool train) = 0;
};

// Derived Class DiscreteCMAC
class DiscreteCMAC : public CMAC
{
public:
    DiscreteCMAC(int gen_factor, int num_weights);
    void updateWeights(std::pair<float, float> data_element, int gen_factor, float lr);
    void train(std::vector<std::pair<float, float>> data, float lowerlimit, float upperlimit, int epochs, float lr, float convergenceThreshold);
    std::vector<std::pair<float, float>> predict(std::vector<std::pair<float, float>> data, float lowerlimit, float upperlimit, float& accuracy, bool train);
};


// Derived Class ContinousCMAC
class ContinousCMAC : public CMAC
{
public:
    ContinousCMAC(int gen_factor, int num_weights);
    std::vector<float> generateInputVector(int associated_vec_size, float lowerlimit, float upperlimit);
    void updateWeights(std::pair<float, float> data_element, std::vector<float> input, int gen_factor, float lr);
    void train(std::vector<std::pair<float, float>> data, float lowerlimit, float upperlimit, int epochs, float lr, float convergenceThreshold);
    std::vector<std::pair<float, float>> predict(std::vector<std::pair<float, float>> data, float lowerlimit, float upperlimit, float& accuracy, bool train);
};

//-----------------------------------------------------------

CMAC::CMAC(int gen_factor, int num_weights) : wt_vector(num_weights, 1)
{
    this->gen_factor = gen_factor;
    this->num_weights = num_weights;
    this->associated_vec_size = num_weights + 1 - gen_factor;
}

void CMAC::setGenFactor(int gf)
{
    gen_factor = gf;
}

int CMAC::getGenFactor()
{
    return gen_factor;
}

int CMAC::getAssociatedVecSize()
{
    return associated_vec_size;
}

std::vector<float> CMAC::getWtVector()
{
    return wt_vector;
}

void CMAC::setWtVector(int start_index, float correction)
{
    for (int i = start_index; i < start_index + gen_factor; i++)
        wt_vector[i] += correction;
}

int CMAC::getAssociationMapValue(float key)
{
    return association_map[key];
}

void CMAC::setAssociationMapValue(float key, int value)
{
    association_map[key] = value;
}

float CMAC::calculateError(std::vector<std::pair<float, float>> data, std::vector<std::pair<float, float>> predicted_data)
{
    int sum = 0; int n = data.size();
    for (int i = 0; i < n; i++)
        sum += pow(data[i].second - predicted_data[i].second, 2);

    return sqrt(sum) / n;
}

/**
 * Generate the Association Hash Map for keeping the track of weight activations.
 *
 * @param data Pair of input and output values
 * @param lowerlimit Lowerlimit value for the data samples
 * @param upperlimit Uperlimit value for the data samples
 * @return Hash Map for mapping between input space and the weight vector start index for corresponding input
 */
void CMAC::generateAssociationMap(std::vector<std::pair<float, float>> data, float lowerlimit, float upperlimit)
{
    association_map.clear();
    int associated_vec_size = getAssociatedVecSize();
    float associated_vec_index;
    for (int i = 0; i < data.size(); i++)
    {
        // Hash function to generate index (proportionate ==> can use other functions too)
        associated_vec_index = (associated_vec_size - 2) * ((data[i].first - lowerlimit) / (upperlimit - lowerlimit)) + 1;
        association_map[data[i].first] = (int)associated_vec_index;
    }
}

//----------------------------------------------------

DiscreteCMAC::DiscreteCMAC(int gen_factor, int num_weights) : CMAC(gen_factor, num_weights) {};

void DiscreteCMAC::updateWeights(std::pair<float, float> data_element, int gen_factor, float lr)
{
    int start_index = getAssociationMapValue(data_element.first);
    int y_pred = 0;
    std::vector<float> weights = getWtVector();

    for (int i = start_index; i < start_index + gen_factor; i++)
        y_pred += weights[i];

    float error = data_element.second - y_pred;
    float correction = (lr * error) / gen_factor;
    setWtVector(start_index, correction);
}


void DiscreteCMAC::train(std::vector<std::pair<float, float>> data, float lowerlimit, float upperlimit, int epochs, float lr, float convergenceThreshold)
{
    generateAssociationMap(data, lowerlimit, upperlimit);

    int epoch = 0;
    float prev_loss = 0, curr_loss = 0;
    bool isConverged = false;
    int gf = getGenFactor();
    float accuracy = 0.0;

    while (epoch <= epochs && !isConverged)
    {
        prev_loss = curr_loss;

        for (int i = 0; i < data.size(); i++)
            updateWeights(data[i], gf, lr);

        predict(data, lowerlimit, upperlimit, accuracy, true);

        curr_loss = 1 - accuracy;

        if (abs(prev_loss - curr_loss) < convergenceThreshold)
            isConverged = true;
        
        epoch++;
        std::cout << "DicreteCMAC Training in Progress: " << " Epoch: " << epoch << " Accuracy: " << accuracy*100 << " Error: " << curr_loss << std::endl;
    }
}

std::vector<std::pair<float, float>> DiscreteCMAC::predict(std::vector<std::pair<float, float>> data, float lowerlimit, float upperlimit, float& accuracy, bool train = false)
{
    std::vector<std::pair<float, float>> predicted_data;
    if (!train)
        generateAssociationMap(data, lowerlimit, upperlimit);

    int gf = getGenFactor();
    for (int i = 0; i < data.size(); i++)
    {
        int start_index = getAssociationMapValue(data[i].first);
        float res = 0;
        std::vector<float> weights = getWtVector();
        for (int j = start_index; j < start_index + gf; j++)
            res += weights[j];

        predicted_data.push_back({ data[i].first, res });
    }
    accuracy = 1 - abs(calculateError(data, predicted_data));
    return predicted_data;
}


//-------------------------------------------

ContinousCMAC::ContinousCMAC(int gen_factor, int num_weights) : CMAC(gen_factor, num_weights) {};

std::vector<float> ContinousCMAC::generateInputVector(int associated_vec_size, float lowerlimit, float upperlimit)
{
    std::vector<float> input;
    float increment = 2 * PI / (associated_vec_size - 1);

    for (int i = 0; i < associated_vec_size; i++)
        input.push_back(i * increment);
    return input;
}

void ContinousCMAC::updateWeights(std::pair<float, float> data_element, std::vector<float> input, int gen_factor, float lr)
{

    int start_index = getAssociationMapValue(data_element.first);
    int next_index;

    if (start_index < getAssociatedVecSize() - (gen_factor + 1))
        next_index = start_index + 1;
    else
        next_index = start_index;

    std::vector<float> weights = getWtVector();

    float left_dist, left_wt;
    float right_dist, right_wt;

    left_dist = abs(input[start_index] - data_element.first);
    right_dist = abs(input[next_index] - data_element.first);
    left_wt = right_dist / (left_dist + right_dist);
    right_wt = 1 - left_wt;

    int y_pred = 0;
    for (int i = start_index; i < start_index + gen_factor; i++)
        y_pred += weights[i] * left_wt;

    for (int i = next_index; i < next_index + gen_factor; i++)
        y_pred += weights[i] * right_wt;


    float error = data_element.second - y_pred;
    float correction = (lr * error) / gen_factor;
    setWtVector(start_index, correction);
    setWtVector(next_index, correction);
}

void ContinousCMAC::train(std::vector<std::pair<float, float>> data, float lowerlimit, float upperlimit, int epochs, float lr, float convergenceThreshold)
{
    generateAssociationMap(data, lowerlimit, upperlimit);

    int associated_vec_size = getAssociatedVecSize();
    std::vector<float> input = generateInputVector(associated_vec_size, lowerlimit, upperlimit);


    int epoch = 0;
    float prev_loss = 0, curr_loss = 0;
    bool isConverged = false;
    int gf = getGenFactor();
    float accuracy = 0.0;

    while (epoch <= epochs && !isConverged)
    {
        prev_loss = curr_loss;

        for (int i = 0; i < data.size(); i++)
            updateWeights(data[i], input, gf, lr);

        predict(data, lowerlimit, upperlimit, accuracy, true);

        curr_loss = 1 - accuracy;

        if (abs(prev_loss - curr_loss) < convergenceThreshold)
            isConverged = true;

        epoch++;
        std::cout << "ContinousCMAC Training in Progress: " << " Epoch: " << epoch << " Accuracy: " << accuracy*100 << " Error: " << curr_loss << std::endl;
    }
}

std::vector<std::pair<float, float>> ContinousCMAC::predict(std::vector<std::pair<float, float>> data, float lowerlimit, float upperlimit, float& accuracy, bool train = false)
{
    std::vector<std::pair<float, float>> predicted_data;
    int associated_vec_size = getAssociatedVecSize();
    std::vector<float> input = generateInputVector(associated_vec_size, lowerlimit, upperlimit);

    if (!train)
        generateAssociationMap(data, lowerlimit, upperlimit);

    int gf = getGenFactor();
    for (int i = 0; i < data.size(); i++)
    {
        int start_index = getAssociationMapValue(data[i].first);
        int next_index;

        if (start_index < getAssociatedVecSize() - (gf + 1))
            next_index = start_index + 1;
        else
            next_index = start_index;

        float left_dist, left_wt;
        float right_dist, right_wt;

        left_dist = abs(input[start_index] - data[i].first);
        right_dist = abs(input[next_index] - data[i].first);
        left_wt = right_dist / (left_dist + right_dist);
        right_wt = 1 - left_wt;

        float res = 0;
        std::vector<float> weights = getWtVector();
        for (int i = start_index; i < start_index + gf; i++)
            res += weights[i] * left_wt;

        for (int i = next_index; i < next_index + gf; i++)
            res += weights[i] * right_wt;

        predicted_data.push_back({ data[i].first, res });

    }
    accuracy = 1 - abs(calculateError(data, predicted_data));
    return predicted_data;
}
