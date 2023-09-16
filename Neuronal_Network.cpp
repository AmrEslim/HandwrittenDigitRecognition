#include "Neuronal_Network.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <numeric>
#include <random>


/// Static class method that calculates the sigmoid of the value n
/// @param n The variable to be sigmoided
double NeuralNetwork::calcSigmoid(double n) {
    return 1.0 / (1.0 + std::exp(-n));
}

/// Constructs the neural network
/// @param inputSize Should equal the number of inputs to be fed into the neural network
/// @param hiddenSize Number of nodes in the hidden layer. A larger number will be able to solve more complex problems but will take longer to learn
/// @param outputSize Number of outputs, should equal the number of class you are trying to differentiate
/// @param learningRate The rate at which the neural network learns. Should be set with care.
NeuralNetwork::NeuralNetwork(int inputSize, int hiddenSize, int outputSize, double learningRate)
        : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize), learningRate(learningRate),
          weights1(hiddenSize, inputSize), biases1(hiddenSize, 1), weights2(outputSize, hiddenSize), biases2(outputSize, 1)
{
    // Initialize weights and biases randomly
    weights1.randomize(-1, 1);
    biases1.randomize(-1, 1);
    weights2.randomize(-1, 1);
    biases2.randomize(-1, 1);
}


/// Calculates the output of the neural network given a set of inputs
/// @param input The inputs to be fed into the neural network. Its size should equal the size of the input layer
std::vector<double> NeuralNetwork::predict(std::vector<double>& input)
{
    // Convert input to column vector
    MyMatrix inputMatrix(input);

    // Feedforward
    MyMatrix hidden = weights1 * inputMatrix + biases1;
    sigmoid(hidden);

    MyMatrix output = weights2 * hidden + biases2;
    sigmoid(output);

    // Convert output to vector
    return output.getColumnAsVector(0);
}


/// Calculates the output of the neural network from the given input and returns the index of the highest value in the output
/// @param input The inputs to be fed into the neural network. Its size should equal the size of the input layer
int NeuralNetwork::oneHotPredict(std::vector<double> &input) {
    std::vector<double> output = predict(input);
    return static_cast<int>(std::max_element(output.begin(), output.end()) - output.begin());
    //return std::max_element(output.begin(), output.end()) - output.begin();
}


/// Trains the neural network for the given input for the specified length of time
/// @param inputs a vector of inputs, where each input is itself a vector of doubles. Each individual input should be the same size as the input layer
/// @param labels a vector of ints that show which output is the correct answer. Should correspond 1-to-1 with the inputs
/// @param epochs The number of times the neural network should loop over the inputs for training
/// @param errors A reference to a vector of doubles. The neural network will push_back the current error of the neural network every 5000 datapoints
void NeuralNetwork::train(std::vector<std::vector<double>>& inputs, std::vector<int>& labels, int epochs, std::vector<double>& errors, int batchSize = 32) {
    int numInputs = static_cast<int>(inputs.size());
    //int numInputs = inputs.size();
    int numBatches = (numInputs + batchSize - 1) / batchSize;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double error = 0.0;
        
        // 1. Shuffle dataset
        std::vector<int> indices(numInputs);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        for (int b = 0; b < numBatches; ++b) {
            int start = b * batchSize;
            int end = std::min(start + batchSize, numInputs);

            for (int i = start; i < end; ++i) {
                int idx = indices[i]; // Using the shuffled index
                
                // Forward pass
                MyMatrix inputMatrix(inputs[idx]);
                MyMatrix hidden = weights1 * inputMatrix + biases1;
                sigmoid(hidden);
                MyMatrix output = weights2 * hidden + biases2;
                sigmoid(output);
                
                // Setup target matrix
                MyMatrix targetMatrix(outputSize, 1);
                targetMatrix.setAll(0);
                targetMatrix(labels[idx], 0) = 1;
                
                // ... [rest remains the same]
            }

            // Log progress
            if (b % (numBatches/10) == 0) {  // log every 10% of progress
                errors.push_back(error / ((b+1) * batchSize));  
            }
        }
        
        // Print error for this epoch
        error /= numInputs;
        std::cout << "Epoch " << epoch << ", Total error: " << error << std::endl;

        // Save after each epoch
        std::string filename = "checkpoint_epoch_" + std::to_string(epoch) + ".nn";
        save(filename);
    }
}




/// Calculates the sigmoid of all the values of the matrix
/// @param matrix The matrix to calculate the sigmoid of its values
void NeuralNetwork::sigmoid(MyMatrix& matrix)
{
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.columns(); ++j) {
            //matrix(i, j) = 1 / (1 + std::exp(-matrix(i, j)));
            matrix(i, j) = calcSigmoid(matrix(i, j));
        }
    }
}

////Serialization:
/////saves the status of the training process

void NeuralNetwork::save(const std::string& filename) const {
    std::ofstream file(filename);

    // Serialize weights1
    file << weights1.rows() << " " << weights1.columns() << "\n";
    for (int i = 0; i < weights1.rows(); ++i) {
        for (int j = 0; j < weights1.columns(); ++j) {
            file << weights1(i, j) << " ";
        }
        file << "\n";
    }
    file << biases1.rows() << " " << biases1.columns() << "\n";
    for (int i = 0; i < biases1.rows(); ++i) {
        for (int j = 0; j < biases1.columns(); ++j) {
            file << biases1(i, j) << " ";
        }
        file << "\n";
    }
    file << weights2.rows() << " " << weights2.columns() << "\n";
    for (int i = 0; i < weights2.rows(); ++i) {
        for (int j = 0; j < weights2.columns(); ++j) {
            file << weights2(i, j) << " ";
        }
        file << "\n";
    }

    file << biases2.rows() << " " << biases2.columns() << "\n";
    for (int i = 0; i < biases2.rows(); ++i) {
        for (int j = 0; j < biases2.columns(); ++j) {
            file << biases2(i, j) << " ";
        }
        file << "\n";
    }

    file.close();
}


//
////Deserialization
///// Loads the status of the training process
//
void NeuralNetwork::load(const std::string& filename) {
    std::ifstream file(filename);
    int rows, cols;

    // Deserialize weights1
    file >> rows >> cols;
    weights1.resize(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file >> weights1(i, j);
        }
    }
    file >> rows >> cols;
    biases1.resize(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file >> biases1(i, j);
        }
    }
    file >> rows >> cols;
    weights2.resize(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file >> weights2(i, j);
        }
    }
    file >> rows >> cols;
    biases2.resize(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file >> biases2(i, j);
        }
    }

    file.close();
}
