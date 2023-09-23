#include "Neuronal_Network.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <numeric>
#include <random>
#include <QString>
#include <omp.h>


// Implementation of Getter Methods
inline MyMatrix NeuralNetwork::getWeights1() const {
    return weights1;
}

inline MyMatrix NeuralNetwork::getBiases1() const {
    return biases1;
}

inline MyMatrix NeuralNetwork::getWeights2() const {
    return weights2;
}

inline MyMatrix NeuralNetwork::getBiases2() const {
    return biases2;
}

inline int NeuralNetwork::getInputSize() const {
    return inputSize;
}

inline int NeuralNetwork::getHiddenSize() const {
    return hiddenSize;
}

inline int NeuralNetwork::getOutputSize() const {
    return outputSize;
}

inline double NeuralNetwork::getLearningRate() const {
    return learningRate;
}
// Static class method that calculates the sigmoid of the value n
double NeuralNetwork::calcSigmoid(double n) {
    return 1.0 / (1.0 + std::exp(-n));
}

// Constructor to initialize the neural network with specified layer sizes and learning rate
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


// Function to predict the output given an input vector
std::vector<double> NeuralNetwork::predict(std::vector<double>& input)
{
    // Convert input to column vector and perform feedforward computation
    MyMatrix inputMatrix(input);
    MyMatrix hidden = weights1 * inputMatrix + biases1;
    sigmoid(hidden);
    MyMatrix output = weights2 * hidden + biases2;
    sigmoid(output);

    // Convert output to vector
    return output.getColumnAsVector(0);
}

// Function to predict the output category given an input vector
int NeuralNetwork::oneHotPredict(std::vector<double> &input) {
    std::vector<double> output = predict(input);
    return static_cast<int>(std::max_element(output.begin(), output.end()) - output.begin());
}


/**
 * @brief Trains the neural network using the provided training data and labels.
 *
 * The function runs the training process over a specified number of epochs,
 * adjusting the network's weights and biases to minimize the error between
 * the network’s output and the target labels. The training data is shuffled
 * at the start of each epoch and is processed in mini-batches. The function
 * utilizes OpenMP for parallel processing of each input in the batch. The
 * backpropagation algorithm is used to calculate the error gradients and
 * update the weights and biases. Progress updates, including the error after
 * each epoch, are emitted as signals.
 *
 * @param inputs A vector of input vectors, each representing the features of a training example.
 * @param labels A vector of integers representing the target labels corresponding to the input vectors.
 * @param epochs The number of times the entire training dataset is processed.
 * @param errors A reference to a vector where the mean squared error is recorded every 5000 data points.
 * @param batchSize The number of training examples in each mini-batch.
 */
void NeuralNetwork::train(std::vector<std::vector<double>>& inputs, std::vector<int>& labels, int epochs, std::vector<double>& errors, int batchSize) {
    // Determine the number of inputs and batches
    int numInputs = static_cast<int>(inputs.size());
    int numBatches = (numInputs + batchSize - 1) / batchSize;

    // Start the training loop for the specified number of epochs
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double error = 0.0;

        // 1. Shuffle dataset: Create a list of indices and shuffle them to randomize the input data for each epoch
        std::vector<int> indices(numInputs);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        // Loop over each batch
        for (int b = 0; b < numBatches; ++b) {
            int start = b * batchSize;
            int end = std::min(start + batchSize, numInputs);

            // Parallelize the processing of each input in the batch using OpenMP
#pragma omp parallel for reduction(+:error) num_threads(numThreads)
            for (int i = start; i < end; ++i) {
                int idx = indices[i]; // Using the shuffled index

                // Forward pass: Compute the output of the network given the input
                MyMatrix inputMatrix(inputs[idx]);
                MyMatrix hidden = weights1 * inputMatrix + biases1;
                sigmoid(hidden);
                MyMatrix output = weights2 * hidden + biases2;
                sigmoid(output);

                // Setup target matrix: Initialize it with zeros and set the corresponding label index to 1
                MyMatrix targetMatrix(output.rows(), 1);
                targetMatrix.setAll(0);
                targetMatrix(labels[idx], 0) = 1;

                // Calculate output error: Difference between the network's output and the target
                MyMatrix outputErrorMatrix = output - targetMatrix;

                // Compute squared error for the current input and accumulate it
                double currentError = outputErrorMatrix.elementWiseProduct(outputErrorMatrix).sum();
                error += currentError;

                // Backpropagation: Compute the error for the hidden layer and the gradients for weights and biases
                MyMatrix hiddenError = weights2.transpose() * outputErrorMatrix;
                MyMatrix hiddenGradient = hidden.elementWiseProduct(MyMatrix::allOnes(hidden.rows(), hidden.columns()) - hidden).elementWiseProduct(hiddenError);

                MyMatrix weights2Delta = outputErrorMatrix * hidden.transpose();
                MyMatrix biases2Delta = outputErrorMatrix;

                MyMatrix weights1Delta = hiddenGradient * inputMatrix.transpose();
                MyMatrix biases1Delta = hiddenGradient;

                // Update weights and biases using the computed gradients and the learning rate
                weights2 -= weights2Delta * learningRate;
                biases2 -= biases2Delta * learningRate;
                weights1 -= weights1Delta * learningRate;
                biases1 -= biases1Delta * learningRate;

                // Log progress: Record the mean squared error every 5000 datapoints
                if (i % 5000 == 0 && i != 0) {
                    errors.push_back(error / i);
                }
            }
        }

        // Compute the mean error for this epoch and emit signals for progress update
        error /= numInputs;
        QString updateMessage = QString("Training Epoch %1 completed. Current error: %2").arg(epoch).arg(error);
        emit trainingProgress(updateMessage);
        emit epochUpdates(epoch);
        emit errorReported(error);
    }
}







// Function to apply the sigmoid function to all elements of the matrix
void NeuralNetwork::sigmoid(MyMatrix& matrix)
{
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.columns(); ++j) {
            matrix(i, j) = calcSigmoid(matrix(i, j));
        }
    }
}

////Serialization:
// Function to save the neural network parameters to a file

void NeuralNetwork::save(const std::string& filename) const {
    std::ofstream file(filename);
     // Writing weights and biases matrices to the file
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
// Function to load the neural network parameters from a file
void NeuralNetwork::load(const std::string& filename) {
    // Reading weights and biases matrices from the file
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
