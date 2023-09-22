#include "TrainModelWorker.h"
#include "Neuronal_Network.h"

// Constructor to initialize the worker with the neural network, training data, and training labels
TrainModelWorker::TrainModelWorker(NeuralNetwork* nn, const std::vector<std::vector<double>>& data, const std::vector<int>& labels)
    : neuralNetwork(nn), trainingData(data), trainingLabels(labels) {
    // Connect the signal from NeuralNetwork to the new signal in TrainModelWorker for progress updates, epoch updates, and error reporting
    connect(neuralNetwork, &NeuralNetwork::trainingProgress, this, &TrainModelWorker::trainingProgressUpdate);
    connect(neuralNetwork, &NeuralNetwork::epochUpdates, this, &TrainModelWorker::epochUpdate);
    connect(neuralNetwork, &NeuralNetwork::errorReported, this, &TrainModelWorker::trainingErrorReported);
}

// Function to execute the worker task, which is to train the neural network
void TrainModelWorker::run() {
    int epochs = 100; // Define the number of epochs for training
    std::vector<double> errors; // Vector to capture errors during training
    int batchSize = 32; // Define the batch size for training

    // Train the neural network with the provided training data, labels, number of epochs, errors vector, and batch size
    neuralNetwork->train(trainingData, trainingLabels, epochs, errors, batchSize);

    // Emit a signal indicating that the training is complete
    emit trainingCompleted("Training complete!");
}
