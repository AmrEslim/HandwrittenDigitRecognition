#include "TrainModelWorker.h"
#include "Neuronal_Network.h"


TrainModelWorker::TrainModelWorker(NeuralNetwork* nn, const std::vector<std::vector<double>>& data, const std::vector<int>& labels)
    : neuralNetwork(nn), trainingData(data), trainingLabels(labels) {
    // Connect the signal from NeuralNetwork to the new signal in TrainModelWorker
    connect(neuralNetwork, &NeuralNetwork::trainingProgress, this, &TrainModelWorker::trainingProgressUpdate);
}

void TrainModelWorker::run() {
    int epochs = 5; // Or any other desired number of epochs
    std::vector<double> errors; // To capture errors
    int batchSize = 32; // Or any other desired batch size

    neuralNetwork->train(trainingData, trainingLabels, epochs, errors, batchSize);

    emit trainingCompleted("Training complete!");
}

