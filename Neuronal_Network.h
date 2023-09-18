#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Matrix.h"
#include <vector>
#include <string>
#include <QThread>


class NeuralNetwork : public QObject {
    Q_OBJECT

public:
    NeuralNetwork(int inputSize, int hiddenSize, int outputSize, double learningRate);
    std::vector<double> predict(std::vector<double>& input);
    int oneHotPredict(std::vector<double>& input);
    void train(std::vector<std::vector<double>>& inputs, std::vector<int>& labels, int epochs, std::vector<double>& errors, int batchSize);

    static double calcSigmoid(double n);
    void sigmoid(MyMatrix& matrix);
    void save(const std::string& filename) const;
    void load(const std::string& filename);

signals:
    void trainingProgress(QString message);
    void epochUpdates(int epoch);
    void errorReported(double error);

private:
    int inputSize;
    int hiddenSize;
    int outputSize;
    double learningRate;
    MyMatrix weights1;
    MyMatrix biases1;
    MyMatrix weights2;
    MyMatrix biases2;
};

#endif
#ifndef NERONAL_NEURONAL_NETWORK_H
#define NERONAL_NEURONAL_NETWORK_H

#endif //NERONAL_NEURONAL_NETWORK_H
