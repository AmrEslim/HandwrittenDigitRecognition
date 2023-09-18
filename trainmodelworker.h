#ifndef TRAINMODELWORKER_H
#define TRAINMODELWORKER_H

#include <QThread>
#include "Neuronal_Network.h"

class TrainModelWorker : public QThread {
    Q_OBJECT

public:
    TrainModelWorker(NeuralNetwork* nn, const std::vector<std::vector<double>>& data, const std::vector<int>& labels);
signals:
    void trainingProgressUpdate(const QString& message);
    void trainingCompleted(QString message);

protected:
    void run() override;

private:
    NeuralNetwork* neuralNetwork;
    std::vector<std::vector<double>> trainingData;   // Remove the reference and const qualifiers
    std::vector<int> trainingLabels;                 // Remove the reference and const qualifiers
};

#endif // TRAINMODELWORKER_H
