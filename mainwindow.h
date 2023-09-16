#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "Neuronal_Network.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void loadEMNISTData(const std::string& filename, std::vector<std::vector<double>>& data, std::vector<int>& labels);
    std::string labelToChar(int label);

private:
    Ui::MainWindow *ui;
    NeuralNetwork *neuralNetwork;
    std::vector<std::vector<double>> trainingData;
    std::vector<int> trainingLabels;
    std::vector<std::vector<double>> testData;
    std::vector<int> testLabels;
    void test_suite(NeuralNetwork& nn, std::vector<std::vector<double>>& inputs, std::vector<int>& labels, int& results);

private slots:
    void trainModel();
    void testModel();
    void saveModel();
    void loadModel();
};
#endif // MAINWINDOW_H