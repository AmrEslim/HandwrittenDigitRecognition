#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "Neuronal_Network.h"
#include <QFile>
#include <QString>
#include <fstream>
#include <sstream>
#include <stdexcept>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    connect(ui->trainButton, &QPushButton::clicked, this, &MainWindow::trainModel);
    connect(ui->testButton, &QPushButton::clicked, this, &MainWindow::testModel);
    connect(ui->saveButton, &QPushButton::clicked, this, &MainWindow::saveModel);
    connect(ui->loadButton, &QPushButton::clicked, this, &MainWindow::loadModel);
    try {
        loadEMNISTData("C:/Users/Amr/Desktop/MPT/HandwrittenDigitRecognition/emnist-balanced-train.csv", trainingData, trainingLabels);
        loadEMNISTData("C:/Users/Amr/Desktop/MPT/HandwrittenDigitRecognition/emnist-balanced-test.csv", testData, testLabels);

    } catch (const std::runtime_error& e) {
        // Display the error message to the user
        ui->statusLabel->setText(QString("Error: ") + e.what());
        return; // Exit the constructor or handle the error as appropriate
    }

    neuralNetwork = new NeuralNetwork(784, 128, 47, 0.1);
}

void MainWindow::loadEMNISTData(const std::string& filename, std::vector<std::vector<double>>& data, std::vector<int>& labels) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Error opening file");
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string val;

        // Read the label
        std::getline(iss, val, ',');
        int label = std::stoi(val);
        labels.push_back(label);

        // Read the pixel values
        std::vector<double> pixels;
        while (std::getline(iss, val, ',')) {
            double pixel = std::stod(val) / 255.0; // Normalize pixel value to range 0-1
            pixels.push_back(pixel);
        }
        data.push_back(pixels);
    }

    file.close();
}

std::string MainWindow::labelToChar(int label) {
    char answer = label;
    answer += 48;
    if (label > 9) {
        answer += 7;
    }
    if (label > 35) {
        answer += 6;
    }
    if (label > 37) {
        answer += 1;
    }
    if (label > 42) {
        answer += 5;
    }
    if (label > 43) {
        answer += 2;
    }
    if (label > 45) {
        answer += 1;
    }
    return std::string(1, answer);
}
void MainWindow::test_suite(NeuralNetwork& nn, std::vector<std::vector<double>>& inputs, std::vector<int>& labels, int& results) {
    results = -1;
    int count = 0;
    for (int i = 0; i < labels.size(); i++) {
        int nnGuess = nn.oneHotPredict(inputs[i]);
        if (nnGuess == labels[i]) {
            count++;
        }
    }
    results = count;
}



void MainWindow::trainModel() {
    // TODO: Train the neural network model
}

void MainWindow::testModel() {
    int correctPredictions = 0;
    test_suite(*neuralNetwork, testData, testLabels, correctPredictions);

    double accuracy = static_cast<double>(correctPredictions) / testLabels.size() * 100.0;

    QString resultText = QString("Accuracy: %1% (%2 out of %3 correct)").arg(accuracy).arg(correctPredictions).arg(testLabels.size());
    ui->statusLabel->setText(resultText);
}

void MainWindow::saveModel() {
    // TODO: Save the model to a file
    neuralNetwork->save("path_to_save_model");
    ui->statusLabel->setText("Model saved successfully!");
}

void MainWindow::loadModel() {
    // TODO: Load the model from a file
    neuralNetwork->load("path_to_load_model");
    ui->statusLabel->setText("Model loaded successfully!");
}


MainWindow::~MainWindow()
{
    delete ui;
}

