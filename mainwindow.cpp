#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "Neuronal_Network.h"
#include "trainmodelworker.h"
#include "qcustomplot.h"
#include <QFile>
#include <QString>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <QTimer>
#include <QImage>
#include <QPixmap>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    connect(ui->trainButton, &QPushButton::clicked, this, &MainWindow::trainModel);
    connect(ui->testButton, &QPushButton::clicked, this, &MainWindow::testModel);
    connect(ui->saveButton, &QPushButton::clicked, this, &MainWindow::saveModel);
    connect(ui->loadButton, &QPushButton::clicked, this, &MainWindow::loadModel);

    worker = nullptr;
    isTraining = false;
    isTestingPeriodically = false;
    testingIndex = 0;

    testingTimer = new QTimer(this);
    connect(testingTimer, &QTimer::timeout, this, &MainWindow::performPeriodicTest);
    // Inform the user that the dataset is loading
    //ui->statusLabel->setText("Loading dataset...");

    // Use QTimer to defer the loading of data
    QTimer::singleShot(0, this, &MainWindow::loadData);

    neuralNetwork = new NeuralNetwork(784, 128, 47, 0.1);
}

void MainWindow::loadData() {
    try {
        //ui->statusLabel->setText("Please wait the dataset is being loaded");
        loadEMNISTData("C:/Users/Amr/Desktop/MPT/HandwrittenDigitRecognition/emnist-balanced-train.csv", trainingData, trainingLabels);
        loadEMNISTData("C:/Users/Amr/Desktop/MPT/HandwrittenDigitRecognition/emnist-balanced-test.csv", testData, testLabels);
        // Inform the user that the dataset has been loaded
        ui->statusLabel->setText("Dataset loaded successfully!");
    } catch (const std::runtime_error& e) {
        // Display the error message to the user
        ui->statusLabel->setText(QString("Error: ") + e.what());
        return; // Exit the method or handle the error as appropriate
    }
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
void MainWindow::performPeriodicTest() {
    if (testingIndex >= testData.size()) {
        testingIndex = 0;
    }

    std::vector<double> image = testData[testingIndex];
    int label = testLabels[testingIndex];
    int networkGuessLabel = neuralNetwork->oneHotPredict(image);

    // Convert the 'image' vector to QImage and set it to the drawingCanvas label.
    QImage qImage = vectorToQImage(image);
    ui->drawingCanvas->setPixmap(QPixmap::fromImage(qImage).scaled(ui->drawingCanvas->size(), Qt::KeepAspectRatio));

    // Update the carLabel and prediction label
    std::string actualChar = labelToChar(label);
    std::string predictedChar = labelToChar(networkGuessLabel);

    ui->carLabel->setText(QString::fromStdString(actualChar));
    ui->prediction->setText(QString::fromStdString(predictedChar));

    testingIndex++;
}
QImage MainWindow::vectorToQImage(const std::vector<double>& image) {
    int width = 28; // As per EMNIST data
    int height = 28; // As per EMNIST data

    QImage qImage(width, height, QImage::Format_Grayscale8);

    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            int value = static_cast<int>(image[y * width + x] * 255);
            qImage.setPixel(x, y, qRgb(value, value, value));
        }
    }

    return qImage;
}
void MainWindow::trainModel() {
    if (!isTraining) {
        ui->trainButton->setText("Starting Training process...");
        worker = new TrainModelWorker(neuralNetwork, trainingData, trainingLabels);

        // Connect the training completed signal to handle completion
        connect(worker, &TrainModelWorker::trainingCompleted, this, &MainWindow::onTrainingCompleted);

        // Connect the training epoch number signal to the progressBar
        connect(worker, &TrainModelWorker::epochUpdate, this, &MainWindow::updateTrainingProgress);

        ///////
        connect(worker, &TrainModelWorker::trainingErrorReported, this, &MainWindow::updateErrorGraph);


        // Connect the training progress update signal to update the status label
        connect(worker, &TrainModelWorker::trainingProgressUpdate, this, [this](const QString& message) {
            ui->statusLabel->setText(message);
        });


        // Ensure the worker is deleted once it finishes execution
        connect(worker, &TrainModelWorker::finished, worker, &QObject::deleteLater);

        worker->start();
        ui->trainButton->setText("Stop Training");
        isTraining = true;
    } else {
        stopTraining();
    }
}


void MainWindow::onTrainingCompleted(QString message) {
    ui->statusLabel->setText(message);
    ui->trainButton->setText("Start Training");
    isTraining = false;
}

void MainWindow::stopTraining() {
    if (worker) {
        worker->terminate(); // forcefully stops the thread (use with caution)
        worker->wait();      // waits for the thread to truly finish
        delete worker;       // clean up
        worker = nullptr;
    }
    ui->statusLabel->setText("Training stopped.");
    ui->trainButton->setText("Start Training");
    isTraining = false;
}

void MainWindow::testModel() {
    // Inform the user that the Algorithm is testing
    ui->statusLabel->setText("testing the detection Algorithm against the testing Dataset");
    int correctPredictions = 0;
    test_suite(*neuralNetwork, testData, testLabels, correctPredictions);

    double accuracy = static_cast<double>(correctPredictions) / testLabels.size() * 100.0;
    QString resultText = QString("TESTING IS DONE. Accuracy: %1% (%2 out of %3 correct)").arg(accuracy).arg(correctPredictions).arg(testLabels.size());
    ui->statusLabel->setText(resultText);
}

void MainWindow::saveModel() {
    // TODO: Save the model to a file
    ui->statusLabel->setText("Model is being saved.....");
    neuralNetwork->save("path_to_save_model");
    ui->statusLabel->setText("Model saved successfully!");
}

void MainWindow::loadModel() {
    // TODO: Load the model from a file
    ui->statusLabel->setText("Model is being loaded.....");
    neuralNetwork->load("path_to_load_model");
    ui->statusLabel->setText("Model loaded successfully!");
}
void MainWindow::updateTrainingProgress(int epoch) {
    ui->trainingProgressBar->setValue(epoch);
}
void MainWindow::on_periodicTest_clicked() {
    isTestingPeriodically = !isTestingPeriodically;
    if (isTestingPeriodically) {
        testingTimer->start(1000); // start testing every second
        ui->periodicTest->setText("Stop Periodic Test");
    } else {
        testingTimer->stop();
        ui->periodicTest->setText("Start Periodic Test");
    }
}
void MainWindow::updateErrorGraph(double error) {
    // Assuming you have a QCustomPlot member called errorPlot
    static QVector<double> xData, yData;
    xData.append(static_cast<double>(xData.size())); // assuming X-axis is just the index/epoch
    yData.append(error);

    ui->errorPlot->addGraph();
    ui->errorPlot->graph(0)->setData(xData, yData);

    // Optionally set axes labels, ranges, etc.

    ui->errorPlot->replot();
}





MainWindow::~MainWindow()
{
    delete ui;
}

