#include "Matrix.h"
#include <cstdlib>
#include <ctime>
#include <stdexcept>

// Default constructor for an empty matrix
MyMatrix::MyMatrix(): m_rows(0), m_cols(0){}

// Constructor to initialize a matrix with the given number of rows and columns
MyMatrix::MyMatrix(int rows, int cols)
    : m_rows(rows), m_cols(cols), m_data(rows * cols){}

// Copy constructor to create a copy of an existing matrix
MyMatrix::MyMatrix(const MyMatrix& other)
    : m_rows(other.m_rows), m_cols(other.m_cols), m_data(other.m_data){}

// Constructor to initialize a matrix from a given vector
MyMatrix::MyMatrix(const std::vector<double>& values, bool isColumn)
{
    if (isColumn) {
        m_rows = values.size();
        m_cols = 1;
        m_data = values;
    } else {
        m_rows = 1;
        m_cols = values.size();
        m_data = values;
    }
}

// Static function to create a matrix with all elements set to one
MyMatrix MyMatrix::allOnes(int rows, int cols) {
    MyMatrix result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = 1.0;
        }
    }
    return result;
}

// Function to get the number of rows in the matrix
int MyMatrix::rows() const{
    return m_rows;
}

// Function to get the number of columns in the matrix
int MyMatrix::columns() const{
    return m_cols;
}

// Overloaded () operator to access matrix elements (read-only)
double MyMatrix::operator()(int row, int col) const{
    return m_data[row * m_cols + col];
}

// Overloaded () operator to access matrix elements (read-write)
double& MyMatrix::operator()(int row, int col){
    return m_data[row * m_cols + col];
}

// Function to calculate the sum of all elements in the matrix
double MyMatrix::sum() const{
    double total = 0.0;
    for (int i = 0; i < rows(); ++i) {
        for (int j = 0; j < columns(); ++j) {
            total += (*this)(i, j);
        }
    }
    return total;
}

// Overloaded + operator for matrix addition
MyMatrix MyMatrix::operator+(const MyMatrix& other) const{
    MyMatrix result(m_rows, m_cols);
    for (int i = 0; i < m_rows * m_cols; ++i) {
        result.m_data[i] = m_data[i] + other.m_data[i];
    }
    return result;
}

// Function to convert the matrix to a 2D vector
std::vector<std::vector<double>> MyMatrix::toList() const {
    std::vector<std::vector<double>> list(rows());
    for (int i = 0; i < rows(); ++i) {
        for (int j = 0; j < columns(); ++j) {
            list[i].push_back((*this)(i, j));
        }
    }
    return list;
}

// Function to initialize the matrix from a 2D vector
void MyMatrix::fromList(const std::vector<std::vector<double>>& list) {
    if (rows() != list.size() || columns() != list[0].size()) {
        throw std::runtime_error("Size mismatch between matrix and provided list");
    }
    for (int i = 0; i < list.size(); ++i) {
        for (int j = 0; j < list[i].size(); ++j) {
            (*this)(i, j) = list[i][j];
        }
    }
}

// Overloaded - operator for matrix subtraction
MyMatrix MyMatrix::operator-(const MyMatrix& other) const{
    MyMatrix result(m_rows, m_cols);
    for (int i = 0; i < m_rows * m_cols; ++i) {
        result.m_data[i] = m_data[i] - other.m_data[i];
    }
    return result;
}

// Overloaded * operator for matrix multiplication
MyMatrix MyMatrix::operator*(const MyMatrix& other) const{
    MyMatrix result(m_rows, other.m_cols);
    for (int i = 0; i < m_rows; ++i) {
        for (int j = 0; j < other.m_cols; ++j) {
            double sum = 0.0;
            for (int k = 0; k < m_cols; ++k) {
                sum += m_data[i * m_cols + k] * other.m_data[k * other.m_cols + j];
            }
            result.m_data[i * other.m_cols + j] = sum;
        }
    }
    return result;
}

// Overloaded * operator for scalar multiplication
MyMatrix MyMatrix::operator*(double scalar) const {
    MyMatrix result(m_rows, m_cols);
    for (int i = 0; i < m_rows; i++) {
        for (int j = 0; j < m_cols; j++) {
            result(i, j) = (*this)(i, j) * scalar;
        }
    }
    return result;
}

// Overloaded += operator for in-place matrix addition
void MyMatrix::operator+=(const MyMatrix& other){
    for (int i = 0; i < m_rows * m_cols; ++i) {
        m_data[i] += other.m_data[i];
    }
}

// Overloaded -= operator for in-place matrix subtraction
void MyMatrix::operator-=(const MyMatrix& other){
    for (int i = 0; i < m_rows * m_cols; ++i) {
        m_data[i] -= other.m_data[i];
    }
}

// Overloaded *= operator for in-place matrix multiplication
void MyMatrix::operator*=(const MyMatrix& other){
    *this = *this * other;
}

// Function to randomize the matrix with values between minVal and maxVal
void MyMatrix::randomize(double minVal, double maxVal){
    std::srand(std::time(nullptr));
    double range = maxVal - minVal;
    for (int i = 0; i < m_rows * m_cols; ++i) {
        m_data[i] = (std::rand() / static_cast<double>(RAND_MAX)) * range + minVal;
    }
}

// Function to get the transpose of the matrix
MyMatrix MyMatrix::transpose() const {
    MyMatrix result(m_cols, m_rows);
    for (int i = 0; i < m_rows; i++) {
        for (int j = 0; j < m_cols; j++) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

// Function to get a single column of the matrix as a vector
std::vector<double> MyMatrix::getColumnAsVector(int colIndex) const{
    if (colIndex < 0 || colIndex >= m_cols) {
        throw std::out_of_range("Invalid column index");
    }
    std::vector<double> column(m_rows);
    for (int i = 0; i < m_rows; ++i) {
        column[i] = m_data[i * m_cols + colIndex];
    }
    return column;
}

// Function to get the element-wise product of two matrices
MyMatrix MyMatrix::elementWiseProduct(const MyMatrix& other) const{
    if (m_rows != other.m_rows || m_cols != other.m_cols) {
        throw std::invalid_argument("Matrices must have the same dimensions");
    }
    MyMatrix result(m_rows, m_cols);
    for (int i = 0; i < m_rows; ++i) {
        for (int j = 0; j < m_cols; ++j) {
            result(i, j) = (*this)(i, j) * other(i, j);
        }
    }
    return result;
}

// Function to set all elements of the matrix to a given value
void MyMatrix::setAll(double value) {
    for (int i = 0; i < m_rows; i++) {
        for (int j = 0; j < m_cols; j++) {
            (*this)(i, j) = value;
        }
    }
}

// Function to resize the matrix to new dimensions
void MyMatrix::resize(int newRows, int newCols) {
    m_rows = newRows;
    m_cols = newCols;
    m_data.resize(m_rows * m_cols);
}
