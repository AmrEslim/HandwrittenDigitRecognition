
#include "Matrix.h"
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <forward_list>

/// Defines an empty 0x0 Matrix. Should not be called.
MyMatrix::MyMatrix(): m_rows(0), m_cols(0){}

/// Constructs a matrix with the specified number of columns and rows. Default value is 0.0 (doubles)
/// @param rows The integer number of rows
/// @param cols The integer number of coloums
MyMatrix::MyMatrix(int rows, int cols)
        : m_rows(rows), m_cols(cols), m_data(rows * cols){

}

/// Constructs a copy of the matrix
/// @param other The other matrix to copy
MyMatrix::MyMatrix(const MyMatrix& other)
        : m_rows(other.m_rows), m_cols(other.m_cols), m_data(other.m_data){
}

/// Turns a vector into a 1 dimensional vector
/// @param values The vector of doubles to turn into a matrix
/// @param isColumn Boolean value, if true the values will be turned into a matrix with only one column. If false the matrix will be one row instead
MyMatrix::MyMatrix(const std::vector<double>& values, bool isColumn)
{
    if (isColumn) {
        m_rows = values.size();
        m_cols = 1;
        m_data = values;
    }
    else {
        m_rows = 1;
        m_cols = values.size();
        m_data = values;
    }
}

/// Returns the number of rows the matrix has
int MyMatrix::rows() const{
    return m_rows;
}

/// Returns the number of colums the matrix has
int MyMatrix::columns() const{
    return m_cols;
}

double MyMatrix::operator()(int row, int col) const{
    return m_data[row * m_cols + col];
}

double& MyMatrix::operator()(int row, int col){
    return m_data[row * m_cols + col];
}
/// sums of two matrices
double MyMatrix::sum() const{
    double total = 0.0;
    for (int i = 0; i < rows(); ++i) {
        for (int j = 0; j < columns(); ++j) {
            total += (*this)(i, j);
        }
    }
    return total;
}


/// Adds this matrix with the other matrix
MyMatrix MyMatrix::operator+(const MyMatrix& other) const{
    MyMatrix result(m_rows, m_cols);
    for (int i = 0; i < m_rows * m_cols; ++i) {
        result.m_data[i] = m_data[i] + other.m_data[i];
    }
    return result;
}

std::vector<std::vector<double>> MyMatrix::toList() const {
    std::vector<std::vector<double>> list(rows());
    for (int i = 0; i < rows(); ++i) {
        for (int j = 0; j < columns(); ++j) {
            list[i].push_back((*this)(i, j));
        }
    }
    return list;
}

void MyMatrix::fromList(const std::vector<std::vector<double>>& list) {
    if (rows() != list.size() || columns() != list[0].size()) {
        // Throw an error or handle the size mismatch in some other way
        throw std::runtime_error("Size mismatch between matrix and provided list");
    }

    for (int i = 0; i < list.size(); ++i) {
        for (int j = 0; j < list[i].size(); ++j) {
            (*this)(i, j) = list[i][j];
        }
    }
}




/// Substracts the other matrix from the values of this matrix
MyMatrix MyMatrix::operator-(const MyMatrix& other) const{
    MyMatrix result(m_rows, m_cols);
    for (int i = 0; i < m_rows * m_cols; ++i) {
        result.m_data[i] = m_data[i] - other.m_data[i];
    }
    return result;
}

/// Matrix multiplication of the matrices (use elementWiseProduct for normal multiplication)
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

/// Multiplies each value of the matrix with the scalar
MyMatrix MyMatrix::operator*(double scalar) const {
    MyMatrix result(m_rows, m_cols);
    for (int i = 0; i < m_rows; i++) {
        for (int j = 0; j < m_cols; j++) {
            result(i, j) = (*this)(i, j) * scalar;
        }
    }
    return result;
}

/// In place addition of the matrix
void MyMatrix::operator+=(const MyMatrix& other){
    for (int i = 0; i < m_rows * m_cols; ++i) {
        m_data[i] += other.m_data[i];
    }
}

/// In place subtraction of the matrix
void MyMatrix::operator-=(const MyMatrix& other){
    for (int i = 0; i < m_rows * m_cols; ++i) {
        m_data[i] -= other.m_data[i];
    }
}


/// In place matrix multiplication of the other matrix (use elementWiseProduct for normal multiplication)
void MyMatrix::operator*=(const MyMatrix& other){
    *this = *this * other;
}


/// Randomizes the values of the matrix with a double
/// @param minVal The minimum value possible
/// @param maxVal The maximum value possible
void MyMatrix::randomize(double minVal, double maxVal){
    std::srand(std::time(nullptr));
    double range = maxVal - minVal;
    for (int i = 0; i < m_rows * m_cols; ++i) {
        m_data[i] = (std::rand() / static_cast<double>(RAND_MAX)) * range + minVal;
    }
}

/// Returns a copy of the matrix that is transposed (mirrored diagonally)
/// The resulting matrix has as many rows as the initial matrix has columns and vice versa
MyMatrix MyMatrix::transpose() const {
    MyMatrix result(m_cols, m_rows);
    for (int i = 0; i < m_rows; i++) {
        for (int j = 0; j < m_cols; j++) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

/// Returns a single column of the matrix as a vector of doubles
/// @param colIndex The index of the column to get as a vector
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

/// Multiplies each element of the matrix with the corresponding element in the other matrix
/// @param other The other matrix to multiply with. THe matrices must be the same size
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


/// Sets all the elements of the matrix to this value
/// @param value The value to be set
void MyMatrix::setAll(double value) {
    for (int i = 0; i < m_rows; i++) {
        for (int j = 0; j < m_cols; j++) {
            (*this)(i, j) = value;
        }
    }
}

void MyMatrix::resize(int newRows, int newCols) {
    m_rows = newRows;
    m_cols = newCols;
    m_data.resize(m_rows * m_cols);
}