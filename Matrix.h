#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <functional>

class MyMatrix {
private:
    int m_rows;
    int m_cols;
    std::vector<double> m_data;

public:
    MyMatrix();
    MyMatrix(int rows, int cols);
    MyMatrix(const MyMatrix& other);
    MyMatrix(const std::vector<double>& values, bool isColumn = true);

    int rows() const;
    int columns() const;
    double sum() const;
    double operator()(int row, int col) const;
    double& operator()(int row, int col);

    MyMatrix operator+(const MyMatrix& other) const;
    MyMatrix operator-(const MyMatrix& other) const;
    MyMatrix operator*(const MyMatrix& other) const;
    MyMatrix operator*(double scalar) const;

    void operator+=(const MyMatrix& other);
    void operator-=(const MyMatrix& other);
    void operator*=(const MyMatrix& other);
    void resize(int newRows, int newCols);

    void randomize(double minVal, double maxVal);
    MyMatrix transpose() const;
    std::vector<std::vector<double>> toList() const;
    void fromList(const std::vector<std::vector<double>>& list);
    std::vector<double> getColumnAsVector(int colIndex) const;
    MyMatrix elementWiseProduct(const MyMatrix& other) const;
    void setAll(double value);
    static MyMatrix allOnes(int rows, int cols);
};

#endif // MATRIX_H
#ifndef NERONAL_MATRIX_H
#define NERONAL_MATRIX_H

#endif //NERONAL_MATRIX_H
