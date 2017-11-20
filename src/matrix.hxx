#include "matrix.hh"
#include <cassert>

namespace ml
{

   inline Matrix::Matrix()
      : rows_(0)
      , cols_(0)
      , data_(nullptr)
   {}

   inline Matrix::Matrix(std::size_t rows, std::size_t cols)
      : rows_(rows)
      , cols_(cols)
      , data_(new double[rows * cols])
   {}

   inline Matrix::Matrix(std::size_t rows, std::size_t cols,
                         const std::initializer_list<double> &vals)
      : Matrix(rows, cols)
   {
      assert(vals.size() == rows * cols);
      std::copy(vals.begin(), vals.end(), begin());
   }

   template <class It>
   Matrix::Matrix(std::size_t rows, std::size_t cols, It begin, It end)
      : Matrix(rows, cols)
   {
      std::copy(begin, end, this->begin());
   }

   inline Matrix::Matrix(const Matrix &m)
      : Matrix(m.rows_, m.cols_)
   {
      std::copy(m.begin(), m.end(), begin());
   }

   inline Matrix::Matrix(Matrix&& m)
      : rows_(m.rows_)
      , cols_(m.cols_)
      , data_(m.data_)
   {
      m.data_ = nullptr;
   }

   inline Matrix::~Matrix()
   {
      delete data_;
   }

   inline Matrix& Matrix::operator=(const Matrix &m)
   {
      if (&m == this)
         return *this;

      delete data_;
      rows_ = m.rows_;
      cols_ = m.cols_;
      data_ = new double[rows_ * cols_];
      std::copy(m.begin(), m.end(), begin());
      return *this;
   }

   inline Matrix& Matrix::operator=(Matrix&& m)
   {
      if (&m == this)
         return *this;

      delete data_;
      rows_ = m.rows_;
      cols_ = m.cols_;
      data_ = m.data_;
      m.data_ = nullptr;
      return *this;
   }

   inline double* Matrix::begin()
   {
      return data_;
   }

   inline const double* Matrix::begin() const
   {
      return data_;
   }

   inline double* Matrix::end()
   {
      return data_ + rows_ * cols_;
   }

   inline const double* Matrix::end() const
   {
      return data_ + rows_ * cols_;
   }

   inline std::size_t Matrix::rows() const
   {
      return rows_;
   }

   inline std::size_t Matrix::cols() const
   {
      return cols_;
   }

   inline double& Matrix::operator()(std::size_t row, std::size_t col)
   {
      return at(row, col);
   }

   inline const double& Matrix::operator()(std::size_t row,
                                           std::size_t col) const
   {
      return at(row, col);
   }

   inline double& Matrix::at(std::size_t row, std::size_t col)
   {
      assert(row < rows_);
      assert(col < cols_);
      return data_[row * cols_ + col];
   }

   inline const double& Matrix::at(std::size_t row, std::size_t col) const
   {
      assert(row < rows_);
      assert(col < cols_);
      return data_[row * cols_ + col];
   }




   inline Matrix Matrix::with(std::size_t rows, std::size_t cols, double val)
   {
      Matrix m(rows, cols);
      std::fill(m.begin(), m.end(), val);
      return m;
   }

   inline Matrix Matrix::zero(std::size_t rows, std::size_t cols)
   {
      return with(rows, cols, 0);
   }


   inline Matrix sigmoid(const Matrix& v)
   {
      Matrix res(v.rows(), v.cols());
      double* out = res.begin();
      for (const double* it = v.begin(); it != v.end(); ++it)
         *(out++) = sigmoid(*it);
      return res;
   }

   inline Matrix sigmoid_prime(const Matrix& v)
   {
      Matrix res(v.rows(), v.cols());
      double* out = res.begin();
      for (const double* it = v.begin(); it != v.end(); ++it)
         *(out++) = sigmoid_prime(*it);
      return res;
   }

}
