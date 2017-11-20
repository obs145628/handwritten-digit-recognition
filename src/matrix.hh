#pragma once

#include <cstddef>
#include <iostream>
#include <vector>
#include "vector.hh"

namespace ml
{

   class Matrix
   {
   public:

      Matrix();
      Matrix(std::size_t rows, std::size_t cols);
      Matrix(std::size_t rows, std::size_t cols,
             const std::initializer_list<double>& vals);

      template <class It>
      Matrix(std::size_t rows, std::size_t cols, It begin, It end);

      Matrix(const Matrix& m);
      Matrix(Matrix&& m);
      ~Matrix();

      Matrix& operator=(const Matrix& m);
      Matrix& operator=(Matrix&& m);

      double* begin();
      const double* begin() const;
      double* end();
      const double* end() const;

      std::size_t rows() const;
      std::size_t cols() const;
      double& operator()(std::size_t row, std::size_t col);
      const double& operator()(std::size_t row, std::size_t col) const;
      double& at(std::size_t row, std::size_t col);
      const double& at(std::size_t row, std::size_t col) const;
      void fill(double val);
      void replace(Matrix& temp);

      Vector solve_lower(const Vector& b);
      Vector solve_upper(const Vector& b);

      Matrix transpose() const;
      Matrix inverse() const;


      bool plu_decomposition(Matrix& p, Matrix& l, Matrix& u,
                             std::vector<std::size_t>& pv,
                             bool& even_permuts) const;
      Vector plu_solve(const Vector& b) const;
      Matrix plu_solve(const Matrix& b) const;
      Matrix plu_inverse() const;

      Matrix& operator+=(const Matrix& m);
      Matrix& operator-=(const Matrix& m);
      Matrix& operator+=(double x);
      Matrix& operator-=(double x);
      Matrix& operator*=(double x);
      Matrix& operator/=(double x);

      static Matrix with(std::size_t rows, std::size_t cols, double val);
      static Matrix zero(std::size_t rows, std::size_t cols);
      static Matrix id(std::size_t n);
      static Matrix mul(const Matrix& a, const Matrix& b);
      static Vector mul(const Matrix& a, const Vector& b);
      static Vector mul(const Vector& a, const Matrix& b);

      static void mul(const Matrix& a, const Vector& b, Vector& out);

   private:
      std::size_t rows_;
      std::size_t cols_;
      double* data_;
   };

   std::ostream& operator<<(std::ostream& os, const Matrix& m);

   namespace serial
   {
      void write(std::ostream& os, const Matrix& m);
      void read(std::istream& is, Matrix& m);
   }

   Matrix operator+(const Matrix& a, const Matrix& b);
   Matrix operator-(const Matrix& a, const Matrix& b);
   Matrix operator+(const Matrix& a, double b);
   Matrix operator-(const Matrix& a, double b);
   Matrix operator*(const Matrix& a, double b);
   Matrix operator/(const Matrix& a, double b);
   Matrix operator+(double a, const Matrix& b);
   Matrix operator-(double a, const Matrix& b);
   Matrix operator*(double a, const Matrix& b);
   Matrix operator/(double a, const Matrix& b);
   Matrix operator-(const Matrix& a);


   Matrix hadamard_product(const Matrix& a, const Matrix& b);
   Matrix outer_product(const Vector& a, const Vector& b);

   inline Matrix sigmoid(const Matrix& v);
   inline Matrix sigmoid_prime(const Matrix& v);

}

#include "matrix.hxx"
