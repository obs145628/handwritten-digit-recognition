#pragma once

#include <cstddef>
#include <iostream>
#include "math.hh"


namespace ml
{

   class Vector
   {
   public:
      Vector();
      Vector(std::size_t size);
      Vector(const std::initializer_list<double>& vals);
      Vector(const Vector& v);
      Vector(Vector&& v);
      ~Vector();

      Vector& operator=(const Vector& v);
      Vector& operator=(Vector&& v);

      void assign(std::size_t size);


      double* begin();
      const double* begin() const;
      double* end();
      const double* end() const;
      double& operator[](std::size_t i);
      double operator[](std::size_t i) const;

      std::size_t size() const;

      double norm() const;
      double norm_square() const;
      Vector normalize() const;

      Vector& operator+=(const Vector& v);
      Vector& operator-=(const Vector& v);
      Vector& operator+=(double x);
      Vector& operator-=(double x);
      Vector& operator*=(double x);
      Vector& operator/=(double x);

      static Vector with(std::size_t n, double val);
      static Vector zero(std::size_t n);

   private:
      std::size_t size_;
      double* data_;
   };

   std::ostream& operator<<(std::ostream& os, const Vector& v);

   namespace serial
   {
      void write(std::ostream& os, const Vector& v);
      void read(std::istream& is, Vector& v);
   }

   Vector operator+(const Vector& a, const Vector& b);
   Vector operator-(const Vector& a, const Vector& b);
   Vector operator+(const Vector& a, double b);
   Vector operator-(const Vector& a, double b);
   Vector operator*(const Vector& a, double b);
   Vector operator/(const Vector& a, double b);
   Vector operator+(double a, const Vector& b);
   Vector operator-(double a, const Vector& b);
   Vector operator*(double a, const Vector& b);
   Vector operator/(double a, const Vector& b);

   Vector hadamard_product(const Vector& a, const Vector& b);


   Vector sigmoid(const Vector& v);
   Vector sigmoid_prime(const Vector& v);

}

#include "vector.hxx"
