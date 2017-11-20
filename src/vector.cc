#include "vector.hh"
#include <algorithm>
#include <cassert>
#include "serial.hh"

namespace ml
{

   double Vector::norm() const
   {
      return std::sqrt(norm_square());
   }

   double Vector::norm_square() const
   {
      double res = 0;
      for (const double* it = begin(); it != end(); ++it)
         res += *it * *it;
      return res;
   }

   Vector Vector::normalize() const
   {
      return *this / norm();
   }

   double &Vector::operator[](std::size_t i)
   {
      assert(i < size_);
      return data_[i];
   }

   double Vector::operator[](std::size_t i) const
   {
      assert(i < size_);
      return data_[i];
   }

   std::size_t Vector::size() const
   {
      return size_;
   }

   Vector& Vector::operator+=(const Vector& v)
   {
      assert(size_ == v.size_);
      for (std::size_t i = 0; i < size_; ++i)
         data_[i] += v.data_[i];
      return *this;
   }

   Vector& Vector::operator-=(const Vector& v)
   {
      assert(size_ == v.size_);
      for (std::size_t i = 0; i < size_; ++i)
         data_[i] -= v.data_[i];
      return *this;
   }

   Vector& Vector::operator+=(double x)
   {
      for (std::size_t i = 0; i < size_; ++i)
         data_[i] += x;
      return *this;
   }

   Vector& Vector::operator-=(double x)
   {
      for (std::size_t i = 0; i < size_; ++i)
         data_[i] -= x;
      return *this;
   }

   Vector& Vector::operator*=(double x)
   {
      for (std::size_t i = 0; i < size_; ++i)
         data_[i] *= x;
      return *this;
   }

   Vector& Vector::operator/=(double x)
   {
      for (std::size_t i = 0; i < size_; ++i)
         data_[i] /= x;
      return *this;
   }




   std::ostream &operator<<(std::ostream &os, const Vector &v)
   {
      os << "{";
      for (std::size_t i = 0; i < v.size(); ++i)
      {
         os << v[i];
         if (i + 1 != v.size())
            os << ", ";
      }
      os << "}";
      return os;
   }

   namespace serial
   {
      void write(std::ostream& os, const Vector& v)
      {
         write_bin(os, v.size());
         write_bin(os, v.begin(), v.end());
      }

      void read(std::istream& is, Vector& v)
      {
         std::size_t size;
         read_bin(is, size);
         v = Vector(size);
         read_bin(is, v.begin(), v.end());
      }
   }

   Vector operator+(const Vector& a, const Vector& b)
   {
      assert(a.size() == b.size());
      Vector res(a.size());
      for (std::size_t i = 0; i < a.size(); ++i)
         res[i] = a[i] + b[i];
      return res;
   }

   Vector operator-(const Vector& a, const Vector& b)
   {
      assert(a.size() == b.size());
      Vector res(a.size());
      for (std::size_t i = 0; i < a.size(); ++i)
         res[i] = a[i] - b[i];
      return res;
   }

   Vector operator+(const Vector& a, double b)
   {
      Vector res(a.size());
      for (std::size_t i = 0; i < a.size(); ++i)
         res[i] = a[i] + b;
      return res;
   }

   Vector operator-(const Vector& a, double b)
   {
      Vector res(a.size());
      for (std::size_t i = 0; i < a.size(); ++i)
         res[i] = a[i] - b;
      return res;
   }

   Vector operator*(const Vector& a, double b)
   {
      Vector res(a.size());
      for (std::size_t i = 0; i < a.size(); ++i)
         res[i] = a[i] * b;
      return res;
   }

   Vector operator/(const Vector& a, double b)
   {
      Vector res(a.size());
      for (std::size_t i = 0; i < a.size(); ++i)
         res[i] = a[i] / b;
      return res;
   }

   Vector operator+(double a, const Vector& b)
   {
      Vector res(b.size());
      for (std::size_t i = 0; i < b.size(); ++i)
         res[i] = a + b[i];
      return res;
   }

   Vector operator-(double a, const Vector& b)
   {
      Vector res(b.size());
      for (std::size_t i = 0; i < b.size(); ++i)
         res[i] = a - b[i];
      return res;
   }

   Vector operator*(double a, const Vector& b)
   {
      Vector res(b.size());
      for (std::size_t i = 0; i < b.size(); ++i)
         res[i] = a * b[i];
      return res;
   }

   Vector operator/(double a, const Vector& b)
   {
      Vector res(b.size());
      for (std::size_t i = 0; i < b.size(); ++i)
         res[i] = a / b[i];
      return res;
   }

   Vector hadamard_product(const Vector& a, const Vector& b)
   {
      assert(a.size() == b.size());
      Vector res(a.size());
      const double* ai = a.begin();
      const double* bi = b.begin();
      for (double* it = res.begin(); it != res.end(); ++it)
         *it = *ai++ * *bi++;
      return res;
   }

}
