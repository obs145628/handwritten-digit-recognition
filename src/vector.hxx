#pragma once

namespace ml
{

   inline Vector::Vector()
      : size_(0)
      , data_(nullptr)
   {}

   inline Vector::Vector(std::size_t size)
      : size_(size)
      , data_(new double[size])
   {}

   inline Vector::Vector(const std::initializer_list<double>& vals)
      : Vector(vals.size())
   {
      std::copy(vals.begin(), vals.end(), begin());
   }

   inline Vector::Vector(const Vector &v)
      : Vector(v.size_)
   {
      std::copy(v.begin(), v.end(), begin());
   }

   inline Vector::Vector(Vector&& v)
      : size_(v.size_)
      , data_(v.data_)
   {
      v.data_ = nullptr;
   }

   inline Vector::~Vector()
   {
      delete data_;
   }

   inline Vector& Vector::operator=(const Vector& v)
   {
      if (&v == this)
         return *this;

      if (v.size() > size())
      {
         delete data_;
         data_ = new double[v.size()];
      }

      size_ = v.size();
      std::copy(v.begin(), v.end(), begin());
      return *this;
   }

   inline Vector& Vector::operator=(Vector&& v)
   {
      if (&v == this)
         return *this;

      delete data_;
      size_ = v.size_;
      data_ = v.data_;
      v.data_ = nullptr;
      return *this;
   }

   inline void Vector::assign(std::size_t size)
   {
      if (size > size_)
      {
         delete data_;
         data_ = new double[size];
      }

      size_ = size;
   }

   inline double* Vector::begin()
   {
      return data_;
   }

   inline const double* Vector::begin() const
   {
      return data_;
   }

   inline double* Vector::end()
   {
      return data_ + size_;
   }

   inline const double* Vector::end() const
   {
      return data_ + size_;
   }




   inline Vector Vector::with(std::size_t n, double val)
   {
      Vector v(n);
      std::fill(v.begin(), v.end(), val);
      return v;
   }

   inline Vector Vector::zero(std::size_t n)
   {
      return with(n, 0);
   }





   inline Vector sigmoid(const Vector& v)
   {
      Vector res(v.size());
      double* out = res.begin();
      for (const double* it = v.begin(); it != v.end(); ++it)
      {
         *(out++) = sigmoid(*it);
      }
      return res;
   }

   inline Vector sigmoid_prime(const Vector& v)
   {
      Vector res(v.size());
      double* out = res.begin();
      for (const double* it = v.begin(); it != v.end(); ++it)
      {
         *(out++) = sigmoid_prime(*it);
      }
      return res;
   }

}
