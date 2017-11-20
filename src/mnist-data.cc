#include "mnist-data.hh"
#include <fstream>

namespace
{
   constexpr int IMG_W = 28;
   constexpr int IMG_H = 28;
}


MNISTData::MNISTData(const std::string& path)
{
   std::ifstream is(path);
   uint8_t buffer[IMG_W * IMG_H + 1];

   while (is.good())
   {
      is.read(reinterpret_cast<char*>(buffer), sizeof(buffer));
      y_.push_back(buffer[0]);

      ml::Vector lx(IMG_W * IMG_H);
      for (std::size_t i = 0; i < IMG_W * IMG_H; ++i)
      {
         double val = buffer[i + 1] / 255.0;
         lx[i] = val;
      }

      x_.push_back(lx);
   }

   x_.pop_back();
   y_.pop_back();
}

const std::vector<ml::Vector>& MNISTData::x_list() const
{
   return x_;
}

std::vector<ml::Vector> MNISTData::y_list() const
{
   std::vector<ml::Vector> res(x_.size());
   for (std::size_t i = 0; i < x_.size(); ++i)
      res[i] = digit_to_vec(y_[i]);
   return res;

}

ml::Vector MNISTData::digit_to_vec(int digit)
{
   auto v = ml::Vector::zero(10);
   v[digit] = 1;
   return v;
}

int MNISTData::vec_to_digit(const ml::Vector& v)
{
   std::size_t max = 0;
   for (std::size_t i = 1; i < 10; ++i)
      if (v[i] > v[max])
         max = i;
   return max;

}
