#pragma once

#include <vector>
#include "vector.hh"

class MNISTData
{

public:
   MNISTData(const std::string& path);

   const std::vector<ml::Vector>& x_list() const;
   std::vector<ml::Vector> y_list() const;

   static ml::Vector digit_to_vec(int digit);
   static int vec_to_digit(const ml::Vector& v);

private:
   std::vector<ml::Vector> x_;
   std::vector<uint8_t> y_;

};
