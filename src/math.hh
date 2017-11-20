#pragma once

#include <cmath>

namespace ml
{

   inline double sigmoid(double z)
   {
      return 1.0 / (1.0 + std::exp(-z));
   }

   inline double sigmoid_prime(double z)
   {
      return std::exp(-z) /  ((1.0 + std::exp(-z)) * (1.0 + std::exp(-z)));
   }

   inline double min(const double* begin, const double* end)
   {
      double res = *begin;
      for (const double* it = begin + 1; it != end; ++it)
         if (*it < res)
            res = *it;
      return res;
   }

   inline double max(const double* begin, const double* end)
   {
      double res = *begin;
      for (const double* it = begin + 1; it != end; ++it)
         if (*it > res)
            res = *it;
      return res;
   }

}
