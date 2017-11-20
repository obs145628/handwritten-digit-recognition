#pragma once

#include <random>

namespace ml
{

   class Random
   {
   public:

      Random();
      Random(long seed);

      double next_double();
      double next_double(double max);
      double next_double(double min, double max);

      long next_long();
      long next_long(long max);
      long next_long(long min, long max);

      void fill_double(double* begin, double* end);
      void fill_double(double* begin, double* end, double max);
      void fill_double(double* begin, double* end, double min, double max);

      template <class It>
      void shuffle(It begin, It end);

   private:
      std::mt19937 mt_;
      std::uniform_real_distribution<double> dist_double_;
      std::uniform_int_distribution<long> dist_long_;
   };

}

#include "random.hxx"
