#pragma once

#include "random.hh"

namespace ml
{

   template <class It>
   void Random::shuffle(It begin, It end)
   {
      if (begin == end)
         return;

      std::size_t n = end - begin;

      for (std::size_t i = n - 1; i < n; --i)
      {
         std::size_t j = next_long(i + 1);
         std::swap(*(begin + i), *(begin + j));
      }
   }

}
