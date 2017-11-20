#include "random.hh"
#include "date.hh"

namespace ml
{

   Random::Random()
      : Random(Date::now())
   {}

   Random::Random(long seed)
      : mt_(seed)
      , dist_double_(0.0, 1.0)
      , dist_long_()
   {}


   double Random::next_double()
   {
      return dist_double_(mt_);
   }

   double Random::next_double(double max)
   {
      return next_double() * max;
   }

   double Random::next_double(double min, double max)
   {
      return next_double() * (max - min) + min;
   }

   long Random::next_long()
   {
      return dist_long_(mt_);
   }

   long Random::next_long(long max)
   {
      return next_long() % max;
   }

   long Random::next_long(long min, long max)
   {
      return next_long() % (max - min) + min;
   }





   void Random::fill_double(double* begin, double* end)
   {
      for (auto it = begin; it != end; ++it)
         *it = next_double();
   }

   void Random::fill_double(double* begin, double* end, double max)
   {
      for (auto it = begin; it != end; ++it)
         *it = next_double(max);
   }

   void Random::fill_double(double* begin, double* end, double min, double max)
   {
      for (auto it = begin; it != end; ++it)
         *it = next_double(min, max);
   }

}
