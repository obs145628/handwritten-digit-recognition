#pragma once

#include "serial.hh"

namespace ml
{

   namespace serial
   {

      template <class T>
      void write_bin(std::ostream& os, const T& val)
      {
         auto n = sizeof(T);
         auto data = reinterpret_cast<const char*>(&val);
         os.write(data, n);
      }

      template <class T>
      void read_bin(std::istream& is, T& val)
      {
         auto n = sizeof(T);
         auto data = reinterpret_cast<char*>(&val);
         is.read(data, n);
      }

      template <class Iit>
      void write_bin(std::ostream& os, Iit begin, Iit end)
      {
         using val_type = decltype(*begin);

         for (Iit it = begin; it != end; ++it)
         {
            write_bin<val_type>(os, *it);
         }
      }

      template <class Oit>
      void read_bin(std::istream& is, Oit begin, Oit end)
      {
         using val_type = decltype(*begin);

         for (Oit it = begin; it != end; ++it)
         {
            read_bin<val_type>(is, *it);
         }
      }

      template <class T>
      void write(std::ostream& os, const std::vector<T>& v)
      {
         write_bin(os, v.size());
         write_bin(os, v.begin(), v.end());
      }

      template <class T>
      void read(std::istream& is, std::vector<T>& v)
      {
         std::size_t size;
         read_bin(is, size);
         v.resize(size);
         read_bin(is, v.begin(), v.end());
      }



   }

}
