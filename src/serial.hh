#pragma once

#include <iostream>
#include <vector>

namespace ml
{

   namespace serial
   {

      template <class T>
      void write_bin(std::ostream& os, const T& val);

      template <class T>
      void read_bin(std::istream& is, T& val);

      template <class Iit>
      void write_bin(std::ostream& os, Iit begin, Iit end);

      template <class Oit>
      void read_bin(std::istream& is, Oit begin, Oit end);

      template <class T>
      void write(std::ostream& os, const std::vector<T>& v);

      template <class T>
      void read(std::istream& is, std::vector<T>& v);

   }

}

#include "serial.hxx"
