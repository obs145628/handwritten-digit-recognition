#pragma once

#include "neural-network.hh"
#include "date.hh"

namespace ml
{

   template <class Test>
   void NeuralNetwork::train(int epochs, std::size_t batch_size, double eta,
                             std::vector<Vector>& x_train,
                             std::vector<Vector>& y_train,
                             const std::vector<Vector>& x_test,
                             const std::vector<Vector>& y_test,
                             Test test)
   {

      std::cout << "No training: ";
      eval(x_test, y_test, test);
      for (int i = 1; i <= epochs; ++i)
      {
         auto t = Date::now();
         sgd(x_train, y_train, batch_size, eta);
         auto dur = Date::now() - t;
         std::cout << "Epoch " << i << " (" << dur  << "ms): ";
         eval(x_test, y_test, test);
      }

   }

   template <class Test>
   void NeuralNetwork::eval(const std::vector<Vector>& x_test,
                            const std::vector<Vector>& y_test,
                            Test test)
   {
      std::size_t succ = 0;
      for (std::size_t i = 0; i < x_test.size(); ++i)
      {
         auto y_hat = forward(x_test[i]);
         succ += !!test(y_hat, y_test[i]);
      }

      double per = (succ * 100.0) / x_test.size();
      std::cout << succ << " / " << x_test.size() << " (" << per << "%)\n";
   }

}
