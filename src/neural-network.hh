#pragma once

#include <cstddef>
#include <vector>
#include "matrix.hh"
#include "random.hh"

namespace ml
{

   class NeuralNetwork
   {

   public:

      /**
       * Build a new neural network
       * Sizes contains the numbers of neurons for each sizes
       * first and last are respectively the number of input and output neurons
       */
      NeuralNetwork(const std::vector<std::size_t>& sizes);

      /**
       * Forward some input values into the network and returns the output
       * size(x) == number of input neurons
       * Each element of x must be between 0 and 1
       *
       * Return output vector y_hat
       * size(y_hat) == number of output neurons
       * Each element of y_hat is between 0 and 1
       */
      Vector forward(const Vector& x);

      /**
       * Apply Stochastich Gradient Descent to train network through one epoch
       * @param x list of inputs data
       * @param y list of expected ouputs data
       * @param batch_size: size of each batch
       * @param eta learning rate
       */
      void sgd(std::vector<Vector>& x,
               std::vector<Vector>& y,
               std::size_t batch_size, double eta);


      template <class Test>
      void train(int epochs, std::size_t batch_size, double eta,
                 std::vector<Vector>& x_train,
                 std::vector<Vector>& y_train,
                 const std::vector<Vector>& x_test,
                 const std::vector<Vector>& y_test,
                 Test test);

      template <class Test>
      void eval(const std::vector<Vector>& x_test,
                const std::vector<Vector>& y_test,
                Test test);



      void dump(std::ostream& os);

   private:
      Random rand_;
      std::size_t layers_;
      std::vector<Matrix> weights_;
      std::vector<Vector> biases_;


      /**
       * Apply Stochastic Gradient Descent to a mini_batch
       *
       *
       * m = end - begin
       *
       * w_k = w_k - (eta / m) * sum(j=begin->end-1, delta C_xj / delta w_k)
       * b_k = b_k - (eta / m) * sum(j=begin->end-1, delta C_xj / delta b_k)
       * w_k weight value
       * b_k bias value
       * delta C_xj / delta w_k and delta C_xj / delta b_k are computed
       * with backpropagation
       * They represent the gradient descend
       *
       * delta C = grad(C) * delta v
       * delta v represent a movement in the input values of C
       * Let's choose delta v = - eta * grad(C)
       * delta C = - eta * grad(C)^2
       * |delta C| = - eta * |grad(C)^2|
       * |delta C| <= 0
       * So C decrease
       *
       */
      void sgd_batch_(std::vector<Vector>& x,
                      std::vector<Vector>& y,
                      std::size_t begin, std::size_t end, double eta);


      /**
       * Backpropagation algorithm
       * Compute delta C_x / delta_w and delta C_x / delta_b
       *
       * Start by going forward to compute a_l and z_l
       * Go backward to compute err_l, delta C_x / delta_w, delta C_x / delta_b
       * Start at output layer to input layer
       */
      void backpropagation_(const Vector& x, const Vector& y,
                            std::vector<Vector>& delta_nabla_b,
                            std::vector<Matrix>& delta_nabla_w);

   };

}

#include "neural-network.hxx"
