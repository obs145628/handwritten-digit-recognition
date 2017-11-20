#include "neural-network.hh"
#include "random.hh"
#include <cassert>

namespace ml
{

   NeuralNetwork::NeuralNetwork(const std::vector<std::size_t>& sizes)
      : rand_()
      , layers_(sizes.size())
   {

      for (std::size_t i = 0; i + 1 < layers_; ++i)
      {
         Matrix weight(sizes[i + 1], sizes[i]);
         Vector bias(sizes[i + 1]);
         rand_.fill_double(weight.begin(), weight.end(), -1, 1);
         rand_.fill_double(bias.begin(), bias.end(), -1, 1);

         weights_.push_back(weight);
         biases_.push_back(bias);
      }

   }

   Vector NeuralNetwork::forward(const Vector& x)
   {
      Vector a = x;
      for (std::size_t i = 0; i + 1 < layers_ ; ++i)
      {
         a = sigmoid(Matrix::mul(weights_[i], a) + biases_[i]);
      }
      return a;
   }

   void NeuralNetwork::sgd(std::vector<Vector>& x,
                           std::vector<Vector>& y,
                           std::size_t batch_size, double eta)
   {
      assert(x.size() == y.size());
      std::size_t n = x.size();

      for (std::size_t i = n - 1; i < n; --i)
      {
         std::size_t j = rand_.next_long(i + 1);
         std::swap(x[i], x[j]);
         std::swap(y[i], y[j]);
      }

      for (std::size_t i = 0; i < n; i += batch_size)
      {
         std::size_t begin = i;
         std::size_t end = std::min(i + batch_size, n);
         sgd_batch_(x, y, begin, end, eta);
      }


   }

   void NeuralNetwork::sgd_batch_(std::vector<Vector>& x,
                                  std::vector<Vector>& y,
                                  std::size_t begin, std::size_t end,
                                  double eta)
   {
      assert(begin < end);
      std::size_t m = end - begin;

      std::vector<Vector> nabla_b;
      std::vector<Matrix> nabla_w;
      for (std::size_t i = 0; i + 1 < layers_; ++i)
      {
         nabla_b.push_back(Vector::zero(biases_[i].size()));
         nabla_w.push_back(Matrix::zero(weights_[i].rows(),
                                        weights_[i].cols()));

      }

      for (std::size_t i = begin; i < end; ++i)
      {
         std::vector<Vector> delta_nabla_b;
         std::vector<Matrix> delta_nabla_w;
         backpropagation_(x[i], y[i], delta_nabla_b, delta_nabla_w);


         for (std::size_t i = 0; i + 1 < layers_; ++i)
         {
            nabla_b[i] += delta_nabla_b[i];
            nabla_w[i] += delta_nabla_w[i];
         }
      }

      for (std::size_t i = 0; i + 1 < layers_; ++i)
      {
         biases_[i] -= (eta / m) * nabla_b[i];
         weights_[i] -= (eta / m) * nabla_w[i];
      }
   }

   void NeuralNetwork::backpropagation_(const Vector& x, const Vector& y,
                                        std::vector<Vector>& delta_nabla_b,
                                        std::vector<Matrix>& delta_nabla_w)
   {
      delta_nabla_b.resize(layers_ - 1);
      delta_nabla_w.resize(layers_ - 1);

      std::vector<Vector> as;
      std::vector<Vector> zs;

      //forward
      Vector a = x;
      as.push_back(a);
      for (std::size_t i = 0; i + 1 < layers_ ; ++i)
      {
         Vector z = Matrix::mul(weights_[i] , a) + biases_[i];
         a = sigmoid(z);
         as.push_back(a);
         zs.push_back(z);
      }



      //backward
      Vector delta = hadamard_product(as.back() - y, sigmoid_prime(zs.back()));
      delta_nabla_b.back() = delta;
      delta_nabla_w.back() = outer_product(delta, as[layers_ - 2]);

      for (std::size_t l = layers_ - 3; l < layers_; --l)
      {
         delta = hadamard_product(Matrix::mul(weights_[l+1].transpose(), delta),
                                  sigmoid_prime(zs[l]));
         delta_nabla_b[l] = delta;
         delta_nabla_w[l] = outer_product(delta, as[l]);
      }

   }

   void NeuralNetwork::dump(std::ostream& os)
   {
      os << "layers: " << layers_ << std::endl;
      os << "weights: " << std::endl;
      for (auto w : weights_)
         std::cout << w << std::endl;

      os << "biases: " << std::endl;
      for (auto b : biases_)
         os << b << std::endl;
   }


}
