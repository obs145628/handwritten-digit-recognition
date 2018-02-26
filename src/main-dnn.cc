#include <ai/datasets/mnist.hh>
#include <ai/datasets/norm4.hh>

#include <ai/dnn/activation.hh>
#include <ai/dnn/initializer.hh>
#include <ai/dnn/fully-connected-layer.hh>
#include <ai/dnn/network.hh>
#include <ai/dnn/sgd-optimizer.hh>
#include <ai/la/tensor_ops.hh>
#include <ai/la/random.hh>


int main()
{
    nrandom::seed(12);
    Matrix x_train;
    Matrix y_train;
    Matrix x_test;
    Matrix y_test;    
    mnist::load(x_train, y_train, x_test, y_test, 60000, 10000);

    auto l1 = new FullyConnectedLayer(784, 100, new SigmoidActivation);
    auto l2 = new FullyConnectedLayer(100, 10, new SigmoidActivation);
    l1->init_weigths(gauss_initializer);
    l2->init_weigths(gauss_initializer);

    Network net(layers_t {l1, l2},
		new CrossEntropyCost, mnist::output_test);

    SGDOptimizer opti(0.5, 10, 0.0, 5.0);

    net.evaluate(x_test, y_test);
    net.train(x_train, y_train, x_test, y_test, opti, 5, true);
    
    return 0;
}
