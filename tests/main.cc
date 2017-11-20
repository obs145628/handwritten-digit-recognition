#include "neural-network.hh"
#include "mnist-data.hh"
#include "date.hh"

int main()
{
    MNISTData train_data("../data/mnist_train.data");
    MNISTData test_data("../data/mnist_test.data");
    std::vector<ml::Vector> x_train = train_data.x_list();
    std::vector<ml::Vector> y_train = train_data.y_list();
    std::vector<ml::Vector> x_test = test_data.x_list();
    std::vector<ml::Vector> y_test = test_data.y_list();
    std::cout << "data ready\n";

    ml::NeuralNetwork net({784, 30, 10});

    auto t = ml::Date::now();
    for (std::size_t i = 0; i < x_train.size(); ++i)
       net.forward(x_train[i]);
    auto dur = ml::Date::now() - t;

    std::cout << "time: " << dur << "ms " << std::endl;

    return 0;
}
