
#include "neural-network.hh"
#include "mnist-data.hh"

int main()
{

    MNISTData train_data("../data/mnist_train.data");
    MNISTData test_data("../data/mnist_test.data");
    std::vector<ml::Vector> x_train = train_data.x_list();
    std::vector<ml::Vector> y_train = train_data.y_list();
    std::vector<ml::Vector> x_test = test_data.x_list();
    std::vector<ml::Vector> y_test = test_data.y_list();
    std::cout << "data ready\n";

    auto test = [](const ml::Vector& a, const ml::Vector& b)
        {
            return MNISTData::vec_to_digit(a) == MNISTData::vec_to_digit(b);
        };


    ml::NeuralNetwork net({784, 30, 10});
    net.train(20, 10, 3.0, x_train, y_train, x_test, y_test, test);
    return 0;
}
