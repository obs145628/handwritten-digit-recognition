#include <iostream>
#include <ai/datasets/mnist.hh>
#include <ai/la/random.hh>
#include <ai/la/matrix.hh>


/**
 * Compute log-likelihood cost function
 * @param x (set_len, nb_features) - matrix of features for each example
 * @param y (set_len) - vector of labels for each example
 * @param w (nb_features) - vector of weights
 * @return cost value
 */
num_t log_likelihood(const Matrix& x, const Vector& y, const Vector& w)
{
    Vector z = dot(x, w);
    num_t cost = 0;
    for (std::size_t i = 0; i < x.rows(); ++i)
	cost -= (y[i] * z[i]) - std::log(1 + std::exp(z[i]));

    return cost;
}

/**
 * Apply gradient descent on the whole training set of size n
 * @param x (n, nb_features) - matrix of features for each example
 * @param y (n) - vectors or labels for each example
 * @param w (nb_features) - vector of weights
 * @param lr - learning rate
 * @return (nb_features) updated weights
 */
Vector sgd(const Matrix& x, const Vector& y, const Vector& w, num_t lr)
{
    Vector y_hat = sigmoid(dot(x, w));
    Vector dw = dot(x.transpose(), y_hat - y);
    return w - lr * dw;
}

/**
 * Apply gradient descent on the whole training set of size n, using mini batches
 * @param x (n, nb_features) - matrix of features for each example
 * @param y (n) - vectors or labels for each example
 * @param w (nb_features) - vector of weights
 * @param lr - learning rate
 * @param batch_size - size of each batch
 * @return (nb_features) updated weights
 */
Vector sgd_mini_batch(const Matrix& x, const Vector& y, const Vector& w,
		      num_t lr, std::size_t batch_size)
{
    std::size_t n = x.rows();

    Vector w2 = w; 

    for (std::size_t k = 0; k < n; k += batch_size)
    {
	std::size_t m = std::min(n - k, batch_size);
	Matrix x_batch = x.sub(k, k + m);
        Vector y_batch = y.sub(k, k + m);
	w2 = sgd(x_batch, y_batch, w2, lr / m);
    }

    return w2;
}

/**
 * Evaluate the current matrix
 * @param x (n, nb_features) - matrix of features for each example
 * @param y (n) - vectors or labels for each example
 * @param w (nb_features) - vector of weights
 */
void evaluate(const Matrix& x, const Vector& y, const Vector& w)
{
    std::size_t succ = 0;
    std::size_t total = x.rows();

    Vector y_hat = sigmoid(dot(x, w));
    for (std::size_t i = 0; i < total; ++i)
        succ += y[i] == std::round(y_hat[i]);

    num_t perc = succ * 100.0 / total;
    std::cout << "Cost: " << log_likelihood(x, y, w) << std::endl;
    std::cout << "Results: " << succ << "/" << total << "(" << perc << "%)" << std::endl;
}

/**
 * Run the training for the several epochs
 * @param x_train (n, nb_features) - matrix of features (training set)
 * @param y_train (n) - vectors or labels (training set)
 * @param x_test (n, nb_features) - matrix of features (testing set)
 * @param y_test (n) - vectors or labels (testing set)
 * @param epochs - number of epochs of learning
 * @param use_intercept - if true, add a bias to the weights
 * @param batch_size - size of batch size for gradient descent. 
 * if -1, gradient descent is on the fll dataset
 * @return optimized weights vector
 */
Vector train(Matrix& x_train, Vector& y_train,
	     Matrix& x_test, Vector& y_test,
	     std::size_t epochs, num_t lr,
	     bool use_intercept = false, int batch_size = -1)
{
    if (use_intercept)
    {
	x_train = Matrix::hstack({Matrix::ones(x_train.rows(), 1), x_train});
	x_test = Matrix::hstack({Matrix::ones(x_test.rows(), 1), x_test});
    }
    Vector w = Vector::zeros(x_train.cols());

    for (std::size_t i = 1; i <= epochs; ++i)
    {
	std::cout << "Epoch " << i << ":" << std::endl;
	if (batch_size == -1)
	    w = sgd(x_train, y_train, w, lr);
	else
	    w = sgd_mini_batch(x_train, y_train, w, lr, batch_size);

	std::cout << "Train:" << std::endl;
	evaluate(x_train, y_train, w);
	std::cout << "Test:" << std::endl;
	evaluate(x_test, y_test, w);
    }

    return w;
}


int main()
{
    nrandom::seed(12);
    Matrix x_train;
    Vector y_train;
    Matrix x_test;
    Vector y_test;
    mnist::load_bin(x_train, y_train, x_test, y_test, 12000, 2780);

    train(x_train, y_train, x_test, y_test, 30, 0.001, true, -1);
}
