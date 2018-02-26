#include <ai/datasets/mnist.hh>
#include <ai/la/matrix.hh>
#include <ai/la/random.hh>


/**
 * Compute cost function
 * @param x (set_len, nb_features) - matrix of features for each example
 * @param y (set_len, nb_labels) - vector of labels for each example
 * @param w (nb_features, nb_labels) - vector of weights
 * @param lbda - coefficient for l2 regularization
 */
num_t cost_function(const Matrix& x, const Matrix& y, const Matrix& w, num_t lbda)
{
    Matrix y_hat = softmax(dot(x, w));
    num_t res = - sum(y * log(y_hat));
    num_t reg2 = lbda * norm_square(w);
    return res + reg2;
}


/**
 * Evaluate the current matrix
 * @param x (set_len, nb_features) - matrix of features for each example
 * @param y (set_len, nb_labels) - vector of labels for each example
 * @param w (nb_features, nb_labels) - vector of weights
 * @param lbda - coefficient for l2 regularization
 */
void evaluate(const Matrix& x, const Matrix& y, const Matrix& w, num_t lbda)
{
    Matrix y_hat = softmax(dot(x, w));
    std::size_t total = x.rows();
    std::size_t succ = 0;

    for (std::size_t i = 0; i < total; ++i)
	succ += mnist::output_test(y[i], y_hat[i]);

    num_t perc = succ * 100.0 / total;
    std::cout << "Cost: " << cost_function(x, y, w, lbda) << std::endl;
    std::cout << "Results: " << succ << "/" << total << "(" << perc << "%)" << std::endl;
}


/**
 * Apply stochastic gradient descent on the whole training set of size n
 * @param x (n, nb_features) - matrix of features for each example
 * @param y (n, nb_labels) - vector of labels for each example
 * @param w (nb_features, nb_labels) - vector of weights
 * @param lr - learning rate
 * @param lbda - coefficient for l2 regularization
 */
Matrix sgd(const Matrix& x, const Matrix& y, const Matrix& w, num_t lr, num_t lbda)
{
   Matrix y_hat = softmax(dot(x, w));
   Matrix dw = - dot(x.transpose(), y - y_hat) + lbda * w;
   return w - lr * dw;
}

/**
 * Run the training for the several epochs
 * @param x_train (n, nb_features) - matrix of features (training set)
 * @param y_train (n, nb_labels) - matrix orflabels (training set)
 * @param x_test (n, nb_features) - matrix of features (testing set)
 * @param y_test (n, nb_labels) - matrix or labels (testing set)
 * @param epochs - number of epochs of learning
 * @param lr - learning rate
 * @param lbda - coefficient for l2 regularization
 * @param use_intercept - if true, add a bias to the weights
 * @return optimized weights vector
 */
Matrix train(Matrix& x_train, Matrix& y_train,
	     Matrix& x_test, Matrix& y_test,
	     std::size_t epochs, num_t lr,
	     num_t lbda = 0, bool use_intercept = false)
{
    if (use_intercept)
    {
	x_train = Matrix::hstack({Matrix::ones(x_train.rows(), 1), x_train});
	x_test = Matrix::hstack({Matrix::ones(x_test.rows(), 1), x_test});
    }
    Matrix w = Matrix::zeros(x_train.cols(), y_train.cols());

    for (std::size_t i = 1; i <= epochs; ++i)
    {
	std::cout << "Epoch " << i << ":" << std::endl;
	w = sgd(x_train, y_train, w, lr, lbda);

	std::cout << "Train:" << std::endl;
	evaluate(x_train, y_train, w, lbda);
	std::cout << "Test:" << std::endl;
	evaluate(x_test, y_test, w, lbda);
    }

    return w;
}


int main()
{
    nrandom::seed(12);
    Matrix x_train;
    Matrix y_train;
    Matrix x_test;
    Matrix y_test;
    mnist::load(x_train, y_train, x_test, y_test, 60000, 10000);


    train(x_train, y_train, x_test, y_test, 100, 0.00002, 0.1, true);
}
