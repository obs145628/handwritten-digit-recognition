# Handwritten digit recognition

C++ implementation of several machine learnings algorithms
to solve handwritten digit recognition.

No extern library is used, all algorithms are built from scratch


## Build

````bash
$ git submodule init
$ git submodule sync
$ git submodule update
$ mkdir _build
$ cd _build/
$ cmake ..
$ make
````

## Execute

```bash
$ make run_dnn
$ make run_logreg
$ make run_softmaxreg
```

## Datasets

The mnist database is used.
It is built using python and scikit-learn.
To build the dataset
```bash
$ make gen_mnist
```
Python3 with numpy and scikit-learn required

## Algorithms

- Neural Network
- Logistic Regression
- Softmax Regression