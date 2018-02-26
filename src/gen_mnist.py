import os
from sklearn.datasets import fetch_mldata
import numpy as np

OUT_PATH = 'mnist.data'



if not os.path.isfile(OUT_PATH):
    print('Generating mnist file...')
    mnist = fetch_mldata('MNIST original')
    arr = list(zip(mnist.data, mnist.target))
    np.random.shuffle(arr)
    X, y = zip(*arr)

    with open(OUT_PATH, 'wb') as f:
        for i in range(len(X)):
            f.write(bytes(X[i].tolist()))
            f.write(bytes([int(y[i])]))
    print('Done')
