import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import Constant
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import Layer


class InitCentersRandom(Initializer):
    """ Initializer for initialization of centers of RBF network
        as random samples from the given data set.
    # Arguments
        X: matrix, dataset to choose the centers from (random rows
          are taken as centers)
    """

    def __init__(self, X, samples_num=9):
        self.X = X
        self.samples_num = samples_num

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]
        idx = np.random.randint(self.X.shape[0], size=self.samples_num)
        return self.X[idx, :]


class InitCentersKMeans(Initializer):
    """ Initializer for initialization of centers of RBF network
        as K clusters produced from K-means from the given data set.
    # Arguments
        X: matrix, dataset to choose the centers from (random rows
          are taken as centers)
        n_clusters: Number of clusters to find
    """

    def __init__(self, X, n_clusters):
        self.X = X
        self.n_clusters = n_clusters

    def __call__(self, shape, dtype=None):
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(self.X)
        return kmeans.cluster_centers_


class RBFLayer(Layer):
    """ Layer of Gaussian RBF units.
    # Example
    ```python
        model = Sequential()
        model.add(RBFLayer(10,
                           initializer=InitCentersRandom(X),
                           betas=1.0,
                           input_shape=(1,)))
        model.add(Dense(1))
    ```
    # Arguments
        output_dim: number of hidden units (i.e. number of outputs of the layer)
        initializer: instance of initiliazer to initialize centers
        betas: float, initial value for betas
    """

    def __init__(self, num_outputs, initializer=None, betas=1.0, **kwargs):
        self.num_outputs = num_outputs
        self.init_betas = betas
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
        else:
            self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.centers = self.add_weight(name='centers',
                                       shape=(self.num_outputs, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.num_outputs,),
                                     initializer=Constant(
                                         value=self.init_betas),
                                     trainable=True)

        super(RBFLayer, self).build(input_shape)

    def call(self, x):

        C = K.expand_dims(self.centers)
        H = K.transpose(C - K.transpose(x))
        return K.exp(-self.betas * K.sum(H ** 2, axis=1))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_outputs)

    def get_config(self):
        config = {"num_outputs": self.num_outputs}
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
