import numpy as np
from scipy.stats import bernoulli
from scipy.stats import norm
from scipy.special import expit

class logistic_generator:
    def __init__(self,
                 num_vars : int,
                 betas: list,
                 input_type: str,
                 linkvars = None):

        '''

        :param num_vars: Number of input variables
        :param betas: list of coefficients for input variables
        :param input_type: type of inputs to be generated -
                                "Binary" will generate a random 0,1 matrix
                                "Real" will generate a random matrix with values in (0,1)
        :param linkvars: dictionary indicating which variable indices are to be correlated
                          (if None then there is no correlation in the dataset)
        :return: data generator object
        '''

        self.num_vars = num_vars
        self.betas = betas
        self.linkvars = linkvars
        self.input_type = input_type

    def generate_random_inputs(self, size):
        if self.input_type == "Binary":
            base_array = np.random.binomial(1, 0.2, size = (size, self.num_vars))
            if self.linkvars is not None:
                for k in self.linkvars:
                    base_array[:, k] = np.where(
                        np.random.rand(size) < 0.999,
                        base_array[:, self.linkvars[k]],
                        1 - base_array[:, self.linkvars[k]]
                    )
        else:
            base_array = np.random.rand(size, self.num_vars)
            if self.linkvars is not None:
                for k in self.linkvars:
                    base_array[:, k] = np.clip(
                        0.9 * base_array[:, self.linkvars[k]]
                        + norm.rvs(0, 0.01, size=size),
                        0,
                        1
                    )

        return base_array

    def generate_random_data(self, size):
        X = self.generate_random_inputs(size)
        eta = X @ self.betas
        probcoeff = expit(eta)

        y = bernoulli.rvs(probcoeff)

        return X, y











