import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# This file implements the data pre-processing units that will
# be used in the scikit learn pipeline


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X):
        return self

    def transform(self, X):
        """
        Select columns from a DataFrame and return a Numpy array\n
        :param X: A pandas DataFrame\n
        :return: Numpy array
        """
        return X[self.attribute_names].values