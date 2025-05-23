from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class LogTransformer(BaseEstimator, TransformerMixin):
    def transform(self, y):
        return 10 * np.log1p(y)

    def inverse_transform(self, y):
        return np.expm1(y / 10)




