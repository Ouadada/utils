from sklearn.base import BaseEstimator, TransformerMixin
from numpy import np


class addStats(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.mean_ = 0.
        self.std_ = 0.
    # pass

    def fit(self, X, y=None):
        self.mean_ = X.mean(axis=1)
        self.std_ = X.std(axis=1)
        return self

    def transform(self, X, y=None):
        self.mean_ = X.mean(axis=1).reshape(-1, 1)
        self.std_ = X.std(axis=1).reshape(-1, 1)
        X = np.append(X, self.mean_, 1)
        X = np.append(X, self.std_, 1)

        return X

    def fit_transform(self, X, y=None):
        X = self.fit(X, y).transform(X)
        return X


class addNoise(BaseEstimator, TransformerMixin):

    def __init__(self, noise_std):
        self.noise_std = noise_std

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X += np.random.normal(0, noise_std, X.shape)
        return X

    def fit_transform(self, X, y=None):
        X = self.fit(X, y).transform(X)
        return X


class smote_transformer(BaseEstimator, TransformerMixin):

    def __init__(self, ratio):
        self.ratio = ratio

    def fit(self, X, y):
        return self

    def transform(self, X, y):
        smote = SMOTE(ratio=self.ratio, n_jobs=-1)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res

    def fit_transform(self, X, y):
        X_res, y_res = self.fit(X, y).transform(X, y)
        return X_res, y_res
