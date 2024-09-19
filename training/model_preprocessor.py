import pickle
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

from warframe_marketplace_predictor.filepaths import *


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, X: pd.DataFrame, y=None) -> 'Preprocessor':
        # Fit the scaler on the log1p of the "re_rolls" column
        self.scaler.fit(np.log1p(X[["re_rolls"]]))
        return self

    def transform(self, X: pd.DataFrame) -> List[pd.DataFrame]:
        X_copy = X.copy()
        re_rolls_log = np.log1p(X_copy[["re_rolls"]])
        X_copy["re_rolls"] = self.scaler.transform(re_rolls_log)
        return self.split_X(X_copy)

    @staticmethod
    def split_X(X: pd.DataFrame) -> List[pd.DataFrame]:
        return [
            X[["weapon_url_name"]],
            X[["re_rolls"]],
            X[["positive1", "positive2", "positive3", "negative"]]
        ]

    def save(self, filepath: str = None):
        filepath = filepath if filepath else model_preprocessor_file_path
        # Save the preprocessor instance to a pickle file
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str = None) -> 'Preprocessor':
        filepath = filepath if filepath else model_preprocessor_file_path
        # Load the preprocessor instance from a pickle file
        with open(filepath, "rb") as f:
            return pickle.load(f)
