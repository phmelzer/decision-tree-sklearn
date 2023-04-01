#!/usr/bin/env python3

import pandas as pd
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split


class Dataset:

    def __init__(self, config):
        self.config = config
        if self.config["name"] == "iris":
            self.dataset = load_iris()
        elif self.config["name"] == "diabetes":
            self.dataset = load_diabetes()
        else:
            raise ValueError("Dataset name not known")
        self.features = None
        self.labels = None
        self.features_train = None
        self.features_test = None
        self.labels_train = None
        self.labels_test = None

    def process(self):
        df = pd.DataFrame(self.dataset.data, columns=self.dataset.feature_names)
        df['target'] = self.dataset.target
        self.features = df.drop(['target'], axis=1)
        self.labels = df['target']

        # Split dataset in trainingset and testset
        self.features_train, self.features_test, self.labels_train, self.labels_test = \
            train_test_split(self.features, self.labels, test_size=0.2, random_state=13)
