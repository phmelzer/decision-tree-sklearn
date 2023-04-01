import os
from dataset import Dataset
from plot import Plotter
from model import DecisionTree
from typing import Dict
from sklearn import metrics


class Train:

    def __init__(self, config: Dict):
        self.config = config

        print(f"Setup train and test data")
        self.dataset = Dataset(self.config["dataset"])
        self.dataset.process()

        self.model = DecisionTree(config["model"])

        self.plotter = Plotter(self.config["plotter"])

    def start(self):
        self.model.model.fit(self.dataset.features_train, self.dataset.labels_train)
        # Evaluate model with testset
        predictions = self.model.model.predict(self.dataset.features_test)

        # Calculate accuracy
        if config["model"]["type"] == "classifier":
            accuracy = metrics.accuracy_score(predictions, self.dataset.labels_test)
            print(f'Accuracy: {accuracy}')

        if config["model"]["type"] == "regressor":
            mean_absolute_error = metrics.mean_absolute_error(predictions, self.dataset.labels_test)
            print(f'Mean absolute error: {mean_absolute_error}')

        feature_names = self.model.get_feature_names()
        print(f"Feature names: {feature_names}")

        feature_importancy = self.model.get_feature_importancy()
        print(f"Feature importance: {feature_importancy}")

        self.plotter.plot_tree(self.model.model, self.dataset.dataset.feature_names, self.dataset.dataset.target_names)


if __name__ == "__main__":
    import yaml

    config = None
    with open(os.path.join("../config/config.yaml")) as file:
        config = yaml.safe_load(file)

    train = Train(config)
    train.start()


