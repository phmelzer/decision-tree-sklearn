#!/usr/bin/env python3

from typing import Dict
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class DecisionTree:
    model = None

    def __init__(self, config: Dict):
        # Create the model
        self.type = config["type"]
        if self.type == "classifier":
            self.model = DecisionTreeClassifier(criterion=config["hyperparameter_classifier"]["criterion"],
                                                splitter=config["hyperparameter_classifier"]["splitter"],
                                                max_depth=config["hyperparameter_classifier"]["max_depth"],
                                                min_samples_split=config["hyperparameter_classifier"]
                                                ["min_samples_split"],
                                                min_samples_leaf=config["hyperparameter_classifier"]
                                                ["min_samples_leaf"],
                                                min_weight_fraction_leaf=config["hyperparameter_classifier"]
                                                ["min_weight_fraction_leaf"],
                                                max_features=config["hyperparameter_classifier"]["max_features"],
                                                random_state=config["hyperparameter_classifier"]["random_state"],
                                                max_leaf_nodes=config["hyperparameter_classifier"]["max_leaf_nodes"],
                                                min_impurity_decrease=config["hyperparameter_classifier"]
                                                ["min_impurity_decrease"],
                                                class_weight=config["hyperparameter_classifier"]["class_weight"],
                                                ccp_alpha=config["hyperparameter_classifier"]["ccp_alpha"])

        elif self.type == 'regressor':
            self.model = DecisionTreeRegressor(criterion=config["hyperparameter_regressor"]["criterion"],
                                                splitter=config["hyperparameter_regressor"]["splitter"],
                                                max_depth=config["hyperparameter_regressor"]["max_depth"],
                                                min_samples_split=config["hyperparameter_regressor"]
                                                ["min_samples_split"],
                                                min_samples_leaf=config["hyperparameter_regressor"]["min_samples_leaf"],
                                                min_weight_fraction_leaf=config["hyperparameter_regressor"]
                                                ["min_weight_fraction_leaf"],
                                                max_features=config["hyperparameter_regressor"]["max_features"],
                                                random_state=config["hyperparameter_regressor"]["random_state"],
                                                max_leaf_nodes=config["hyperparameter_regressor"]["max_leaf_nodes"],
                                                min_impurity_decrease=config["hyperparameter_regressor"]
                                                ["min_impurity_decrease"],
                                                ccp_alpha=config["hyperparameter_regressor"]["ccp_alpha"])
        else:
            raise ValueError("Model type not known")

    def get_feature_names(self):
        return self.model.feature_names_in_

    def get_feature_importancy(self):
        return self.model.feature_importances_
