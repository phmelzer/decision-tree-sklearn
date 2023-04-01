from matplotlib import pyplot as plt
import os
from sklearn import tree


class Plotter:
    def __init__(self, config):
        self.config = config
        self.plot_dir = config["directory"]
        self.plot_name = "decision_tree"

    def plot_tree(self, model, feature_names, target_names):
        fig = plt.figure(figsize=(10, 10))
        tree.plot_tree(model, feature_names=feature_names,
                       class_names=target_names, filled=True, impurity=True, node_ids=False, proportion=False,
                       rounded=True, precision=2)
        plot_file = os.path.join(self.plot_dir, self.plot_name + '.png')
        fig.savefig(plot_file, transparent=False, dpi=300, bbox_inches='tight')
        plt.show()
