# src/visualizer/plot_manager.py
import matplotlib.pyplot as plt

class PlotManager:
    def plot_heatmap(self, data):
        plt.imshow(data, cmap='hot', interpolation='nearest')
        plt.show()

    def plot_distribution(self, data, axis):
        pass