import matplotlib.pyplot as plt
import numpy as np


def plot_loss():
    pass


def plot_acc(data_matrix, epochs):
    legend_labels = ['Train', 'Validation', 'Testing']
    plt.figure(figsize=(7, 5), dpi=300)
    plt.title("Training & Validation Accuracy vs. Number of Epochs")
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    step = 50
    x_ticks = [tick for tick in range(0, epochs + step, step)]
    epoch_lst = [epoch for epoch in range(0, epochs)]
    for i, curr_lst in enumerate(data_matrix):
        plt.plot(epoch_lst, curr_lst, label=legend_labels[i])

    plt.ylim((0, 1.))
    plt.xticks(x_ticks)
    plt.legend()
    plt.show()
