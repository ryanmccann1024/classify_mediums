import json

import matplotlib.pyplot as plt

with open('../../data/processed/v1/train_output_v1.json', 'r') as file_obj:
    DES_DATA = json.load(file_obj)

LEGEND_LABELS = ['Train', 'Validation', 'Testing']
STEP = 1
VERSION = 1


def plot_setup(plot_type=None):
    plt.figure(figsize=(7, 5), dpi=300)
    plt.grid()

    if plot_type == 'train_acc':
        plt.title("Training & Validation Accuracy vs. Number of Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
    elif plot_type == 'train_loss':
        plt.title("Training & Validation Loss vs. Number of Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
    else:
        raise NotImplementedError


def plot_train_loss():
    epoch_labels = [epoch for epoch in range(0, len(DES_DATA['train_loss']), STEP)]

    for key, data_lst in DES_DATA.items():
        if 'loss' in key:
            label = key.split('_')[0].title()
            plt.plot(epoch_labels, data_lst, label=label)

    plt.ylim((0, 1.))
    plt.xlim(epoch_labels[0], epoch_labels[-1])
    plt.xticks(epoch_labels)
    plt.legend()
    plt.savefig(f'./train_loss_v{VERSION}.png')


def plot_train_acc():
    epoch_labels = [epoch for epoch in range(0, len(DES_DATA['train_acc']), STEP)]

    for key, data_lst in DES_DATA.items():
        if 'acc' in key:
            label = key.split('_')[0].title()
            plt.plot(epoch_labels, data_lst, label=label)

    plt.ylim((0, 1.))
    plt.xlim(epoch_labels[0], epoch_labels[-1])
    plt.xticks(epoch_labels)
    plt.legend()
    plt.savefig(f'./train_acc_v{VERSION}.png')
    # plt.show()


if __name__ == '__main__':
    plot_setup(plot_type='train_loss')
    # plot_train_acc()
    plot_train_loss()
