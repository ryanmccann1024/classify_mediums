import json
import os
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

with open('../../data/processed/v1/train_output_v1.json', 'r') as file_obj:
    DES_DATA = json.load(file_obj)

LEGEND_LABELS = ['Train', 'Validation', 'Testing']
STEP = 10
VERSION = 2

with open(f'../../data/processed/v{VERSION}/train_output_v{VERSION}.json', 'r') as file_obj:
    DES_DATA = json.load(file_obj)


def get_data():
    resp_dict = {'years': list(), 'artists': list(), 'mediums': dict()}
    base_path = f'../../data/processed/v{VERSION}'

    des_dirs = [item for item in os.listdir(base_path) if os.path.isdir(f'{base_path}/{item}')]
    for curr_dir in des_dirs:
        for medium_dir in os.listdir(f'{base_path}/{curr_dir}'):
            for file in os.listdir(f'{base_path}/{curr_dir}/{medium_dir}'):
                # Keep track of how many mediums are in our dataset
                medium = medium_dir.title()
                if medium not in resp_dict['mediums']:
                    resp_dict['mediums'][medium] = 1
                else:
                    resp_dict['mediums'][medium] += 1

                if file.split('_')[-1].strip('.jpg') != 'None' and file.split('_')[-1].strip('.jpg') != '0':
                    year = int(file.split('_')[-1].strip('.jpg'))
                    resp_dict['years'].append(year)

                artist = ''
                for word in file.split('_')[:-1]:
                    artist += f'{word} '
                artist = artist.title().strip()

                if artist not in resp_dict['artists']:
                    resp_dict['artists'].append(artist)

    return resp_dict


# TODO: Remove unknown years
# TODO: Custom bins to be more descriptive
def plot_data():
    plot_dict = get_data()

    # Plot mediums bar graph
    plt.figure(1)
    plt.bar(plot_dict['mediums'].keys(), plot_dict['mediums'].values(), color='#6666ff')
    plt.title('Medium Distribution')
    plt.xlabel('Mediums', fontweight='semibold')
    plt.ylabel('Number of Images', fontweight='semibold')
    plt.savefig(f'./v{VERSION}/mediums_v{VERSION}.png')

    plt.figure(2)
    plt.hist(plot_dict['years'], color='#6666ff', edgecolor='black')
    plt.title('Year Created Distribution')
    plt.xlabel('Years', fontweight='semibold')
    plt.ylabel('Number of Images', fontweight='semibold')
    plt.savefig(f'./v{VERSION}/years_v{VERSION}.png')


def plot_setup(plot_type=None):
    """
    Sets up the plots depending on what is to be plotted, or not to be plotted. - AI William Shakespeare.
    """
    # plt.figure(figsize=(7, 5), dpi=300)
    plt.clf()
    plt.figure(dpi=300)

    if plot_type == 'train_acc':
        plt.title("Training & Validation Accuracy vs. Number of Epochs")
        plt.xlabel("Epochs", fontweight='semibold')
        plt.ylabel("Accuracy", fontweight='semibold')
        plt.grid()
    elif plot_type == 'train_loss':
        plt.title("Training & Validation Loss vs. Number of Epochs")
        plt.xlabel("Epochs", fontweight='semibold')
        plt.ylabel("Loss", fontweight='semibold')
        plt.grid()
    elif plot_type == 'train_prec':
        plt.title("Training & Validation Mean Precision vs. Number of Epochs")
        plt.xlabel("Epochs", fontweight='semibold')
        plt.ylabel("Mean Precision", fontweight='semibold')
        plt.grid()
    elif plot_type == 'train_recall':
        plt.title("Training & Validation Mean Recall vs. Number of Epochs")
        plt.xlabel("Epochs", fontweight='semibold')
        plt.ylabel("Mean Recall", fontweight='semibold')
        plt.grid()
    elif plot_type == 'train_f1_score':
        plt.title("Training & Validation Mean F1 Score vs. Number of Epochs")
        plt.xlabel("Epochs", fontweight='semibold')
        plt.ylabel("Mean F1 Score", fontweight='semibold')
        plt.grid()


def plot_train_loss():
    """
    Plot loss related to training, this includes the validation loss.
    """
    epochs = [epoch for epoch in range(0, len(DES_DATA['train_loss']))]
    epoch_labels = [epoch for epoch in range(0, len(DES_DATA['train_loss']), STEP)]

    for key, data_lst in DES_DATA.items():
        if 'loss' in key:
            print(data_lst)
            label = key.split('_')[0].title()
            plt.plot(epochs, data_lst, label=label)

    plt.ylim((0, 2.0))
    plt.xlim(epoch_labels[0], epoch_labels[-1])
    plt.xticks(epoch_labels)
    plt.legend()
    plt.savefig(f'./v{VERSION}/train_loss_v{VERSION}.png')


def plot_train_prec():
    """
    Plot precision related to training, this includes the validation loss.
    """
    epochs = [epoch for epoch in range(0, len(DES_DATA['train_precision']))]
    epoch_labels = [epoch for epoch in range(0, len(DES_DATA['train_precision']), STEP)]

    for key, data_lst in DES_DATA.items():
        if 'precision' in key:
            print(data_lst)
            label = key.split('_')[0].title()
            plt.plot(epochs, data_lst, label=label)

    plt.ylim((0, 1.))
    plt.xlim(epoch_labels[0], epoch_labels[-1])
    plt.xticks(epoch_labels)
    plt.legend()
    plt.savefig(f'./v{VERSION}/train_precision_v{VERSION}.png')

def plot_train_recall():
    """
    Plot recall related to training, this includes the validation loss.
    """
    epochs = [epoch for epoch in range(0, len(DES_DATA['train_recall']))]
    epoch_labels = [epoch for epoch in range(0, len(DES_DATA['train_recall']), STEP)]

    for key, data_lst in DES_DATA.items():
        if 'recall' in key:
            print(data_lst)
            label = key.split('_')[0].title()
            plt.plot(epochs, data_lst, label=label)

    plt.ylim((0, 1.))
    plt.xlim(epoch_labels[0], epoch_labels[-1])
    plt.xticks(epoch_labels)
    plt.legend()
    plt.savefig(f'./v{VERSION}/train_recall_v{VERSION}.png')

def plot_train_f1_score():
    """
    Plot F1 score related to training, this includes the validation loss.
    """
    epochs = [epoch for epoch in range(0, len(DES_DATA['train_f1_score']))]
    epoch_labels = [epoch for epoch in range(0, len(DES_DATA['train_f1_score']), STEP)]

    for key, data_lst in DES_DATA.items():
        if 'f1_score' in key:
            print(data_lst)
            label = key.split('_')[0].title()
            plt.plot(epochs, data_lst, label=label)

    plt.ylim((0, 1.))
    plt.xlim(epoch_labels[0], epoch_labels[-1])
    plt.xticks(epoch_labels)
    plt.legend()
    plt.savefig(f'./v{VERSION}/train_f1_score_v{VERSION}.png')

def plot_train_acc():
    """
    Plot accuracy related to training, this included validation in the accuracy.
    """
    epochs = [epoch for epoch in range(0, len(DES_DATA['train_acc']))]
    epoch_labels = [epoch for epoch in range(0, len(DES_DATA['train_acc']), STEP)]

    for key, data_lst in DES_DATA.items():
        if 'acc' in key:
            label = key.split('_')[0].title()
            plt.plot(epochs, data_lst, label=label)

    plt.ylim((0, 1.))
    plt.xlim(epoch_labels[0], epoch_labels[-1])
    plt.xticks(epoch_labels)
    plt.legend()
    plt.savefig(f'./v{VERSION}/train_acc_v{VERSION}.png')
    # plt.show()


def plot_confusion_matrix():
    with open(f'../../data/processed/v1/test_output_v{VERSION}.json', 'r') as file_obj:
        test_data = json.load(file_obj)

    # TODO: Check if this has to be in a certain order
    classes = ('oil', 'pastel', 'pencil', 'tempera', 'watercolor')

    cf_matrix = confusion_matrix(test_data['actual'], test_data['predicted'])
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])

    cmap_color = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
    sns.heatmap(df_cm, annot=True, cmap=cmap_color).set(title='Feature Heat Map')
    plt.xlabel('Actual', fontweight='semibold')
    plt.ylabel('Predicted', fontweight='semibold')

    plt.savefig(f'./v{VERSION}/confusion_matrix_v{VERSION}.png')


if __name__ == '__main__':
    plot_setup()
    plot_data()
    plot_setup('train_acc')
    plot_train_acc()
    plot_setup('train_loss')
    plot_train_loss()
    plot_setup('train_prec')
    plot_train_prec()
    plot_setup('train_recall')
    plot_train_recall()
    plot_setup('train_f1_score')
    plot_train_f1_score()
    plot_setup()
    plot_confusion_matrix()
