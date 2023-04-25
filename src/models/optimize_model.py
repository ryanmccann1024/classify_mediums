from __future__ import print_function
from __future__ import division
import os
import contextlib
import torch
from torchvision import datasets, transforms
from bayes_opt import BayesianOptimization
from train_model import initialize_model, train_model

BASE_FP = "../../data/external/hymenoptera_data"
NUM_CLASSES = 2
FEATURE_EXTRACT = True
TARGET_VAL = 'val_f1_score'
MINIMIZE_TARGET = False

def optimize(batch_size, learn_rate, momentum, num_epochs):

    batch_size = int(batch_size)
    num_epochs = int(num_epochs)

    # Initialize Model
    with open(os.devnull, "w") as dev_null, contextlib.redirect_stderr(dev_null):
        model_ft, input_size = initialize_model(NUM_CLASSES, FEATURE_EXTRACT, use_pretrained=False)

    # Create training and validation datasets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create image datasets
    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(BASE_FP, x),
            data_transforms[x]
        )
        for x in ['train', 'val']
    }

    # Create training and validation dataloaders
    batch_size = 32
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        for x in ['train', 'val']
    }

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Observe that all parameters are being optimized
    params_to_update = model_ft.parameters()
    optimizer_ft = torch.optim.SGD(params_to_update, lr=learn_rate, momentum=momentum)

    # Set up the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Train and evaluate
    with open(os.devnull, "w") as dev_null, contextlib.redirect_stdout(dev_null):
        model_ft, resp_lst = train_model(
            model_ft,
            dataloaders_dict,
            criterion,
            optimizer_ft,
            NUM_CLASSES,
            num_epochs=num_epochs,
            is_inception=False,
            device=device
        )


    results = {'train_acc': resp_lst[0][num_epochs - 1].item(),
               'train_loss': resp_lst[1][num_epochs - 1],
               'train_precision': resp_lst[2][num_epochs - 1],
               'train_recall': resp_lst[3][num_epochs - 1],
               'train_f1_score': resp_lst[4][num_epochs - 1],
               'val_acc': resp_lst[5][num_epochs - 1].item(),
               'val_loss': resp_lst[6][num_epochs - 1],
               'val_precision': resp_lst[7][num_epochs - 1],
               'val_recall': resp_lst[8][num_epochs - 1],
               'val_f1_score': resp_lst[9][num_epochs - 1]}

    return -results[TARGET_VAL] if MINIMIZE_TARGET else results[TARGET_VAL]

def main():

    pbounds = {'batch_size': (32, 128),
               'learn_rate': (0.0, 0.1),
               'momentum': (0.7, 0.99),
               'num_epochs': (1, 20)}

    optimizer = BayesianOptimization(f=optimize, pbounds=pbounds, verbose=10, random_state=42)

    optimizer.maximize(n_iter=100, init_points=10)

    # Print params optimized for target
    print(f"target ({ TARGET_VAL }): { -optimizer.max['target'] if MINIMIZE_TARGET else optimizer.max['target'] }")
    print(f"batch_size: { int(optimizer.max['params']['batch_size']) }")
    print(f"learn_rate: { optimizer.max['params']['learn_rate'] }")
    print(f"momentum: { optimizer.max['params']['momentum'] }")
    print(f"num_epochs: { int(optimizer.max['params']['num_epochs']) }")

if __name__ == '__main__':
    main()
