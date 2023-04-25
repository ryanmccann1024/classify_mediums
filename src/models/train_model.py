from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import time
import os
import copy
import json


# TODO: Run DenseNet-161
# TODO: Reference this code from the docs


def train_model(model, dataloaders, criterion, optimizer, num_classes, num_epochs=25, is_inception=False, device=None):
    since = time.time()

    train_loss = list()
    train_acc = list()
    train_precision = list()
    train_recall = list()
    train_f1_score = list()

    val_loss = list()
    val_acc = list()
    val_precision = list()
    val_recall = list()
    val_f1_score = list()

    best_model_wts = copy.deepcopy(model.state_dict())

    best_f1_score = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            tp = [0] * num_classes
            fp = [0] * num_classes
            tn = [0] * num_classes
            fn = [0] * num_classes
            test = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward; Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                for i in range(num_classes):
                    tp[i] += ((preds == i) & (labels.data == i)).sum().item()
                    fp[i] += ((preds == i) & (labels.data != i)).sum().item()
                    tn[i] += ((preds != i) & (labels.data != i)).sum().item()
                    fn[i] += ((preds != i) & (labels.data == i)).sum().item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            precision = []
            recall = []
            f1_score = []

            for i in range(num_classes):
                p = tp[i] / (tp[i] + fp[i] + 1e-8)
                r = tp[i] / (tp[i] + fn[i] + 1e-8)
                f1 = 2 * p * r / (p + r + 1e-8)
                precision.append(p)
                recall.append(r)
                f1_score.append(f1)

            mean_precision = sum(precision) / len(precision)
            mean_recall = sum(recall) / len(recall)
            mean_f1_score = sum(f1_score) / len(f1_score)

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
                train_precision.append(mean_precision)
                train_recall.append(mean_recall)
                train_f1_score.append(mean_f1_score)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)
                val_precision.append(mean_precision)
                val_recall.append(mean_recall)
                val_f1_score.append(mean_f1_score)

            print('[{}]\tLoss: {:.4f}, Acc: {:.4f}, Mean Precision: {:.4f}, Mean Recall: {:.4f}, Mean F1 Score: {:.4f}'.format(phase, epoch_loss, epoch_acc, mean_precision, mean_recall, mean_f1_score))

            # deep copy the model
            if phase == 'val' and mean_f1_score > best_f1_score:
                best_f1_score = mean_f1_score
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val F1 Score: {:4f}'.format(best_f1_score))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, [train_acc, train_loss, train_precision, train_recall, train_f1_score, val_acc, val_loss, val_precision, val_recall, val_f1_score]


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(num_classes, feature_extract, use_pretrained=True, model='densenet-121'):
    if model == 'densenet-121':
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model == 'densenet-161':
        model_ft = models.densenet161(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model == 'efficient':
        # TODO: There are variations of this!
        model_ft = models.efficientnet_v2_m(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Manually printed the model to find this! (Check it)
        num_ftrs = 1280
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model == 'vision':
        # TODO: There are variations of this AS WELL!
        model_ft = models.vit_b_16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Manually printed the model to find this! (Check it)
        num_ftrs = 768
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    else:
        raise NotImplementedError(f'Model {model} not found')

    return model_ft, input_size

def main():
    num_classes = 5
    batch_size = 64
    num_epochs = 60
    learn_rate = 0.001
    momentum = 0.9
    version = 2
    model = 'efficient'
    feature_extract = False

    base_fp = f"../../data/processed/v{version}"

    model_ft, input_size = initialize_model(num_classes, feature_extract, use_pretrained=False, model=model)

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

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(base_fp, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
        ['train', 'val']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model_ft = model_ft.to(device)

    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=learn_rate, momentum=momentum)

    # Set up the loss function
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, resp_lst = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_classes, 
                                     num_epochs=num_epochs, is_inception=False, device=device)

    train_acc = [num.cpu().tolist() for num in resp_lst[0]]
    train_loss = resp_lst[1]
    train_precision = resp_lst[2]
    train_recall = resp_lst[3]
    train_f1_score = resp_lst[4]
    val_acc = [num.cpu().tolist() for num in resp_lst[5]]
    val_loss = resp_lst[6]
    val_precision = resp_lst[7]
    val_recall = resp_lst[8]
    val_f1_score = resp_lst[9]

    # torch.save(model_ft.state_dict(), f'../../models/densenet-121_v{version}')
    torch.save(model_ft.state_dict(), f'../../models/check_{version}_{model}')

    res_dict = {
        'train_acc': train_acc,
        'train_loss': train_loss,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_f1_score': train_f1_score,
        'val_acc': val_acc,
        'val_loss': val_loss,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1_score': val_f1_score,
        'hyperparams': {
            'epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learn_rate,
            'momentum': momentum,
        }
    }

    with open(f'{base_fp}/train_output_v{version}.json', 'w') as file_obj:
        json.dump(res_dict, file_obj)


if __name__ == '__main__':
    main()
