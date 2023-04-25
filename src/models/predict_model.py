from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import json

VERSION = 2

def test_model(model, dataloaders, criterion, device=None):
    predicted_labels = []
    actual_labels = []

    test_acc_history = []
    test_loss_history = []
    best_acc = 0.0
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        actual_labels.extend(labels.data.cpu().numpy())

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        predicted_labels.extend(preds.data.cpu().numpy())

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        curr_loss = running_loss / len(dataloaders['test'].dataset)
        curr_acc = running_corrects.double() / len(dataloaders['test'].dataset)

        # deep copy the model
        if curr_acc > best_acc:
            best_acc = curr_acc

        test_acc_history.append(curr_acc)
        test_loss_history.append(curr_loss)

    # Sigh...I shouldn't be doing this but oh well
    predicted_labels = [int(val) for val in predicted_labels]
    actual_labels = [int(val) for val in actual_labels]

    res_dict = {'predicted': predicted_labels, 'actual': actual_labels}
    with open(f'../../data/processed/v{VERSION}/test_output_v{VERSION}.json', 'w') as file_obj:
        json.dump(res_dict, file_obj)


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
    test_fp = f"../../data/processed/v{VERSION}"

    batch_size = 8
    num_classes = 5
    feature_extract = False
    model ='efficient'

    model_ft, input_size = initialize_model(num_classes, feature_extract, use_pretrained=False, model=model)
    model_ft.load_state_dict(torch.load(f'../../models/check_{VERSION}_{model}'))

    data_transforms = {
        'test': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Create testing dataset
    image_datasets = {x: datasets.ImageFolder(os.path.join(test_fp, x), data_transforms[x]) for x in ['test']}
    # Create testing dataloader
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
        ['test']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Set up the loss function
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    test_model(model_ft, dataloaders_dict, criterion, device=device)


if __name__ == '__main__':
    main()
