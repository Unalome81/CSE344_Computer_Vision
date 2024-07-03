import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import wandb

import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score

import torch.optim as optim

from torchvision.models import resnet18, ResNet18_Weights

import seaborn as sns
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import confusion_matrix

from sklearn.manifold import TSNE

from torch.utils.data import ConcatDataset


# Taking input and storing it in as Image Paths and labels

file_path = 'Cropped_final'
class_order = ['amur_leopard', 'amur_tiger', 'birds', 'black_bear', 'brown_bear', 'dog', 'roe_deer', 'sika_deer', 'wild_boar', 'people']

Image_Path_List = []
labels = []

for i in range(len(class_order)):
    class_name = class_order[i]
    class_dir = os.path.join(file_path, class_name)
    for file_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, file_name)
        Image_Path_List.append(image_path)
        labels.append(i)

# Splitting the input into train, validation and testing sets
        
X_train, X_val_test, y_train, y_val_test = train_test_split(Image_Path_List, labels, test_size=0.3, random_state=42, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=1/3, random_state=42, stratify=y_val_test)


# factor = 20
# n_train = len(X_train) // factor               # this was used to take a fraction of the data, for testing the code
# n_val = len(X_val) // factor
# X_train = X_train[:n_train]
# y_train = y_train[:n_train]
# X_val = X_val[:n_val] 
# y_val = y_val[:n_val]

image_height = 64
image_width = 64

transform = transforms.Compose([
    transforms.Resize((image_height, image_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Building a custom dataset to preprocess the data and store it as tensors using its file paths
class CustomImageDataset(Dataset):
    def __init__(self, Image_Paths, labels, transform=None):
        self.Image_Paths = Image_Paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.Image_Paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Building training, validation and testing datasets
train_dataset = CustomImageDataset(X_train, y_train, transform=transform)
val_dataset = CustomImageDataset(X_val, y_val, transform=transform)
test_dataset = CustomImageDataset(X_test, y_test, transform=transform)


batch_size = 64

# Building Dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Plotting the class distribution of training and validation sets

def class_distribution(y, class_order, title):
    class_counts = np.bincount(y)
    plt.bar(range(len(class_counts)), class_counts)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.xticks(range(len(class_order)), class_order, rotation=25)

    plt.show()

class_distribution(y_train, class_order, "Training Dataset Distribution")
class_distribution(y_val, class_order, "Validation Dataset Distribution")


# Plotting the first batch of images from the training dataset

def Batch_Show(img):
    img /= 2
    img += 0.5
    img_np= img.numpy()
    plt.imshow(np.transpose(img_np, (1, 2, 0)))
    plt.axis('off')
    plt.show()

for images, labels in train_loader:
    Batch_Show(torchvision.utils.make_grid(images))
    break 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Training

def Training(model, criterion, optimizer, num_epochs, train_loader, val_loader, device, model_name):

    print("Training Starts")

    model.to(device)

    for epoch in range(num_epochs):
        model.train()

        train_loss = 0.0
        train_accuracy = 0.0
        total_train_labels = 0
        correct_train_labels = 0

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted_labels = torch.max(outputs.data, 1)
            total_train_labels += labels.size(0)
            correct_train_labels += (predicted_labels == labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = (correct_train_labels / total_train_labels) * 100

        wandb.log({f"{model_name} Training Loss": train_loss, f"{model_name} Training Accuracy": train_accuracy}, step=epoch)


        model.eval()

        val_loss = 0.0
        val_accuracy = 0.0
        total_val_labels = 0
        correct_val_labels = 0

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted_labels = torch.max(outputs.data, 1)
                total_val_labels += labels.size(0)
                correct_val_labels += (predicted_labels == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = (correct_val_labels / total_val_labels) * 100

        wandb.log({f"{model_name} Validation Loss": val_loss, f"{model_name} Validation Accuracy": val_accuracy}, step=epoch)


        print("Epoch", epoch + 1)
        print(f"{model_name} Training Loss = {train_loss}, {model_name} Validation Loss = {val_loss}")
        print(f"{model_name} Training Accuracy = {train_accuracy}, {model_name} Validation Accuracy = {val_accuracy}")

# Model Testing
        
def Testing(model, test_loader, device, model_name):
    print("Testing Starts")
    model.to(device)
    model.eval()

    test_f1_score = 0.0
    test_accuracy = 0.0

    total_test_labels = 0
    correct_test_labels = 0
    ground_truth = []
    predicted_labels_list = []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            _, predicted_labels = torch.max(outputs.data, 1)
            total_test_labels += labels.size(0)
            correct_test_labels += (predicted_labels == labels).sum().item()

            ground_truth.extend(labels.cpu().numpy())
            predicted_labels_list.extend(predicted_labels.cpu().numpy())

    test_accuracy = (correct_test_labels / total_test_labels) * 100
    test_f1_score = f1_score(ground_truth, predicted_labels_list, average='weighted')

    print(model_name, "Test Accuracy = ", test_accuracy)
    print(model_name, "Test F1 Score = ", test_f1_score)

    confusion = confusion_matrix(ground_truth, predicted_labels_list)

    plt.figure(figsize=(10, 10))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=class_order, yticklabels=class_order)
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title(f"{model_name} Confusion Matrix")

    image_path = f"{model_name}_confusion_matrix.png"
    plt.savefig(image_path, format='png')
    plt.close()

    wandb.log({f"{model_name} Confusion Matrix": wandb.Image(image_path)})
    os.remove(image_path)


# Defining CNN class

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, 3, stride = 1, padding=1)  
        self.conv3 = nn.Conv2d(64, 128, 3, stride = 1, padding=1)

        self.pool1 = nn.MaxPool2d(4, 4) 
        self.pool2 = nn.MaxPool2d(2, 2) 
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(128 * 4 * 4, 10)


    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x

# Defining Resnet class
    
class Resnet_Predictor(nn.Module):

    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.resnet18 = resnet18(weights=weights, progress=False)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 512)
        self.fc = nn.Linear(self.resnet18.fc.in_features, 10)


    def forward(self, x):
        x = self.resnet18(x)
        x = F.relu(x)
        x = self.fc(x)
        return x


# WandB initialization for CNN

wandb.login()
wandb.init(project='CV_A1_Image_Classification', entity='vivaswan21217')

print("CNN Model Begins: ")
cnn = CNN()
# cnn.load_state_dict(torch.load('cnn_model.pth'))

Training(cnn, nn.CrossEntropyLoss(), optim.Adam(cnn.parameters()), 10, train_loader, val_loader, device, "CNN")
# torch.save(cnn.state_dict(), 'cnn_model.pth')

Testing(cnn, test_loader, device, "CNN")
wandb.finish()


# WandB initialization for ResNet

wandb.init(project='CV_A1_Image_Classification', entity='vivaswan21217')

print("ResNet Model Begins: ")
resnet = Resnet_Predictor()
resnet = resnet.to(device)
resnet.load_state_dict(torch.load('resnet_model.pth'))

Training(resnet, nn.CrossEntropyLoss(), optim.Adam(resnet.parameters()), 10, train_loader, val_loader, device, "Resnet")
# torch.save(resnet.state_dict(), 'resnet_model.pth')

Testing(resnet, test_loader, device, "Resnet")
wandb.finish()


# Extracting Features for tSNE plotting

def Feature_Extraction(model, data_loader, device):
    model.fc = nn.Identity()
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            features_list.extend(outputs.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    return np.array(features_list), np.array(labels_list)

def Plotting_TSNE_2D(features_list, labels_list, title):
    tsne = TSNE(n_components=2, random_state=42)    

    embeddings = tsne.fit_transform(features_list)
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels_list, cmap='tab10', alpha=0.7)
    plt.title(title)
    plt.colorbar()
    plt.grid(True)
    plt.show()

def Plotting_TSNE_3D(features_list, labels_list, title):
    tsne = TSNE(n_components=3, random_state=42)
    embeddings = tsne.fit_transform(features_list)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=labels_list, cmap='tab10', alpha=0.7)
    plt.title(title)
    plt.colorbar(scatter)
    plt.grid(True)
    plt.show()


# Extracting features from the fine-tuned ResNet model
train_features, train_labels = Feature_Extraction(resnet, train_loader, device)
val_features, val_labels = Feature_Extraction(resnet, val_loader, device)
test_features, test_labels = Feature_Extraction(resnet, test_loader, device)

# Plotting TSNE plots
Plotting_TSNE_2D(train_features, train_labels, "Training Data, Global Average Pool layer Feature Space, tSNE 2D plot")
Plotting_TSNE_2D(val_features, val_labels, "Validation Data, Global Average Pool layer Feature Space, Feature Space tSNE 2D plot")
Plotting_TSNE_3D(test_features, test_labels, "Test Data, Global Average Pool layer Feature Space, Feature Space tSNE 3D Plot")
    

# Data Augmentation transform
transform_augmented = transforms.Compose([
    transforms.Resize((image_height, image_width)),
    
    transforms.RandomHorizontalFlip(p=0.5),  
    transforms.RandomRotation(degrees=(-10, 10)), 
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  

    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#Building augmented dataset

augmented_train_dataset = CustomImageDataset(X_train, y_train, transform=transform_augmented)

merged_train_dataset = ConcatDataset([train_dataset, augmented_train_dataset])

merged_train_loader = DataLoader(merged_train_dataset, batch_size=batch_size, shuffle=True)


wandb.init(project='CV_A1_Image_Classification', entity='vivaswan21217')

print("Augmented ResNet Model Begins: ")

resnet_augmented = Resnet_Predictor()
resnet_augmented = resnet_augmented.to(device)

# resnet_augmented.load_state_dict(torch.load('resnet_augmented_model.pth'))

Training(resnet_augmented, nn.CrossEntropyLoss(), optim.Adam(resnet_augmented.parameters()), 10, merged_train_loader, val_loader, device, "Augmented_Resnet")

# torch.save(resnet_augmented.state_dict(), 'resnet_augmented_model.pth')

Testing(resnet_augmented, test_loader, device, "Augmented_Resnet")

wandb.finish()

print("done")