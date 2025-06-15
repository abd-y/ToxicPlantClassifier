'''
Dataset Link: https://www.kaggle.com/datasets/hanselliott/toxic-plant-classification
'''

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import torch
import pandas as pd
import numpy as np
from PIL import Image
import time, datetime
import matplotlib.pyplot as plt
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

class ToxicPlantDataset(Dataset):
    def __init__(self, X, y, transform=None, augmentation=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = Image.open(self.X.iloc[idx])
        if self.transform:
            original_image = self.transform(image)
        
        if self.augmentation:
            augmented_image = self.augmentation(image)
            return original_image, augmented_image, self.y[idx]
        else:
            return original_image, self.y[idx]

class ModelLoader:
    def __init__(self, classes:list):
        self.classes = classes
        self.num_classes = len(classes)
        self.loaded_model = None
    
    def _vit(self, model, dropout):
        model.heads = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.heads.head.in_features, self.num_classes)
        )

        return model

    def vit_b_16(self, dropout=0.4):
        self.loaded_model = "vit_b_16"
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model = self._vit(model, dropout)

        return model


    def vit_l_32(self, dropout=0.4):
        self.loaded_model = "vit_l_32"
        model = models.vit_l_32(weights=models.ViT_L_32_Weights.DEFAULT)
        model = self._vit(model, dropout)

        return model
    
    def _vgg(self, model):
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, self.num_classes)
        return model

    def vgg16(self):
        self.laoded_model = "vgg16"
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model = self._vgg(model)

        return model

    def vgg19(self):
        self.laoded_model = "vgg19"
        model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        model = self._vgg(model)

        return model

    def _efficientnet(self, model):
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, self.num_classes)
        return model

    def efficientnet_b0(self):
        self.loaded_model = "efficientnet_b0"
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model = self._efficientnet(model)

        return model
    
    def efficientnet_b3(self):
        self.loaded_model = "efficientnet_b3"
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        model = self._efficientnet(model)

        return model

    def efficientnet_b4(self):
        self.loaded_model = "efficientnet_b4"
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        model = self._efficientnet(model)

        return model
    
    def efficientnet_v2_l(self):
        self.loaded_model = "efficientnet_v2_l"
        model = models.efficientnet_v2_l()
        model = self._efficientnet(model)

        return model


def prepare_binary_classification():
    df = pd.read_csv("./tpc-imgs/full_metadata.csv")
    X = df["path"].apply(lambda x: f"./{x.split("../input/toxic-plant-classification/")[1]}")
    y = df[["toxicity"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train, y_test = torch.Tensor(y_train.values.astype(np.int32)).to(device), torch.Tensor(y_test.values.astype(np.int32)).to(device)

    return ["nontoxic", "toxic"], X_train, X_test, y_train, y_test

def prepare_multi_classification():
    df = pd.read_csv("./tpc-imgs/full_metadata.csv")
    X = df["path"].apply(lambda x: f"./{x.split("../input/toxic-plant-classification/")[1]}")
    y = df[["scientific_name"]]

    encoder = LabelEncoder()
    encoder.fit_transform(y["scientific_name"])

    y.loc[:, "scientific_name"] = encoder.transform(y["scientific_name"])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train, y_test = torch.Tensor(y_train.values.astype(np.int32)).to(device), torch.Tensor(y_test.values.astype(np.int32)).to(device)
    
    return encoder.classes_.tolist(), X_train, X_test, y_train, y_test

# image_size = (224, 224)
# image_size = (380, 380)
image_size = (300, 300)

transformer = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize(image_size),
                                      transforms.Normalize(std=[0.229, 0.224, 0.225], mean=[0.485, 0.456, 0.406])
                                      ])

def to_numpy_safe(x):
    if isinstance(x, torch.Tensor):
        x = x.detach()  # remove from computation graph if necessary
        if x.is_cuda:
            x = x.cpu()
        return x.numpy()
    return x  # already a NumPy array or list

def make_plots(history):
    epochs = to_numpy_safe(history["Epoch"])
    train_loss = to_numpy_safe(history["Train_loss"])
    train_acc = to_numpy_safe(history["Train_acc"])
    test_acc = to_numpy_safe(history["Test_acc"])
    test_f1 = to_numpy_safe(history["Test-f1"])

    # Plot 1: Train Loss
    plt.figure()
    plt.plot(epochs, train_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Train Loss")
    plt.savefig("loss.png")
    plt.close()

    # Plot 2: Accuracy and F1 Score
    plt.figure()
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, test_acc, label="Test Accuracy")
    plt.plot(epochs, test_f1, label="Test F1-Score")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Accuracy and F1 Score")
    plt.legend()
    plt.savefig("results.png")
    plt.close()

def accuracy(y_pred, y_true):
    return (y_pred == y_true).float().mean()

def main():
    history = {"Epoch": [], "Train_loss": [], "Train_acc": [], "Test_acc": [], "Test-f1": []}
    batch_size = 16
    epochs = 100
    lr = 0.00001

    # classes, X_train, X_test, y_train, y_test = prepare_multi_classification()
    classes, X_train, X_test, y_train, y_test = prepare_binary_classification()

    augmentation = transforms.Compose([transforms.Resize(image_size),
                                      transforms.RandomHorizontalFlip(0.5),
                                      transforms.RandomVerticalFlip(0.5),
                                      transforms.ColorJitter(contrast=0.2),
                                      transforms.ToTensor(),
                                      transforms.Normalize(std=[0.229, 0.224, 0.225], mean=[0.485, 0.456, 0.406])
                                      ])
    
    train = ToxicPlantDataset(X_train, y_train, transformer, augmentation)
    test = ToxicPlantDataset(X_test, y_test, transformer)
    train_loader = DataLoader(train, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)

    model_loader = ModelLoader(classes)
    model = model_loader.efficientnet_b3()
    model = model.to(device)
    

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    try:
        for epoch in range(epochs):
            model.train()

            total_loss = 0
            total_train_acc = 0
            total_test_acc = 0
            start_time = time.time()

            for x, augmented_x, y in train_loader:
                x, y = x.to(device), y.to(device)
                augmented_x = augmented_x.to(device)

                optimizer.zero_grad()

                original_logits = model(x)
                augmented_logits = model(augmented_x)
                original_y_preds = torch.argmax(original_logits, dim=1)
                augmented_y_preds = torch.argmax(augmented_logits, dim=1)
                y_true = y.squeeze().long()
                loss = (loss_fn(original_logits, y_true) + loss_fn(augmented_logits, y_true)) / 2
                total_train_acc += (accuracy(original_y_preds, y_true) + accuracy(augmented_y_preds, y_true)) / 2

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            
            total_loss = total_loss / len(train_loader)
            total_train_acc = total_train_acc / len(train_loader)

            all_preds = []
            all_labels = []

            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    y_preds = torch.argmax(logits, dim=1)
                    y_true = y.squeeze().long()

                    total_test_acc += accuracy(y_preds, y_true)

                    # Collect predictions and labels
                    all_preds.extend(y_preds.cpu().numpy())
                    all_labels.extend(y_true.cpu().numpy())

            # Average accuracy
            total_test_acc = total_test_acc / len(test_loader)

            f1 = f1_score(all_labels, all_preds, average='binary')
            end_time = time.time()
            processing_time = str(datetime.timedelta(seconds=end_time - start_time))

            history["Epoch"], history["Train_loss"], history["Train_acc"], history["Test_acc"], history["Test-f1"] = epoch + 1, total_loss, total_train_acc, total_test_acc, f1

            print(f"Epoch {epoch + 1} | Loss {total_loss:6f} | Train Accuracy {total_train_acc:6f} | Test Accuracy {total_test_acc:6f} | Test F1-Score {f1:6f} | Epoch Processing time {processing_time}")

        torch.save(model.state_dict(), "model.pth")
        make_plots(history)

    except KeyboardInterrupt:
        torch.save(model.state_dict(), "model.pth")
        make_plots(history)

def load_and_evaluate():
    classes, _, X_test, _, y_test = prepare_binary_classification()
    model = ModelLoader(classes)
    model = model.efficientnet_b3()
    model.load_state_dict(torch.load("./models/binary-classification/model.pth"), strict=False)

    test = ToxicPlantDataset(X_test, y_test, transformer)
    test_loader = DataLoader(test)
    
    model.to(device)
    model.eval()

    total_test_acc = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            y_preds = torch.argmax(logits, dim=1)
            y_true = y.squeeze().long()

            # Accumulate accuracy
            total_test_acc += accuracy(y_preds, y_true)

            preds_np = y_preds.cpu().numpy()
            labels_np = y_true.cpu().numpy()

            if preds_np.ndim == 0:
                all_preds.append(preds_np.item())
            else:
                all_preds.extend(preds_np)

            if labels_np.ndim == 0:
                all_labels.append(labels_np.item())
            else:
                all_labels.extend(labels_np)

    total_test_acc = total_test_acc / len(test_loader)

    f1 = f1_score(all_labels, all_preds, average='binary')

    print(f"Test Accuracy: {total_test_acc:.4f}")
    print(f"Test F1 Score: {f1:.4f}")


def test_the_model(image_path:str):
    classes = ["nontoxic", "toxic"]
    metadata = pd.read_csv("./tpc-imgs/full_metadata.csv")
    metadata = metadata[['path', 'toxicity']]
    
    checker = os.path.normpath(f"../input/toxic-plant-classification/{image_path}").replace("\\", "/")
    y_true = int(metadata[metadata["path"] == checker]["toxicity"].iloc[0])

    model = ModelLoader(classes)
    model = model.efficientnet_b3()
    model.load_state_dict(torch.load("./models/binary-classification/model.pth"), strict=False)

    image = transforms.ToTensor()(Image.open(image_path)).unsqueeze(0).to(device)
    
    model.to(device)
    model.eval()

    with torch.no_grad():
        logit = model(image)
        y_pred = torch.argmax(logit, dim=1)

        print("Predicted:", classes[y_pred])
        print("Real value:", classes[y_true])
    


if __name__ == "__main__":
    # main()
    # load_and_evaluate()
    test_the_model("./tpc-imgs/toxic_images/003/990.jpg")