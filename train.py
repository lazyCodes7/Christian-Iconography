import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch.nn.functional as F
import torchvision.transforms.functional as FT
from utils.collector import DatasetCollector
from utils.collector import ArtDLDataset
from utils.transform import Transform
from utils.clf import ArtDLClassifier
import torch
import torch.optim as optim
class Trainer:
    def __init__(self, data_dir = None, labels_path = None):
        tf = Transform()
        train_dataset = ArtDLDataset(
            data_dir = data_dir,
            transform = tf.transform,
            labels_path = labels_path,
            set_type = 'train'
        )

        test_dataset = ArtDLDataset(
            data_dir = data_dir,
            transform = tf.val_transform,
            labels_path = labels_path,
            set_type = 'test'
        )

        val_dataset = ArtDLDataset(
            data_dir = data_dir,
            transform = val_transform,
            labels_path = labels_path,
            set_type = 'val'
        )
        self.train_loader = DataLoader(dataset = train_dataset, shuffle=True, batch_size = 50)
        self.test_loader = DataLoader(dataset = test_dataset, batch_size = 1)
        self.val_loader = DataLoader(dataset = val_dataset, batch_size = 10)
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.model = ArtDLClassifier(num_classes = 19).to(device)
            
    def train(self):
        optimizer = optim.SGD(clf.trainable_params(), lr = 0.01, momentum = 0.9)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            # Setting the train mode
            self.model.train()
            train_loss = 0
            val_loss = 0
            for idx, (image, label) in enumerate(self.train_loader):
                image = image.to(device)
                #print(image.shape)
                label = label.to(device)

                # Zeroing the gradients before re-computing them
                optimizer.zero_grad()
                outputs = self.model(image).squeeze()

                # Calculating the loss
                loss = criterion(outputs, label)
                train_loss += loss.item()

                # Calculating the gradients == diff(loss w.r.t weights)
                loss.backward()

                # Updating the weights
                optimizer.step()
                
                with torch.no_grad():
                    self.model.eval()
                    val_score = 0
                    for idx, (image, label) in enumerate(self.val_loader):
                        image = image.to(device)
                        label = label.to(device)
                        outputs = self.model(image).squeeze()

                        # Getting the predictions
                        pred = outputs.argmax(dim = 1, keepdim = True)

                        # Updating scores and losses
                        val_score += pred.eq(label.view_as(pred)).sum().item()
                        loss = criterion(outputs, label)
                        val_loss += loss.item()
                
            print("=================================================")
            print("Epoch: {}".format(epoch+1))
            print("Validation Loss: {}".format(val_loss/len(self.val_loader)))
            print("Training Loss: {}".format(train_loss/len(self.train_loader)))
            print("Validation Accuracy: {}".format((val_score)/len(self.val_loader)*10))
            
    def test(self):
        self.model.eval()
        test_score = 0
        img_count = 0
        images = []
        preds = []
        labels = []
        for idx, (image, label) in enumerate(self.test_loader):
            image = image.to(device)
            label = label.to(device)
            outputs = model(image).squeeze()
            #print(outputs)
            pred = outputs.argmax()
            preds.append(pred.item())
            labels.append(label.item())
            #print(pred)
            if(pred == label):
            if(test_score<10):
                images.append(image)
            test_score+=1

        print("Test Accuracy {:.3f}".format(test_score/len(test_loader)))

        return preds, labels, images
    

    