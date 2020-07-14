import torch
import os
import torch.multiprocessing as mp
import pickle
from torchvision import transforms
import numpy as np
from aoddataset import AODDataset 
import sys

def log(message):
    verbose = False 
    if verbose:
        print(message)
    else:
        with open("logs/" + str(sys.argv[1]) + str(os.getpid()) + ".log", "a") as logfile:
            logfile.write(str(message) + "\n")

def saveStats(stats):
    with open(str(sys.argv[1]) + ".stat", "wb") as pfile:
        pickle.dump(stats, pfile)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=10, stride=2)
        self.conv_2 = torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, stride=2)
        self.linear_1 = torch.nn.Linear(8 * 21 * 16 + 80 * 100, 128)
        self.linear_2 = torch.nn.Linear(128, 1)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        im = x[0].unsqueeze(1)
        mask = x[1].unsqueeze(1).float()
        mask = self.conv_1(mask)
        mask = self.relu(mask)
        mask = self.conv_2(mask)
        mask = self.relu(mask)

        num_pts = np.count_nonzero(x[1])
        mean = np.sum(x[0].detach().numpy()) / num_pts

        im = im.reshape(im.size(0), -1)
        mask = mask.reshape(mask.size(0), -1)
        x = torch.from_numpy(np.append(im.detach().numpy(), mask.detach().numpy(), axis=1))
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        pred = self.linear_2(x)
        return mean + pred

def train(model):
    trainset = None
    if os.path.isfile("data/mask_train"):
        with open("data/mask_train", "rb") as train_pickle:
            log("Opening train_pickle")
            trainset = pickle.load(train_pickle)
            log("Done")
    else:
        trainset = AODDataset("data/training/")
        with open("data/mask_train", "wb") as train_pickle:
            pickle.dump(trainset, train_pickle) 

    testset = None
    if os.path.isfile("data/mask_test"):
        with open("data/mask_test", "rb") as test_pickle:
            log("Opening test_pickle")
            testset = pickle.load(test_pickle)
            log("Done")
    else:
        testset = AODDataset("data/testing/")
        with open("data/mask_test", "wb") as test_pickle:
            pickle.dump(testset, test_pickle)

    valset, testset = torch.utils.data.random_split(testset, [int(0.9 * len(testset)), len(testset) - int(0.9 * len(testset))])

    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    log("Training dataset size: {}".format(len(trainset)))
    log("Validation dataset size: {}".format(len(valset)))
    log("Testing dataset size: {}".format(len(testset)))

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


    no_epochs = 150
    train_loss = list()
    val_loss = list()
    best_val_loss = 1
    for epoch in range(no_epochs):
        total_train_loss = 0
        total_val_loss = 0

        model.train()

        # Training for an epoch
        for itr, (image, label) in enumerate(train_dataloader):

            if (torch.cuda.is_available()):
                image = image.cuda()
                label = label.cuda()

            optimizer.zero_grad()

            pred = model(image)
            loss = criterion(pred, label)
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            log("Ran iteration {}/{}, with loss of {}. Current total_train_loss: {}".format(itr, len(trainset) / 128.0, loss.item(), total_train_loss))

        total_train_loss = total_train_loss / (itr + 1)
        train_loss.append(total_train_loss)

        # Validation with a smaller and completely separate subset
        model.eval()
        total = 0
        for itr, (image, label) in enumerate(val_dataloader):
            if (torch.cuda.is_available()):
                image = image.cuda()
                label = label.cuda()

            pred = model(image)

            log(pred)
            log(label)
            loss = criterion(pred, label)
            total_val_loss += loss.item()

            pred = torch.nn.functional.softmax(pred, dim=1)
            for i, p in enumerate(pred):
                if label[i] == torch.max(p.data, 0)[1]:
                    total = total + 1

        accuracy = total / len(valset)

        total_val_loss = total_val_loss / (itr + 1)
        val_loss.append(total_val_loss)

        log('\nEpoch: {}/{}, Train Loss: {:.8f}, Val Loss: {:.8f}, Val Accuracy: {:.8f}'.format(epoch + 1, no_epochs, total_train_loss, total_val_loss, accuracy))

        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss

            log("Saving the model state dictionary for Epoch: {} with Validation loss: {:.8f}".format(epoch + 1, total_val_loss))
            torch.save(model.state_dict(), str(sys.argv[1]) + ".dth")
    stats = {"tot_train_loss":train_loss, "tot_val_loss":val_loss, "best_val_loss":best_val_loss}
    saveStats(stats)

if __name__ == "__main__":
    model = Model()

    if (torch.cuda.is_available()):
        log("Using cuda")
        model.cuda()

    train(model)
