import torch
import os
import torch.multiprocessing as mp
import pickle
from torchvision import transforms
import numpy as np
from aoddataset import AODDataset 
import matplotlib.pyplot as plt
import sys

def log(message):
    verbose = True 
    if verbose:
        print(message)
    else:
        with open("logs/" + str(os.getpid()) + ".log", "a") as logfile:
            logfile.write(str(message) + "\n")

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

        im = im.reshape(im.size(0), -1)
        mask = mask.reshape(mask.size(0), -1)
        x = torch.from_numpy(np.append(im.detach().numpy(), mask.detach().numpy(), axis=1))
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        pred = self.linear_2(x)
        return pred



mean_val = 0.1623

def test(model):

    testset = None
    if os.path.isfile("data/mask_test"):
        with open("data/mask_test", "rb") as test_pickle:
            log("Opening test_pickle")
            testset = pickle.load(test_pickle)
            log("Done")
    else:
        testset = AODDataset("data/testing/")
        with open("data/test_pickle", "wb") as test_pickle:
            pickle.dump(testset, test_pickle)

    valset, testset = torch.utils.data.random_split(testset, [int(0.9 * len(testset)), len(testset) - int(0.9 * len(testset))])

    val_dataloader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    log("Validation dataset size: {}".format(len(valset)))
    log("Testing dataset size: {}".format(len(testset)))
    # test model
    model.load_state_dict(torch.load(sys.argv[1]))
    print("Loaded model")
    model.eval()

    criterion = torch.nn.MSELoss()
    results = list()
    goodness = 0
    negs = 0
    for itr, (image, label) in enumerate(test_dataloader):

        if (torch.cuda.is_available()):
            image = image.cuda()
            label = label.cuda()

        pred = model(image)
        loss = criterion(pred, label)
        print(loss)

        for i in range(len(label)):
            results.append([image[0][i], pred[i].detach().numpy(), label[i].numpy()])
            classification = 100 * (label[i].numpy().item() - mean_val) * (pred[i].detach().numpy().item() - mean_val)
            print(classification < 0)
            negs += 1 if classification < 0 else 0
            goodness += classification
        log("Done with itr {}".format(itr))
    print("GOODNESS")
    print(goodness)
    print(negs) 

    # visualize results
    fig=plt.figure(figsize=(20, 10))
    for i in range(1, 11):
        img = transforms.ToPILImage(mode='L')(results[i][0].squeeze(0).detach().cpu())
        fig.add_subplot(2, 5, i)
        plt.title("pred: {:.4f}, act: {:.4f}".format(results[i][1].item(), results[i][2].item()))
        plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    num_processes = 4
    model = Model()

    if (torch.cuda.is_available()):
        log("Using cuda")
        model.cuda()

    test(model)


