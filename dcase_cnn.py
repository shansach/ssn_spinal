import torch
from torch import  nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
from torchsummary import summary
import torch.nn.functional as F
import torch.optim as optim

n_epochs = 8
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 100

BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.01
first_HL = 8
torch.backends.cudnn.enabled = True


class train_data(Dataset):

    def __init__(self):

        self.x_train = np.load('x_train_ssn.npy')
        self.y_train = np.load('y_train_ssn.npy')
        #self.x_train = np.squeeze(self.x_train)
        #self.x_train = np.expand_dims(self.x_train, axis=1)
        self.x_train = np.transpose(self.x_train, (0,3,1,2))
        self.X_train = torch.from_numpy(self.x_train)
        self.Y_train = torch.from_numpy(self.y_train)

    def __len__(self):

        return len(self.X_train)

    def __getitem__(self, idx):

        mel_spec = self.X_train[idx]
        label = self.Y_train[idx]

        return mel_spec, label

class val_data(Dataset):

    def __init__(self):

        self.x_test = np.load('x_val_ssn.npy')
        self.y_test = np.load('y_val_ssn.npy')
        self.x_test = np.transpose(self.x_test, (0, 3, 1, 2))
        self.Y_val = torch.from_numpy(self.y_test)
        self.X_val = torch.from_numpy(self.x_test)

    def __len__(self):

        return len(self.X_val)

    def __getitem__(self, idx):

        mel_spec = self.X_val[idx]
        label = self.Y_val[idx]

        return mel_spec, label


class test_data(Dataset):

    def __init__(self):

        self.x_test = np.load('x_test_ssn.npy')
        self.y_test = np.load('y_test_ssn.npy')
        self.x_test = np.transpose(self.x_test, (0, 3, 1, 2))
        self.y_test = torch.from_numpy(self.y_test)
        self.x_test = torch.from_numpy(self.x_test)

    def __len__(self):

        return len(self.x_test)

    def __getitem__(self, idx):

        mel_spec = self.x_test[idx]
        label = self.y_test[idx]


        return  mel_spec, label



class base_model_spinal_Net(nn.Module):
    def __init__(self):
        super(spinal_Net2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding='same'
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5),
            nn.Dropout2d(p=0.3, inplace=False)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding='same'
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4,100)),
            nn.Dropout2d(p=0.3, inplace=False)
        )
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(64, first_HL)  # changed from 16 to 8
        self.fc1_1 = nn.Linear(64 + first_HL, first_HL)  # added
        self.fc1_2 = nn.Linear(64 + first_HL, first_HL)  # added
        self.fc1_3 = nn.Linear(64 + first_HL, first_HL)  # added
        self.fc1_4 = nn.Linear(64 + first_HL, first_HL)  # added
        self.fc1_5 = nn.Linear(64 + first_HL, first_HL)  # added
        self.fc2 = nn.Linear(first_HL * 6, 10)  # changed first_HL from second_HL
        self.soft = nn.Softmax(dim=1)
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x= self.conv1(x)
        x= self.conv2(x)
        # converts the matrix into a 320 unit vector
        x = x.view(-1, 128)

        x1 = x[:, 0:64]

        x1 = F.relu(self.fc1(x1))
        x2 = torch.cat([x[:, 64:128], x1], dim=1)
        x2 = F.relu(self.fc1_1(x2))
        x3 = torch.cat([x[:, 0:64], x2], dim=1)
        x3 = F.relu(self.fc1_2(x3))
        x4 = torch.cat([x[:, 64:128], x3], dim=1)
        x4 = F.relu(self.fc1_3(x4))
        x5 = torch.cat([x[:, 0:64], x4], dim=1)
        x5 = F.relu(self.fc1_4(x5))
        x6 = torch.cat([x[:, 64:128], x5], dim=1)
        x6 = F.relu(self.fc1_5(x6))

        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)
        x = torch.cat([x, x5], dim=1)
        x = torch.cat([x, x6], dim=1)

        logits = self.fc2(x)
        prediction = self.soft(logits)

        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        return logits, prediction




class base_model(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding='same'
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5),
            nn.Dropout2d(p=0.3, inplace=False)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding='same'
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4,100)),
            nn.Dropout2d(p=0.3, inplace=False)
        )

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128, 10)
        self.finaldrop = nn.Dropout(0.3)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)

        x = self.flatten(x)
        logits = self.linear(x)
        logits = self.finaldrop(logits)
        predictions = self.softmax(logits)

        return logits, predictions

def train(model, data_loader,test_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        train_loss=0
        print(f"Epoch {i+1}")

        for input, target in data_loader:
            input, target = input.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
            model.train()
            # calculate loss
            logits, prediction = model(input)
            loss = loss_fn(logits, torch.max(target, 1)[1])
            train_loss += loss.item()
            # backpropagate error and update weights
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        print(f"loss: {loss.item()}")



        # print epoch results
        train_loss /= len(data_loader)

        print(f'Epoch: {i + 1}/{EPOCHS}.. Training loss: {train_loss}')
        print("---------------------------")

    print("Finished training")

    print("RESULTS")

    # set the model to eval mode
    model.eval()
    test_loss = 0
    correct = 0
    # turn off gradients for validation
    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
            # forward pass
            logits, prediction = model(input)
            # validation batch loss
            loss = loss_fn(logits, torch.max(target, 1)[1])
            # accumulate the valid_loss
            test_loss += loss.item()
            # calculate the accuracy
            predicted = torch.argmax(prediction, 1)
            correct += (predicted == torch.max(target, 1)[1]).sum().item()

    ########################
    ## PRINT TEST RESULTS ##
    ########################
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print(f'Test loss: {test_loss}.. Test Accuracy: {accuracy}')


def test(model, test_loader,loss_fn):

    # set the model to eval mode
    model.eval()
    test_loss = 0
    correct = 0
    # turn off gradients for validation
    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
            # forward pass
            logits, prediction = model(input)
            # validation batch loss
            loss = loss_fn(prediction, torch.max(target, 1)[1])
            # accumulate the valid_loss
            test_loss += loss.item()
            # calculate the accuracy
            predicted = torch.argmax(prediction, 1)
            correct += (predicted == torch.max(target, 1)[1]).sum().item()

    ########################
    ## PRINT TEST RESULTS ##
    ########################
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print(f'Test loss: {test_loss}.. Test Accuracy: {accuracy}')

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    devset = train_data()

    valset = val_data()

    testset = test_data()

    train_dataloader = DataLoader(devset,batch_size=BATCH_SIZE,shuffle=True)


    val_dataloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataloader = DataLoader(testset, batch_size=200, shuffle=True )


    # construct model and assign it to device
    cnn = base_model2().to(device)





    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)



    # train model (model, data_loader,val_loader, loss_fn, optimiser, device, epochs)
    train(cnn, train_dataloader,val_dataloader, loss_fn, optimiser, device, EPOCHS)

    #cnn.load_state_dict(torch.load("base_model_weights.pth"))
    test(cnn, test_dataloader, loss_fn)
    torch.save(cnn.state_dict(), "spinal_net2_weights.pth")
    






