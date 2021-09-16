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
import math
from torchvision import transforms, utils
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite._utils import convert_tensor
from torch.utils.tensorboard import SummaryWriter


n_epochs = 8
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 100

BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 0.01
first_HL = 8
torch.backends.cudnn.enabled = True

# dataset classes for train and validation
class train_data(Dataset):

    def __init__(self, transform):

        self.x_train = np.load('x_train_ssn.npy')
        self.y_train = np.load('y_train_ssn.npy')
        #self.x_train = np.squeeze(self.x_train)
        self.x_train = np.transpose(self.x_train, (0, 3, 1, 2))
        self.x_train = np.expand_dims(self.x_train, axis=4)
        self.X_train = torch.from_numpy(self.x_train)
        self.Y_train = torch.from_numpy(self.y_train)
        self.transform = transform
    def __len__(self):

        return len(self.X_train)

    def __getitem__(self, idx):

        mel_spec = self.X_train[idx]

        label = self.Y_train[idx]
        label = torch.argmax(label)

        sample = {'data': mel_spec, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample


class val_data(Dataset):

    def __init__(self, transform):

        self.x_test = np.load('x_val_ssn.npy')
        self.y_test = np.load('y_val_ssn.npy')
        self.x_test = np.transpose(self.x_test, (0, 3, 1, 2))
        self.x_test = np.expand_dims(self.x_test, axis=4)
        self.Y_val = torch.from_numpy(self.y_test)
        self.X_val = torch.from_numpy(self.x_test)
        self.transform = transform
    def __len__(self):

        return len(self.X_val)

    def __getitem__(self, idx):

        mel_spec = self.X_val[idx]
        label = self.Y_val[idx]
        label = torch.argmax(label)

        sample = {'data': mel_spec, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample

class test_data(Dataset):

    def __init__(self, transform):

        self.x_test = np.load('x_test_ssn.npy')
        self.y_test = np.load('y_test_ssn.npy')
        self.x_test = np.transpose(self.x_test, (0, 3, 1, 2))
        self.x_test = np.expand_dims(self.x_test, axis=4)
        self.y_test = torch.from_numpy(self.y_test)
        self.x_test = torch.from_numpy(self.x_test)
        self.transform = transform
    def __len__(self):

        return len(self.x_test)

    def __getitem__(self, idx):

        mel_spec = self.x_test[idx]
        label = self.y_test[idx]
        label = torch.argmax(label)

        sample = {'data': mel_spec, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample
 
 
class ToSubSpectrograms(object):
    """ Generate Sub-Spectrogram Tensors """

    def __init__(self, sub_spectrogram_size, sub_spectrogram_mel_hop, n_mel_bins):
        """
        Parameters
        ----------
        sub_spectrogram_size : int
            Size of the SubSpectrogram. Default: 20
        sub_spectrogram_mel_hop : int
            Mel-bin hop size of the SubSpectrogram. Default 10
        n_mel_bins : int
            Number of mel-bins of the Spectrogram extracted. Default: 40.
        """
        self.sub_spectrogram_size, self.sub_spectrogram_mel_hop, self.n_mel_bins = sub_spectrogram_size, sub_spectrogram_mel_hop, n_mel_bins

    def __call__(self, sample):
        """
        Parameters
        ----------
        sample : PyTorch tensor
            The input tensor data and label
        Returns
        -------
        sub_spectrograms: tensor
            A list of sub-spectrograms. Default size [channels, sub_spectrogram_size, time_indices, n_sub_spectrograms]
        label: tensor
            Corresponding label
        """
        spectrogram, label = sample['data'], sample['label']

        i = 0
        sub_spectrograms = torch.from_numpy(np.asarray([]))
        while (self.sub_spectrogram_mel_hop * i <= self.n_mel_bins - self.sub_spectrogram_size):

            # Extract a Sub-Spectrogram
            subspectrogram = spectrogram[:,
                             i * self.sub_spectrogram_mel_hop:i * self.sub_spectrogram_mel_hop + self.sub_spectrogram_size,
                             :, :]

            if i == 0:
                sub_spectrograms = subspectrogram
            else:
                sub_spectrograms = torch.cat((subspectrogram, sub_spectrograms), 3)

            i = i + 1

        return sub_spectrograms, label


def create_summary_writer(model, data_loader, log_dir):
	"""
	Create the summary writer for TensorBoard
	Parameters
	----------
	model : PyTorch model object
		Size of the training batch.
	data_loader : data_loader
		Data loader object to create the graph
	log_dir : str
		Directory to save the logs
	Returns
	-------
	train_loader and val_loader
		data loading objects
	"""
	writer = SummaryWriter(log_dir=log_dir)
	data_loader_iter = iter(data_loader)
	x, y = next(data_loader_iter)
	try:
		writer.add_graph(model, x)
	except Exception as e:
		print("Failed to save model graph: {}".format(e))
	return writer


class SubSpectralNet(nn.Module):
    """ SubSpectralNet architecture """

    def __init__(self, sub_spectrogram_size, sub_spectrogram_mel_hop, n_mel_bins, use_gpu):
        """
        Init the model layers

        Parameters
        ----------
        sub_spectrogram_size : int
            Size of the SubSpectrogram. Default: 20

        sub_spectrogram_mel_hop : int
            Mel-bin hop size of the SubSpectrogram. Default 10
        n_mel_bins : int
            Number of mel-bins of the Spectrogram extracted. Default: 40.
        use_gpu : Bool
            Use GPU or not. Default: True
        """
        super(SubSpectralNet, self).__init__()
        self.sub_spectrogram_size, self.sub_spectrogram_mel_hop, self.n_mel_bins, self.use_gpu = sub_spectrogram_size, sub_spectrogram_mel_hop, n_mel_bins, use_gpu

        # For max-pool after the second conv-layer
        self.n_max_pool = int(self.sub_spectrogram_size / 10)

        # Number of SubSpectrograms: used for defining the number of conv-layers
        self.n_sub_spectrograms = 0

        while (self.sub_spectrogram_mel_hop * self.n_sub_spectrograms <= self.n_mel_bins - self.sub_spectrogram_size):
            self.n_sub_spectrograms = self.n_sub_spectrograms + 1

        # init the layers
        self.conv1 = nn.ModuleList(
            [nn.Conv2d(in_channels=2, out_channels=32, kernel_size=7, stride=1, padding=3) for _ in
             range(self.n_sub_spectrograms)])
        self.conv1_bn = nn.ModuleList([nn.BatchNorm2d(32) for _ in range(self.n_sub_spectrograms)])
        self.mp1 = nn.ModuleList([nn.MaxPool2d((self.n_max_pool, 5)) for _ in range(self.n_sub_spectrograms)])
        self.drop1 = nn.ModuleList([nn.Dropout(0.3) for _ in range(self.n_sub_spectrograms)])
        self.conv2 = nn.ModuleList(
            [nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3) for _ in
             range(self.n_sub_spectrograms)])
        self.conv2_bn = nn.ModuleList([nn.BatchNorm2d(64) for _ in range(self.n_sub_spectrograms)])
        self.mp2 = nn.ModuleList([nn.MaxPool2d((4, 100)) for _ in range(self.n_sub_spectrograms)])
        self.drop2 = nn.ModuleList([nn.Dropout(0.3) for _ in range(self.n_sub_spectrograms)])

        self.fc1 = nn.ModuleList([nn.Linear(1 * 2 * 64, 32) for _ in range(self.n_sub_spectrograms)])
        self.drop3 = nn.ModuleList([nn.Dropout(0.3) for _ in range(self.n_sub_spectrograms)])
        self.fc2 = nn.ModuleList([nn.Linear(32, 10) for _ in range(self.n_sub_spectrograms)])

        numFCs = int(math.log(self.n_sub_spectrograms * 32, 2))
        neurons = int(math.pow(2, numFCs))

        self.fcGlobal = []
        tempNeurons = int(32 * self.n_sub_spectrograms)
        while (neurons >= 64):
            self.fcGlobal.append(nn.Linear(tempNeurons, neurons))
            self.fcGlobal.append(nn.ReLU(0.3))
            self.fcGlobal.append(nn.Dropout(0.3))
            tempNeurons = neurons
            neurons = int(neurons / 2)
        self.fcGlobal.append(nn.Linear(tempNeurons, 10))
        self.fcGlobal = nn.ModuleList(self.fcGlobal)

    def forward(self, x):
        """
        Feed-forward pass
        Parameters
        ----------
        x : tensor
            Input batch. Size: [batch_size, channels, sub_spectrogram_size, n_time_indices, n_sub_spectrograms]. Default [16, 1, 20, 500, 3]
        Returns
        -------
        outputs: tensor
            final output of the model. Size: [batch_size, n_sub_spectrograms, n_labels]. Default: [16, 4, 10]
        """
        outputs = []
        intermediate = []
        x = x.float()
        if self.use_gpu:
            x = x.cuda()
        input_var = x

        # for every sub-spectrogram
        for i in range(x.shape[4]):
            x = input_var
            x = self.conv1[i](x[:, :, :, :, i])
            x = self.conv1_bn[i](x)
            x = F.relu(x)
            x = self.mp1[i](x)
            x = self.drop1[i](x)
            x = self.conv2[i](x)
            x = self.conv2_bn[i](x)
            x = F.relu(x)
            x = self.mp2[i](x)
            x = self.drop2[i](x)
            x = x.view(-1, 1 * 2 * 64)
            x = self.fc1[i](x)
            x = F.relu(x)
            intermediate.append(x)
            x = self.drop3[i](x)
            x = self.fc2[i](x)
            x = x.view(-1, 1, 10)
            outputs.append(x)

        # extracted intermediate layers
        x = torch.cat((intermediate), 1)

        # global classification
        for i in range(len(self.fcGlobal)):
            x = self.fcGlobal[i](x)
        x = x.view(-1, 1, 10)
        outputs.append(x)

        # all the outputs (low, mid and high band + global classifier)
        outputs = torch.cat((outputs), 1)
        outputs = F.log_softmax(outputs, dim=2)
        return outputs


class SubSpectralNet_spinal(nn.Module):
    """ SubSpectralNet architecture """

    def __init__(self, sub_spectrogram_size, sub_spectrogram_mel_hop, n_mel_bins, use_gpu):
        """
        Init the model layers

        Parameters
        ----------
        sub_spectrogram_size : int
            Size of the SubSpectrogram. Default: 20

        sub_spectrogram_mel_hop : int
            Mel-bin hop size of the SubSpectrogram. Default 10
        n_mel_bins : int
            Number of mel-bins of the Spectrogram extracted. Default: 40.
        use_gpu : Bool
            Use GPU or not. Default: True
        """
        super(SubSpectralNet_spinal, self).__init__()
        self.sub_spectrogram_size, self.sub_spectrogram_mel_hop, self.n_mel_bins, self.use_gpu = sub_spectrogram_size, sub_spectrogram_mel_hop, n_mel_bins, use_gpu

        # For max-pool after the second conv-layer
        self.n_max_pool = int(self.sub_spectrogram_size / 10)

        # Number of SubSpectrograms: used for defining the number of conv-layers
        self.n_sub_spectrograms = 0

        while (self.sub_spectrogram_mel_hop * self.n_sub_spectrograms <= self.n_mel_bins - self.sub_spectrogram_size):
            self.n_sub_spectrograms = self.n_sub_spectrograms + 1

        # init the layers
        self.conv1 = nn.ModuleList(
            [nn.Conv2d(in_channels=2, out_channels=32, kernel_size=7, stride=1, padding=3) for _ in
             range(self.n_sub_spectrograms)])
        self.conv1_bn = nn.ModuleList([nn.BatchNorm2d(32) for _ in range(self.n_sub_spectrograms)])
        self.mp1 = nn.ModuleList([nn.MaxPool2d((self.n_max_pool, 5)) for _ in range(self.n_sub_spectrograms)])
        self.drop1 = nn.ModuleList([nn.Dropout(0.3) for _ in range(self.n_sub_spectrograms)])
        self.conv2 = nn.ModuleList(
            [nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3) for _ in
             range(self.n_sub_spectrograms)])
        self.conv2_bn = nn.ModuleList([nn.BatchNorm2d(64) for _ in range(self.n_sub_spectrograms)])
        self.mp2 = nn.ModuleList([nn.MaxPool2d((4, 100)) for _ in range(self.n_sub_spectrograms)])
        self.drop2 = nn.ModuleList([nn.Dropout(0.3) for _ in range(self.n_sub_spectrograms)])

        self.fc1 = nn.ModuleList([nn.Linear(1 * 2 * 64, 32) for _ in range(self.n_sub_spectrograms)])

        self.fc1 = nn.ModuleList([nn.Linear(64, first_HL)for _ in range(self.n_sub_spectrograms)])  # changed from 16 to 8
        self.fc1_1 = nn.ModuleList([nn.Linear(64 + first_HL, first_HL)for _ in range(self.n_sub_spectrograms)])  # added
        self.fc1_2 = nn.ModuleList([nn.Linear(64 + first_HL, first_HL)for _ in range(self.n_sub_spectrograms)])  # added
        self.fc1_3 = nn.ModuleList([nn.Linear(64 + first_HL, first_HL)for _ in range(self.n_sub_spectrograms)])  # added
        self.fc1_4 = nn.ModuleList([nn.Linear(64 + first_HL, first_HL)for _ in range(self.n_sub_spectrograms)]) # added
        self.fc1_5 = nn.ModuleList([nn.Linear(64 + first_HL, first_HL)for _ in range(self.n_sub_spectrograms)])  # added
        self.drop3 = nn.ModuleList([nn.Dropout(0.3) for _ in range(self.n_sub_spectrograms)])
        self.fc2 = nn.ModuleList([nn.Linear(first_HL * 6, 10) for _ in range(self.n_sub_spectrograms)])

        self.soft = nn.Softmax(dim=1)


        numFCs = int(math.log(self.n_sub_spectrograms * 48, 2))
        neurons = int(math.pow(2, numFCs))

        self.fcGlobal = []
        tempNeurons = int(48 * self.n_sub_spectrograms)
        while (neurons >= 64):
            self.fcGlobal.append(nn.Linear(tempNeurons, neurons))
            self.fcGlobal.append(nn.ReLU(0.3))
            self.fcGlobal.append(nn.Dropout(0.3))
            tempNeurons = neurons
            neurons = int(neurons / 2)
        self.fcGlobal.append(nn.Linear(tempNeurons, 10))
        self.fcGlobal = nn.ModuleList(self.fcGlobal)

    def forward(self, x):
        """
        Feed-forward pass
        Parameters
        ----------
        x : tensor
            Input batch. Size: [batch_size, channels, sub_spectrogram_size, n_time_indices, n_sub_spectrograms]. Default [16, 1, 20, 500, 3]
        Returns
        -------
        outputs: tensor
            final output of the model. Size: [batch_size, n_sub_spectrograms, n_labels]. Default: [16, 4, 10]
        """
        outputs = []
        intermediate = []
        x = x.float()
        if self.use_gpu:
            x = x.cuda()
        input_var = x

        # for every sub-spectrogram
        for i in range(x.shape[4]):
            x = input_var
            x = self.conv1[i](x[:, :, :, :, i])
            x = self.conv1_bn[i](x)
            x = F.relu(x)
            x = self.mp1[i](x)
            x = self.drop1[i](x)
            x = self.conv2[i](x)
            x = self.conv2_bn[i](x)
            x = F.relu(x)
            x = self.mp2[i](x)
            x = self.drop2[i](x)
            x = x.view(-1, 1 * 2 * 64)

            x1 = x[:, 0:64]

            x1 = F.relu(self.fc1[i](x1))
            x2 = torch.cat([x[:, 64:128], x1], dim=1)
            x2 = F.relu(self.fc1_1[i](x2))
            x3 = torch.cat([x[:, 0:64], x2], dim=1)
            x3 = F.relu(self.fc1_2[i](x3))
            x4 = torch.cat([x[:, 64:128], x3], dim=1)
            x4 = F.relu(self.fc1_3[i](x4))
            x5 = torch.cat([x[:, 0:64], x4], dim=1)
            x5 = F.relu(self.fc1_4[i](x5))
            x6 = torch.cat([x[:, 64:128], x5], dim=1)
            x6 = F.relu(self.fc1_5[i](x6))

            x = torch.cat([x1, x2], dim=1)
            x = torch.cat([x, x3], dim=1)
            x = torch.cat([x, x4], dim=1)
            x = torch.cat([x, x5], dim=1)
            x = torch.cat([x, x6], dim=1)

            #logits = self.fc2(x)

            #x = self.fc1[i](x)
            #x = F.relu(x)
            intermediate.append(x)
            x = self.drop3[i](x)
            x = self.fc2[i](x)
            x = x.view(-1, 1, 10)
            outputs.append(x)

        # extracted intermediate layers
        x = torch.cat((intermediate), 1)

        # global classification
        for i in range(len(self.fcGlobal)):
            x = self.fcGlobal[i](x)
        x = x.view(-1, 1, 10)
        outputs.append(x)

        # all the outputs (low, mid and high band + global classifier)
        outputs = torch.cat((outputs), 1)
        outputs = F.log_softmax(outputs, dim=2)
        return outputs

def prepare_batch(batch, device=None, non_blocking=False):
	"""
	Inbuilt function in the ignite._utils, for converting the data to tensors.
	Returns the tensors of the input data, using convert_tensor function.
	"""
	x, y = batch
	return (convert_tensor(x, device=device, non_blocking=non_blocking),
		convert_tensor(y, device=device, non_blocking=non_blocking))

# run function
def run(train_dataloader, val_dataloader, train_batch_size, test_batch_size, epochs, lr, log_interval, log_dir, no_cuda, sub_spectrogram_size,
        sub_spectrogram_mel_hop, n_mel_bins, seed):
    """
    Model runner
    Parameters
    ----------
    train_batch_size : int
        Size of the training batch. Default: 16
    test_batch_size : int
        size of the testing batch. Default: 16
    epochs : int
        Number of training epochs. Default: 200
    lr : float
        Learning rate for the ADAM optimizer. Default: 0.001
    log_interval : int
        Interval for logging data: Default: 10
    log_dir : str
        Directory to save the logs
    no_cuda : Bool
        Should you NOT use cuda? Default: False
    sub_spectrogram_size : int
        Size of the SubSpectrogram. Default 20

    sub_spectrogram_mel_hop : int
        Mel-bin hop size of the SubSpectrogram. Default 10
    n_mel_bins : int
        Number of mel-bins of the Spectrogram extracted. Default: 40.
    seed : int
        Torch random seed value, for reproducable results. Default: 1
    root_dir : str
        Directory of the folder which contains the dataset (has 'audio' and 'evaluation_setup' folders inside)
    train_dir : str
        Set as default: 'evaluation_setup/train_fold1.txt'
    eval_dir : str
        Set as default: 'evaluation_setup/evaluate_fold1.txt'
    """

    # check if possible to use CUDA
    use_cuda = not no_cuda and torch.cuda.is_available()

    # set seed
    torch.manual_seed(seed)

    # Map to GPU
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load the data loaders
    train_loader= train_dataloader
    val_loader = val_dataloader

    # Get the model
    model = SubSpectralNet(sub_spectrogram_size, sub_spectrogram_mel_hop, n_mel_bins, use_cuda).to(device)

    # Init the TensorBoard summary writer
    writer = create_summary_writer(model, train_loader, log_dir)

    # Init the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Use GPU if possible
    if device:
        model.to(device)

    def update_model(engine, batch):
        """Prepare batch for training: pass to a device with options.
        """
        model.train()
        optimizer.zero_grad()

        inputs, label = prepare_batch(batch, device=device)
        output = model(inputs)
        losses = []
        for ite in range(output.shape[1]):
            losses.append(F.nll_loss(output[:, ite, :], label))
        loss = sum(losses)
        loss.backward()
        optimizer.step()
        return losses, output

    # get the trainer module
    trainer = Engine(update_model)

    def evaluate(engine, batch):
        """Prepare batch for training: pass to a device with options.
        """
        model.eval()
        with torch.no_grad():
            inputs, label = prepare_batch(batch, device=device)
            output = model(inputs)
            losses = []
            correct = []
            for ite in range(output.shape[1]):
                losses.append(F.nll_loss(output[:, ite, :], label, reduction='sum').item())
        return losses, output, label

    # get the evaluator module
    evaluator = Engine(evaluate)

    # define output transforms for multiple outputs.
    def output_transform1(output):
        # `output` variable is returned by above `process_function`
        losses, correct, label = output
        return correct[:, 0, :], label

    metric = Accuracy(output_transform=output_transform1)
    metric.attach(evaluator, "acc_highband")
    metric = Loss(F.nll_loss, output_transform=output_transform1)
    metric.attach(evaluator, "loss_highband")

    def output_transform2(output):
        # `output` variable is returned by above `process_function`
        losses, correct, label = output
        return correct[:, 1, :], label

    metric = Accuracy(output_transform=output_transform2)
    metric.attach(evaluator, "acc_midband")
    metric = Loss(F.nll_loss, output_transform=output_transform2)
    metric.attach(evaluator, "loss_midband")

    def output_transform3(output):
        # `output` variable is returned by above `process_function`
        losses, correct, label = output
        return correct[:, 2, :], label

    metric = Accuracy(output_transform=output_transform3)
    metric.attach(evaluator, "acc_lowband")
    metric = Loss(F.nll_loss, output_transform=output_transform3)
    metric.attach(evaluator, "loss_lowband")

    def output_transform(output):
        # `output` variable is returned by above `process_function`
        losses, correct, label = output
        return correct[:, 3, :], label

    metric = Accuracy(output_transform=output_transform)
    metric.attach(evaluator, "acc_globalclassifier")
    metric = Loss(F.nll_loss, output_transform=output_transform)
    metric.attach(evaluator, "loss_globalclassifier")

    # Log the events in Ignite: EVERY ITERATION
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % log_interval == 0:
            losses, output = engine.state.output
            epoch = engine.state.epoch
            print(
                'Train Epoch: {} [{}/{}]\tLosses: {:.6f} (Top Band), {:.6f} (Mid Band), {:.6f} (Low Band), {:.6f} (Global Classifier)'.format(
                    epoch, iter, len(train_loader), losses[0].item(), losses[1].item(), losses[2].item(),
                    losses[3].item()))
            # TensorBoard Logs
            writer.add_scalar("training/loss_topband_itr", losses[0].item(), engine.state.iteration)
            writer.add_scalar("training/loss_midband_itr", losses[1].item(), engine.state.iteration)
            writer.add_scalar("training/loss_lowband_itr", losses[2].item(), engine.state.iteration)
            writer.add_scalar("training/loss_global_itr", losses[3].item(), engine.state.iteration)

    # Log the events in Ignite: Test the training data on EVERY EPOCH
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        print("Training Results - Epoch: {}  Global accuracy: {:.2f} Avg loss: {:.2f}"
              .format(engine.state.epoch, evaluator.state.metrics['acc_globalclassifier'],
                      evaluator.state.metrics['loss_globalclassifier']))
        # TensorBoard Logs
        writer.add_scalar("training/global_loss", evaluator.state.metrics['loss_globalclassifier'], engine.state.epoch)
        writer.add_scalar("training/lowband_loss", evaluator.state.metrics['loss_lowband'], engine.state.epoch)
        writer.add_scalar("training/midband_loss", evaluator.state.metrics['loss_midband'], engine.state.epoch)
        writer.add_scalar("training/highband_loss", evaluator.state.metrics['loss_highband'], engine.state.epoch)
        writer.add_scalar("training/global_acc", evaluator.state.metrics['acc_globalclassifier'], engine.state.epoch)
        writer.add_scalar("training/lowband_acc", evaluator.state.metrics['acc_lowband'], engine.state.epoch)
        writer.add_scalar("training/midband_acc", evaluator.state.metrics['acc_midband'], engine.state.epoch)
        writer.add_scalar("training/highband_acc", evaluator.state.metrics['acc_highband'], engine.state.epoch)

    # Log the events in Ignite: Test the validation data on EVERY EPOCH
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        print("Validation Results - Epoch: {}  Global accuracy: {:.2f} Avg loss: {:.2f}"
              .format(engine.state.epoch, evaluator.state.metrics['acc_globalclassifier'],
                      evaluator.state.metrics['loss_globalclassifier']))
        # TensorBoard Logs
        writer.add_scalar("validation/global_loss", evaluator.state.metrics['loss_globalclassifier'],
                          engine.state.epoch)
        writer.add_scalar("validation/lowband_loss", evaluator.state.metrics['loss_lowband'], engine.state.epoch)
        writer.add_scalar("validation/midband_loss", evaluator.state.metrics['loss_midband'], engine.state.epoch)
        writer.add_scalar("validation/highband_loss", evaluator.state.metrics['loss_highband'], engine.state.epoch)
        writer.add_scalar("validation/global_acc", evaluator.state.metrics['acc_globalclassifier'], engine.state.epoch)
        writer.add_scalar("validation/lowband_acc", evaluator.state.metrics['acc_lowband'], engine.state.epoch)
        writer.add_scalar("validation/midband_acc", evaluator.state.metrics['acc_midband'], engine.state.epoch)
        writer.add_scalar("validation/highband_acc", evaluator.state.metrics['acc_highband'], engine.state.epoch)

    # kick everything off
    trainer.run(train_loader, max_epochs=epochs)

    # close the writer
    writer.close()

    # return the model
    return model


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    # turn off gradients for validation
    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
            # forward pass
            prediction = model(input)
            prediction = prediction[:,3,:]
            print(prediction.shape)
            # validation batch loss
            #loss = loss_fn(prediction, torch.max(target, 1)[1])
            # accumulate the valid_loss
            #test_loss += loss.item()
            # calculate the accuracy
            predicted = torch.argmax(prediction, 1)
            print(predicted.shape)
            correct += (predicted == target).sum().item()

    ########################
    ## PRINT TEST RESULTS ##
    ########################
    print(len(test_loader.dataset))
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print(f'Test loss: {test_loss}.. Test Accuracy: {accuracy}')

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    sub_spectrogram_size = 20
    sub_spectrogram_mel_hop = 10
    n_mel_bins=40

    data_transform = transforms.Compose([ToSubSpectrograms(sub_spectrogram_size, sub_spectrogram_mel_hop, n_mel_bins)])
    devset = train_data(transform=data_transform)
    valset = val_data(transform= data_transform)

    train_dataloader = DataLoader(devset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True)

    # get number of classifiers (number of sub-spectrograms + 1 for global classifier)
    numClassifiers = 0
    while (sub_spectrogram_mel_hop * numClassifiers <= n_mel_bins - sub_spectrogram_size):
        numClassifiers = numClassifiers + 1
    # + 1 for global classifier
    numClassifiers = numClassifiers + 1

    # Run the model
    #model = run(args.batch_size, args.test_batch_size, args.epochs, args.lr, args.log_interval, args.log_dir,
    #            args.no_cuda, sub_spectrogram_size, sub_spectrogram_mel_hop, n_mel_bins, args.seed, root_dir, train_dir,
    #            eval_dir)

    model = run(train_dataloader, val_dataloader, 16, 16, 200, 0.001, 10, "tensorboard_logs", False, 20, 10, 40, 1)

    test(model,test_loader)

    torch.save(model.state_dict(), "subspectralnet_dual_cnn.pt")



