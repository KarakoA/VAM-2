import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/
#
def main1():
    train_dataset = dsets.MNIST(root='./data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.MNIST(root='./data',
                               train=False,
                               transform=transforms.ToTensor())


    batch_size = 100
    n_iters = 3000


    num_epochs = n_iters / (len(train_dataset) / batch_size)
    num_epochs = int(num_epochs)
    print(num_epochs)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
            super(LSTMModel, self).__init__()
            # Hidden dimensions
            self.hidden_dim = hidden_dim

            # Number of hidden layers
            self.layer_dim = layer_dim

            # Building your LSTM
            # batch_first=True causes input/output tensors to be of shape
            # (batch_dim, seq_dim, feature_dim)
            self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

            # Readout layer
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            # Initialize hidden state with zeros
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

            # Initialize cell state
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

            # 28 time steps
            # We need to detach as we are doing truncated backpropagation through time (BPTT)
            # If we don't, we'll backprop all the way to the start even after going through another batch
            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

            # Index hidden state of last time step
            # out.size() --> 100, 28, 100
            # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
            out = self.fc(out[:, -1, :])
            # out.size() --> 100, 10
            return out


    input_dim = 28
    hidden_dim = 100
    layer_dim = 1
    output_dim = 10

    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)

    criterion = nn.CrossEntropyLoss()

    learning_rate = 0.1

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    # Number of steps to unroll
    seq_dim = 28

    iter = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Load images as a torch tensor with gradient accumulation abilities
            images = images.view(-1, seq_dim, input_dim).requires_grad_()

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            # outputs.size() --> 100, 10
            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            iter += 1

            if iter % 100 == 0:
                # Calculate Accuracy
                correct = 0
                total = 0
                # Iterate through test dataset
                for images, labels in test_loader:
                    # Resize images
                    images = images.view(-1, seq_dim, input_dim)

                    # Forward pass only to get logits/output
                    outputs = model(images)

                    # Get predictions from the maximum value
                    _, predicted = torch.max(outputs.data, 1)

                    # Total number of labels
                    total += labels.size(0)

                    # Total correct predictions
                    correct += (predicted == labels).sum()

                accuracy = 100 * correct.item() / total

                # Print Loss
                print('Epoch {} Iteration: {}. Loss: {}. Accuracy: {}'.format(epoch,iter, loss.item(), accuracy))



def main2_no_detach():
    train_dataset = dsets.MNIST(root='./data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.MNIST(root='./data',
                               train=False,
                               transform=transforms.ToTensor())


    batch_size = 100
    n_iters = 3000


    num_epochs = n_iters / (len(train_dataset) / batch_size)
    num_epochs = int(num_epochs)
    print(num_epochs)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
            super(LSTMModel, self).__init__()
            # Hidden dimensions
            self.hidden_dim = hidden_dim

            # Number of hidden layers
            self.layer_dim = layer_dim

            # Building your LSTM
            # batch_first=True causes input/output tensors to be of shape
            # (batch_dim, seq_dim, feature_dim)
            self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

            # Readout layer
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            # Initialize hidden state with zeros
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

            # Initialize cell state
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

            # 28 time steps
            # We need to detach as we are doing truncated backpropagation through time (BPTT)
            # If we don't, we'll backprop all the way to the start even after going through another batch
            out, (hn, cn) = self.lstm(x, (h0, c0))

            # Index hidden state of last time step
            # out.size() --> 100, 28, 100
            # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
            out = self.fc(out[:, -1, :])
            # out.size() --> 100, 10
            return out


    input_dim = 28
    hidden_dim = 100
    layer_dim = 1
    output_dim = 10

    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)

    criterion = nn.CrossEntropyLoss()

    learning_rate = 0.1

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    # Number of steps to unroll
    seq_dim = 28

    iter = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Load images as a torch tensor with gradient accumulation abilities
            images = images.view(-1, seq_dim, input_dim).requires_grad_()

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            # outputs.size() --> 100, 10
            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            iter += 1

            if iter % 100 == 0:
                # Calculate Accuracy
                correct = 0
                total = 0
                # Iterate through test dataset
                for images, labels in test_loader:
                    # Resize images
                    images = images.view(-1, seq_dim, input_dim)

                    # Forward pass only to get logits/output
                    outputs = model(images)

                    # Get predictions from the maximum value
                    _, predicted = torch.max(outputs.data, 1)

                    # Total number of labels
                    total += labels.size(0)

                    # Total correct predictions
                    correct += (predicted == labels).sum()

                accuracy = 100 * correct.item() / total

                # Print Loss
                print('Epoch {} Iteration: {}. Loss: {}. Accuracy: {}'.format(epoch,iter, loss.item(), accuracy))

def main3():
    train_dataset = dsets.MNIST(root='./data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.MNIST(root='./data',
                               train=False,
                               transform=transforms.ToTensor())


    batch_size = 100
    n_iters = 3000


    num_epochs = n_iters / (len(train_dataset) / batch_size)
    num_epochs = int(num_epochs)
    print(num_epochs)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
            super(LSTMModel, self).__init__()
            # Hidden dimensions
            self.hidden_dim = hidden_dim

            # Number of hidden layers
            self.layer_dim = layer_dim

            # Building your LSTM
            # batch_first=True causes input/output tensors to be of shape
            # (batch_dim, seq_dim, feature_dim)
            self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

            # Readout layer
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):

            # Initialize hidden state with zeros
            hn = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)#.requires_grad_()

            # Initialize cell state
            cn = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)#.requires_grad_()

            for i in range(28):
                # 28 time steps
                # We need to detach as we are doing truncated backpropagation through time (BPTT)
                # If we don't, we'll backprop all the way to the start even after going through another batch
                out, (hn, cn) = self.lstm(x[:,i,:].unsqueeze(1), (hn, cn))

            # Index hidden state of last time step
            # out.size() --> 100, 28, 100
            # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
            out = self.fc(out[:, -1, :])
            # out.size() --> 100, 10
            return out


    input_dim = 28
    hidden_dim = 100
    layer_dim = 1
    output_dim = 10

    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)

    criterion = nn.CrossEntropyLoss()

    learning_rate = 0.1

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    # Number of steps to unroll
    seq_dim = 28

    iter = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Load images as a torch tensor with gradient accumulation abilities
            images = images.view(-1, seq_dim, input_dim).requires_grad_()

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            # outputs.size() --> 100, 10
            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            iter += 1

            if iter % 100 == 0:
                # Calculate Accuracy
                correct = 0
                total = 0
                # Iterate through test dataset
                for images, labels in test_loader:
                    # Resize images
                    images = images.view(-1, seq_dim, input_dim)

                    # Forward pass only to get logits/output
                    outputs = model(images)

                    # Get predictions from the maximum value
                    _, predicted = torch.max(outputs.data, 1)

                    # Total number of labels
                    total += labels.size(0)

                    # Total correct predictions
                    correct += (predicted == labels).sum()

                accuracy = 100 * correct.item() / total

                # Print Loss
                print('Epoch {} Iteration: {}. Loss: {}. Accuracy: {}'.format(epoch,iter, loss.item(), accuracy))

if __name__ == '__main__':
    main3()