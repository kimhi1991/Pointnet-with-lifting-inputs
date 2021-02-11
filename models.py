import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, input):
        # input.shape == (bs,n,3)
        bs = input.size(0)
        xb = F.relu(self.bn1(self.conv1(input)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        pool = nn.MaxPool1d(xb.size(-1))(xb)
        flat = nn.Flatten(1)(pool)
        xb = F.relu(self.bn4(self.fc1(flat)))
        xb = F.relu(self.bn5(self.fc2(xb)))

        # initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(bs, 1, 1)
        if xb.is_cuda:
            init = init.cuda()
        matrix = self.fc3(xb).view(-1, self.k, self.k) + init
        return matrix

class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input, 1, 2), matrix3x3).transpose(1, 2)

        xb = F.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb, 1, 2), matrix64x64).transpose(1, 2)

        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        output = nn.Flatten(1)(xb)
        return output, matrix3x3, matrix64x64

class PointNet(nn.Module):
    def __init__(self, train_loader, val_loader, classes=40,sampled_data=False):
        super(PointNet, self).__init__()
        self.model_name = 'PointNet'
        self.train_loader = train_loader
        self.valid_loader = val_loader
        self.best_model = None
        self.sampled_data=sampled_data #if we use a database already sampled points

        self.transform = Transform()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        xb, matrix3x3, matrix64x64 = self.transform(input)
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        output = self.fc3(xb)
        return self.logsoftmax(output), matrix3x3, matrix64x64

    def pointnetloss(self, predictions, labels, m3x3, m64x64, alpha=0.0001):
        criterion = torch.nn.NLLLoss()
        bs = predictions.size(0)
        id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
        id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)
        if predictions.is_cuda:
            id3x3 = id3x3.cuda()
            id64x64 = id64x64.cuda()
        diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
        diff64x64 = id64x64 - torch.bmm(m64x64, m64x64.transpose(1, 2))
        return criterion(predictions, labels) + alpha * (torch.norm(diff3x3) + torch.norm(diff64x64)) / float(bs)

    def train_all(self, epochs=10, with_val=True):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        best_acc = -1
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            total = correct = 0
            for i, data in enumerate(self.train_loader, 0):
                if not self.sampled_data:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    optimizer.zero_grad()
                    outputs, m3x3, m64x64 = self(inputs.transpose(1, 2))  # forward
                else:
                    inputs, labels = data#['pointcloud'].to(device).float(), data['category'].to(device)
                    optimizer.zero_grad()
                    outputs, m3x3, m64x64 = self(inputs)  # forward


                predicted = torch.argmax(outputs,1)

                total += labels.size(0)
                correct += (labels == predicted).sum().item()
                loss = self.pointnetloss(outputs, labels, m3x3, m64x64)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 10 == 9:
                    print(f'Epoch: {epoch + 1}, Batch: {i + 1}, loss: {running_loss / 10} Train accuracy: {correct/total}')
                    running_loss = 0.0
                    total = correct = 0

            # validation
            if with_val:
                val_acc = self.test_all()
                if val_acc > best_acc:
                    best_acc = val_acc
                    self.best_model = copy.deepcopy(self.state_dict())
                print('Valid accuracy: %d %%' % val_acc)
            torch.save(self.best_model, f"best_{self.model_name}_model.pth")

    def test_all(self):
        self.eval()
        correct = total = 0
        with torch.no_grad():
            for data in self.valid_loader:
                if not self.sampled_data:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs, _, _ = self(inputs.transpose(1, 2))  # forward
                    _, predicted = torch.max(outputs.data, 1)
                else:
                    inputs, labels = data
                    outputs, _, _ = self(inputs)  # forward
                    predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = 100. * correct / total
        return val_acc

class Momentnet(nn.Module):
    def __init__(self, train_loader, val_loader, classes=40,sampled_data=False):
        super(PointNet, self).__init__()
        self.model_name = 'PointNet'
        self.train_loader = train_loader
        self.valid_loader = val_loader
        self.best_model = None
        self.sampled_data=sampled_data #if we use a database already sampled points

        self.transform = Transform()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        xb, matrix3x3, matrix64x64 = self.transform(input)
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        output = self.fc3(xb)
        return self.logsoftmax(output), matrix3x3, matrix64x64

    def pointnetloss(self, predictions, labels, m3x3, m64x64, alpha=0.0001):
        criterion = torch.nn.NLLLoss()
        bs = predictions.size(0)
        id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
        id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)
        if predictions.is_cuda:
            id3x3 = id3x3.cuda()
            id64x64 = id64x64.cuda()
        diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
        diff64x64 = id64x64 - torch.bmm(m64x64, m64x64.transpose(1, 2))
        return criterion(predictions, labels) + alpha * (torch.norm(diff3x3) + torch.norm(diff64x64)) / float(bs)

    def train_all(self, epochs=10, with_val=True):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        best_acc = -1
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            total = correct = 0
            for i, data in enumerate(self.train_loader, 0):
                if not self.sampled_data:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    optimizer.zero_grad()
                    outputs, m3x3, m64x64 = self(inputs.transpose(1, 2))  # forward
                else:
                    inputs, labels = data#['pointcloud'].to(device).float(), data['category'].to(device)
                    optimizer.zero_grad()
                    outputs, m3x3, m64x64 = self(inputs)  # forward


                predicted = torch.argmax(outputs,1)

                total += labels.size(0)
                correct += (labels == predicted).sum().item()
                loss = self.pointnetloss(outputs, labels, m3x3, m64x64)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 10 == 9:
                    print(f'Epoch: {epoch + 1}, Batch: {i + 1}, loss: {running_loss / 10} Train accuracy: {correct/total}')
                    running_loss = 0.0
                    total = correct = 0

            # validation
            if with_val:
                val_acc = self.test_all()
                if val_acc > best_acc:
                    best_acc = val_acc
                    self.best_model = copy.deepcopy(self.state_dict())
                print('Valid accuracy: %d %%' % val_acc)
            torch.save(self.best_model, f"best_{self.model_name}_model.pth")

    def test_all(self):
        self.eval()
        correct = total = 0
        with torch.no_grad():
            for data in self.valid_loader:
                if not self.sampled_data:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs, _, _ = self(inputs.transpose(1, 2))  # forward
                    _, predicted = torch.max(outputs.data, 1)
                else:
                    inputs, labels = data
                    outputs, _, _ = self(inputs)  # forward
                    predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = 100. * correct / total
        return val_acc
