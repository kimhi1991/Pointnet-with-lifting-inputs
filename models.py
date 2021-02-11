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
    def __init__(self, train_loader, val_loader, classes=40,lr = 1e-4,alpha = 1e-4,sampled_data=False):
        super(PointNet, self).__init__()
        self.model_name = 'PointNet'
        self.train_loader = train_loader
        self.valid_loader = val_loader
        self.lr = lr
        self.alpha = alpha #hyper parameter for the PN loss
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

    def pointnetloss(self, predictions, labels, m3x3, m64x64):
        criterion = torch.nn.NLLLoss()
        bs = predictions.size(0)
        id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
        id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)
        if predictions.is_cuda:
            id3x3 = id3x3.cuda()
            id64x64 = id64x64.cuda()
        diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
        diff64x64 = id64x64 - torch.bmm(m64x64, m64x64.transpose(1, 2))
        return criterion(predictions, labels) + self.alpha * (torch.norm(diff3x3) + torch.norm(diff64x64)) / float(bs)

    def train_all(self, epochs=10, with_val=True):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
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
    def __init__(self, train_loader, val_loader, classes=40,lr = 1e-4,alpha = 1e-4,sampled_data=False):
        super(Momentnet, self).__init__()
        self.model_name = 'PointNet'
        self.train_loader = train_loader
        self.valid_loader = val_loader
        self.lr = lr
        self.alpha = alpha #hyper parameter for the PN loss
        self.best_model = None
        self.sampled_data=sampled_data #if we use a database already sampled points
        self.k = 20

        self.transform = Transform()
        self.conv1 = nn.Conv1d(12, 64, 1) #12 is for second order

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def concat_moment(self,input, moment=2):
        """
        lift input with concatinate to higher moments
        :param input: tensor shape (btz x n x 3)
        :param moment: what moment to lift (1/2/3)
        :return: tensor shape of (btz x n x lifting_size)
        when lifting_size =3 for first moment, 9 for second moment and 19 for third
        """
        if moment == 1:
            return input
        else:
            print(input.shape)
            a = (input * input).to(device)  # x^2,y^2,z^2
            b1 = (input[:, :, 0] * input[:, :, 1]).to(device)  # xy
            b2 = (input[:, :, 0] * input[:, :, 2]).to(device)  # xz
            b3 = (input[:, :, 1] * input[:, :, 2]).to(device)  # yz
            b = torch.stack((b1, b2, b3)).T.transpose(0, 1).to(device)
            second_moment = torch.cat((input.T, a.T, b.T)).T.to(device)
            if moment == 2:
                return second_moment
            a = (input ** 3).to(device)  # x^3 y^3 z^3
            b1 = (input[:, :, 0] ** 2 * input[:, :, 1]).to(device)  # x^2y
            b2 = (input[:, :, 0] ** 2 * input[:, :, 2]).to(device)  # x^2z
            b = torch.stack((b1, b2)).T.transpose(0, 1).to(device)
            c1 = (input[:, :, 1] ** 2 * input[:, :, 0]).to(device)  # y^2x
            c2 = (input[:, :, 1] ** 2 * input[:, :, 2]).to(device)  # y^2z
            c = torch.stack((c1, c2)).T.transpose(0, 1).to(device)
            d1 = (input[:, :, 2] ** 2 * input[:, :, 0]).to(device)  # z^2x
            d2 = (input[:, :, 2] ** 2 * input[:, :, 1]).to(device)  # z^2y
            d = torch.stack((d1, d2)).T.transpose(0, 1).to(device)
            e = torch.prod(input, 2).to(device)  # xyz
            third_moment = torch.cat((second_moment.T, a.T, b.T, c.T, d.T, e.unsqueeze(2).T)).T.to(device)
        return third_moment

    def knn(self, input):
        inner = -2 * torch.matmul(input.transpose(2, 1), input)
        xx = torch.sum(input ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)

        idx = pairwise_distance.topk(k=self.k, dim=-1)[1]  # (batch_size, num_points, k)
        return idx

    def forward(self, input):

        #imput size: btz,3,n
        #create the higher_order layer, here for second moment
        xb_moment =self.concat_moment(input.transpose(1,2),moment=2) #btz,n,9
        xb_moment = xb_moment.repeat(self.k,1,1,1).transpose(0,1).transpose(1,2).float()#.to(device) # (btz,n,k,9)
        xb_knn = self.knn(input) #should be (btz,n,k,3)
        xb = self.conv1(torch.cat((xb_moment,xb_knn)))
        #xb = nn.MaxPool1d(xb.size(-1))(xb)
        xb = F.adaptive_max_pool1d(xb, 1).squeeze()
        #=======end of aditional=======
        xb, matrix3x3, matrix64x64 = self.transform(xb)
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
