import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from utils.plotter import *
from utils.plotter import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
#from torch.utils.tensorboard import SummaryWriter
#from datetime import datetime
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from sklearn.metrics import precision_score,recall_score



def get_lifting_dict():
    no_fun = lambda x: x
    log_fun = lambda x: -2 * torch.log(x)
    sin_fun = lambda x: torch.sin(x)
    sinpi_fun = lambda x: torch.sin(np.pi * x)
    exp_fun = lambda x: torch.exp(-0.2 * x)
    FUNCTIONS = [{'keep_dims': True, 'function': no_fun, 'name': ""},
                 {'keep_dims': False, 'function': sin_fun, 'name': "_sin_lift"},
                 {'keep_dims': False, 'function': exp_fun, 'name': "_exp_lift"},
                 {'keep_dims': False, 'function': sinpi_fun, 'name': "_sinpi_lift"},
                 {'keep_dims': False, 'function': log_fun, 'name': "_log_lift"}]
    return FUNCTIONS

#===================ARCHITECTURE FUNCTIONS=================#
def concat_moment(input, moment=1):
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
        #print(input.shape)
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

def knn(x,k):
    """
    return the k nn for each point in each batch of input x
    :param x: tensor of btz x 3 x n
    :param k: k for knn
    :return: btz x n x k
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    return feature -x
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

NO_LIFT = {'keep_dims': True,'function': lambda x: x,'name': ""}

def lift_with_fuc(input, lift_func=NO_LIFT):
    """
    lift input with concatinate to higher moments
    :param input: tensor shape (btz x n x chanels)
    :param keep_dims: True if we substitute the input, False for concatinage
    :param fun: the function we implement over all inputs
    :return: tensor shape of (btz x n x 2*channels) or (btz x n x channels) if keep_dims is True
    """
    lift = lift_func['function'](input)
    if lift_func['keep_dims']:
        return lift
    out = torch.cat((input,lift),dim=1)
    return out

#===================POINTNET CODE=================#

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
        xb = F.relu(self.bn1(self.conv1(input.to(device))))
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
    def __init__(self,v_normals=False, lift_func=NO_LIFT):
        super().__init__()
        self.v_normals   = v_normals
        self.in_channels = 6 if v_normals else 3
        if not lift_func['keep_dims']:
            self.in_channels *=2
        self.lift_func = lift_func


        self.input_transform = Tnet(k=3).to(device)
        self.feature_transform = Tnet(k=64).to(device)

        self.conv1 = nn.Conv1d(self.in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, input):
        if self.v_normals:
            xb, norms = torch.split(input, 3, dim=1)
            matrix3x3 = self.input_transform(xb)
            xb = torch.bmm(xb.transpose(1, 2), matrix3x3)
            norms = torch.bmm(norms.transpose(1, 2), matrix3x3)
            xb = torch.cat((xb,norms),dim=2).transpose(1, 2)
        else:
            matrix3x3 = self.input_transform(input)
            xb = torch.bmm(input.transpose(1, 2), matrix3x3).transpose(1, 2)
        xb = lift_with_fuc(xb,self.lift_func)
        xb = F.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb, 1, 2), matrix64x64).transpose(1, 2)

        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        output = nn.Flatten(1)(xb)
        return output, matrix3x3, matrix64x64

class PointNet(nn.Module):
    def __init__(self, train_loader, val_loader, classes=40,lr=1e-4,alpha=1e-4,v_normals=False,lift_func=NO_LIFT):
        super(PointNet, self).__init__()
        self.model_name = 'PointNet'
        if v_normals:
            self.model_name +='_normals'
        self.model_name +=lift_func['name']
        self.train_loader = train_loader
        self.valid_loader = val_loader
        self.lr = lr
        self.alpha = alpha #param for PN loss
        self.best_model = None

        self.transform = Transform(v_normals=v_normals,lift_func=lift_func).to(device)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        xb = input


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
        return criterion(predictions,labels.long()) + self.alpha * (torch.norm(diff3x3) + torch.norm(diff64x64)) / float(bs)

    def train_all(self, epochs=10, with_val=True):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        best_acc = -1
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            total = correct = 0
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data['data'],data['label']#['pointcloud'].to(device).float(), data['category'].to(device)
                optimizer.zero_grad()
                outputs, m3x3, m64x64 = self(inputs.transpose(1, 2))  # forward

                predicted = torch.argmax(outputs,1)

                total += labels.size(0)
                correct += (labels == predicted).sum().item()
                loss = self.pointnetloss(outputs, labels, m3x3, m64x64)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 10 == 9:

                    """summary_writer.add_scalar(f'Train Loss {self.model_name}', running_loss / 10, epoch * len(self.train_loader) + i)
                    summary_writer.add_scalar(f'Train Accuracy {self.model_name}', correct / total,epoch * len(self.train_loader) + i)"""
                    write_summerize(True, self.model_name, correct/total, epoch, i+1, running_loss/10)
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

    def test_all(self,summary_writer=None,cm=False):
        self.eval()
        correct = total = 0
        if cm:
            all_preds = []
            all_labels = []
        with torch.no_grad():
            for i,data in enumerate(self.valid_loader):
                inputs, labels = data['data'],data['label']#['pointcloud'].to(device).float(), data['category'].to(device)
                outputs, _, _ = self(inputs.transpose(1, 2))  # forward
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if cm:
                    all_preds += list(predicted.numpy())
                    all_labels += list(labels.numpy())
                if summary_writer:
                    summary_writer.add_scalar(f'Test Accuracy {self.model_name}', correct / total,  i)
        val_acc = 100. * correct / total
        write_summerize(False, self.model_name, accuracy=val_acc)
        if cm:
            cm = confusion_matrix(all_labels, all_preds)
            path = os.path.join('data', 'modelnet40_ply_hdf5_2048')
            classes = {i: class_name for i, class_name in
                       enumerate([line.rstrip() for line in open((os.path.join(path, 'shape_names.txt')))])}
            plot_confusion_matrix(cm, list(classes.keys()), normalize=True)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds,average='macro')
        print(f'{self.model_name} F1: {(2*precision * recall)/(precision + recall)}')
        return val_acc

#===================MOMEN(E)T CODE=================#

class Momentum_Transform(nn.Module):
    """
    this class preform as
    """
    def __init__(self,moment_order=2,v_normals=False,k=20,lift_func = NO_LIFT):
        super().__init__()
        self.k = k
        self.moment_order = moment_order
        self.v_normals=v_normals
        self.channels_in = 12 if moment_order ==2 else 22 #after concat with the KNN
        if v_normals:
            self.channels_in += 3
        if not lift_func['keep_dims']:
            self.channels_in = self.channels_in*2 - 3
        self.lift_func = lift_func

        self.input_transform = Tnet(k=3).to(device)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn6 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Conv2d(self.channels_in, 64, kernel_size=1) #, bias=False)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 64, 1)
        self.conv5 = nn.Conv1d(64, 128, 1)
        self.conv6 = nn.Conv1d(128, 1024, 1)


    def forward(self, input):
        #when we use vertex normals
        if self.v_normals:
            xb, norms = torch.split(input, 3, dim=1)
            matrix3x3 = self.input_transform(xb)
            xb = torch.bmm(xb.transpose(1, 2), matrix3x3).transpose(1, 2)
            norms = torch.bmm(norms.transpose(1, 2), matrix3x3)
        else:
            matrix3x3 = self.input_transform(input)
            xb = torch.bmm(input.transpose(1, 2), matrix3x3).transpose(1, 2)

        knn_idx = knn(xb, self.k)  # (btz x n x k)
        xb_knn = get_graph_feature(xb, k=20, idx=knn_idx)  # (btz x n x k x 3)


        #channels size: 9 for 2nd moment and 19 for 3rd moment
        xb_moment = concat_moment(xb.transpose(1,2), moment=self.moment_order) #btz,n, channels

        # ch + 3 if v_normals (total 12 for 2nd and 22 for 3rd)
        if self.v_normals:
            xb_moment = torch.cat((xb_moment, norms), dim=2)

        xb_moment = lift_with_fuc(xb_moment.transpose(1,2), self.lift_func).transpose(1,2)
        xb_moment = xb_moment.repeat(self.k,1,1,1).transpose(0,1).transpose(1,2).float()#.to(device) # (btz,n,k,ch)
        xb = torch.cat((xb_moment, xb_knn),dim=3).transpose(1,3) #(btz,n,k,12/22)
        xb = F.relu(self.bn1(self.conv1(xb))) # should be (btz,n,k,64)

        # todo: check if need nn.MaxPool1d or adaptive_max_pool1d instead
        #xb = nn.MaxPool1d(xb.size(-1))(xb) #should be (btz,n,64)
        xb = xb.max(dim=2, keepdim=False)[0] #(btz x 64 x n)

        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        xb = F.relu(self.bn4(self.conv4(xb)))
        xb = F.relu(self.bn5(self.conv5(xb)))
        xb = self.conv6(xb)

        xb = xb.max(dim=-1, keepdim=False)[0]
        output = nn.Flatten(1)(xb)
        return output #, matrix3x3, matrix64x64

class Momentnet(nn.Module):
    def __init__(self, train_loader, val_loader, classes=40,lr = 1e-4, moment_order=2,v_normals=False,lift_func=NO_LIFT):
        super(Momentnet, self).__init__()
        self.model_name = 'Momenet_' + str(moment_order)
        if v_normals:
            self.model_name +='_normals'
        self.model_name +=lift_func['name']
        self.lr = lr
        self.classes = classes
        self.loss_function = torch.nn.NLLLoss()
        #self.loss_function = nn.CrossEntropyLoss(reduction='sum')
        self.k = 20
        self.best_model = None

        self.train_loader = train_loader
        self.valid_loader = val_loader
        self.Momentum_Transform = Momentum_Transform(moment_order=moment_order,v_normals=v_normals, k=20,lift_func=lift_func).to(device)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        #imput size: btz,3,n
        xb = self.Momentum_Transform(input)
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        output = self.fc3(xb)
        return self.logsoftmax(output)

    def train_all(self, epochs=10, with_val=True):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        best_acc = -1
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            total = correct = 0
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data['data'],data['label']
                # inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                optimizer.zero_grad()
                outputs = self(inputs.transpose(1, 2))  # forward
                predicted = torch.argmax(outputs, 1)

                total += labels.size(0)
                correct += (labels == predicted).sum().item()
                loss = self.loss_function(outputs, labels.long())
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 10 == 9:
                    """summary_writer.add_scalar(f'Train Loss {self.model_name}', running_loss / 10, epoch * len(self.train_loader) + i)
                    summary_writer.add_scalar(f'Train Accuracy {self.model_name}', correct / total,epoch * len(self.train_loader) + i)"""
                    write_summerize(True, self.model_name, correct/total, epoch, i+1, running_loss/10)
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

    def test_all(self,summary_writer=None,cm=False):
        self.eval()
        correct = total = 0
        if cm:
            all_preds = []
            all_labels = []
        with torch.no_grad():
            for i,data in enumerate(self.valid_loader):
                inputs, labels = data['data'],data['label']
                # inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                outputs = self(inputs.transpose(1, 2))  # forward
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if cm:
                    all_preds += list(predicted.numpy())
                    all_labels += list(labels.numpy())
                if summary_writer:
                    summary_writer.add_scalar(f'Test Accuracy {self.model_name}', correct / total,  i)
        val_acc = 100. * correct / total
        write_summerize(False, self.model_name, accuracy=val_acc)
        if cm:
            cm = confusion_matrix(all_labels, all_preds)
            path = os.path.join('data', 'modelnet40_ply_hdf5_2048')
            classes = {i: class_name for i, class_name in
                       enumerate([line.rstrip() for line in open((os.path.join(path, 'shape_names.txt')))])}
            plot_confusion_matrix(cm, list(classes.keys()), normalize=True)
        precision = precision_score(all_labels, all_preds,average='macro')
        recall = recall_score(all_labels, all_preds,average='macro')
        print(f'{self.model_name} F1: {(2 * precision * recall) / (precision + recall)}')
        return val_acc
