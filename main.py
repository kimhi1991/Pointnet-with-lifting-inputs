from models import *
import utils.data_load as dl
from utils.plotter import read_summaries
import yaml
import os
import torch
from torch.utils.data import Dataset, DataLoader
import random

#from torch.utils.tensorboard import SummaryWriter
random.seed = 42
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():
    """
    train all architectures
    for using tensorboard use the following line and send to train_all and valid_all funcs:
    date_time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    summary_writer = SummaryWriter(os.path.join('config', 'results', date_time))
    """
    path = os.path.join('data', 'modelnet40_ply_hdf5_2048')
    #parameters and initilizations
    with open(os.path.join(os.getcwd(), 'config/config.yml'), 'r') as yml_file:
        config = yaml.load(yml_file)
        print('------------ config ------------')
        print(yaml.dump(config))
        SAMPLING_POINTS = config['data']['sampling_points']
        NUM_CLASSES     = config['data']['num_classes']
        TRAIN_BATCH     = config['model']['train_batch']
        TEST_BATCH      = config['model']['test_batch']
        EPOCHS          = config['model']['EPOCHS']
        LR              = config['model']['learning_rate']
        alpha           =  config['model']['alpha']

    for v_normals in [True,False]:
        #orgenize the dataset and dataloader
        train_ds = dl.PointCloudDataSet(path,numOfPoints = SAMPLING_POINTS ,v_normals=v_normals)
        valid_ds = dl.PointCloudDataSet(path,numOfPoints = SAMPLING_POINTS, valid=True,v_normals=v_normals)
        train_loader = DataLoader(dataset=train_ds, batch_size=TRAIN_BATCH, shuffle=True)
        valid_loader = DataLoader(dataset=valid_ds, batch_size=TEST_BATCH)

        FUNCTIONS = get_lifting_dict()

        #CREATE THE MODELS
        for lift_func in FUNCTIONS:
            pointnet = PointNet( train_loader, valid_loader, classes=NUM_CLASSES, lr=LR,alpha=alpha, v_normals=v_normals,lift_func=lift_func).to(device)
            Momenet  = Momentnet(train_loader, valid_loader, classes=NUM_CLASSES, lr=LR,moment_order=2, v_normals=v_normals,lift_func=lift_func).to(device)
            Momenet3 = Momentnet(train_loader, valid_loader, classes=NUM_CLASSES, lr=LR,moment_order=3, v_normals=v_normals,lift_func=lift_func).to(device)
            networks = [pointnet,Momenet, Momenet3]
            for model in networks:
                print("---TRAINING {0}---".format(model.model_name))
                #model_path = os.path.join('weights', f'best_{model.model_name}_model.pth')
                #model.load_state_dict(torch.load(model_path))
                model.train_all(epochs=EPOCHS, with_val=True)

                model.load_state_dict(model.best_model)
                print("---TESTING {0}---".format(model.model_name))
                val = model.test_all()
                print(f'Test accuracy:{val}%%')

def validation():
    """
    if already trained, use only validation and save confusion matrix
    """
    path = os.path.join('data', 'modelnet40_ply_hdf5_2048')
    with open(os.path.join(os.getcwd(), 'config/config.yml'), 'r') as yml_file:
        config = yaml.load(yml_file)
        SAMPLING_POINTS = config['data']['sampling_points']
        NUM_CLASSES     = config['data']['num_classes']
        TRAIN_BATCH     = config['model']['train_batch']
        TEST_BATCH      = config['model']['test_batch']
        LR              = config['model']['learning_rate']
        alpha           =  config['model']['alpha']
    for v_normals in [True,False]:
        train_ds = dl.PointCloudDataSet(path,numOfPoints = SAMPLING_POINTS ,v_normals=v_normals)
        valid_ds = dl.PointCloudDataSet(path,numOfPoints = SAMPLING_POINTS, valid=True,v_normals=v_normals)
        train_loader = DataLoader(dataset=train_ds, batch_size=TRAIN_BATCH, shuffle=True)
        valid_loader = DataLoader(dataset=valid_ds, batch_size=TEST_BATCH)

        FUNCTIONS = get_lifting_dict()
        #CREATE THE MODELS
        for lift_func in FUNCTIONS:
            pointnet = PointNet( train_loader, valid_loader, classes=NUM_CLASSES, lr=LR,alpha=alpha, v_normals=v_normals,lift_func=lift_func).to(device)
            Momenet  = Momentnet(train_loader, valid_loader, classes=NUM_CLASSES, lr=LR,moment_order=2, v_normals=v_normals,lift_func=lift_func).to(device)
            Momenet3 = Momentnet(train_loader, valid_loader, classes=NUM_CLASSES, lr=LR,moment_order=3, v_normals=v_normals,lift_func=lift_func).to(device)

            networks = [pointnet,Momenet, Momenet3]
            for model in networks:
                model_path = os.path.join('weights',f'best_{model.model_name}_model.pth')
                if not os.path.exists(model_path):
                    continue
                model.load_state_dict(torch.load(model_path))
                val = model.test_all(cm=True)
                print(f'model {model.model_name} Test with Acc: {val}')

def plot_graphs():
    read_summaries(train=False)
    read_summaries(train=True)

if __name__ == "__main__":
    """
    WARNING:    please don't run train if you dont want to train the networks
                if you do, cearfully change parameters in train func to have less runs
                the defult setup train alot of networks
    """
    #train()
    validation()
    #plot_graphs()
