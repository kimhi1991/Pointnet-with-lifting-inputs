from models import *
#from utils.data_utils import * # when working with the off dataset
import utils.provider as utils
import yaml
import os
import torch
from torch.utils.data import Dataset, DataLoader
import random
random.seed = 42

if __name__ == "__main__":
    #parameters and initilizations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        alpha           = config['model']['alpha']
    v_normals = True

    #orgenize the dataset and dataloader
    path = 'data\\modelnet40_ply_hdf5_2048'
    # path=os.path.join('data','ModelNet40-1024')

    train_ds = utils.PointCloudDataSet(path,numOfPoints = SAMPLING_POINTS ,v_normals=v_normals)
    valid_ds = utils.PointCloudDataSet(path,numOfPoints = SAMPLING_POINTS, valid=True,v_normals=v_normals)

    train_loader = DataLoader(dataset=train_ds, batch_size=TRAIN_BATCH, shuffle=True)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=TEST_BATCH)

    #CREATE THE MODELS
    pointnet = PointNet( train_loader, valid_loader, classes=NUM_CLASSES, lr=LR,alpha=alpha, v_normals=v_normals).to(device)
    Momenet  = Momentnet(train_loader, valid_loader, classes=NUM_CLASSES, lr=LR,moment_order=2, v_normals=v_normals).to(device)
    Momenet3 = Momentnet(train_loader, valid_loader, classes=NUM_CLASSES, lr=LR,moment_order=3, v_normals=v_normals).to(device)

    #networks = [pointnet, Momenet, Momenet3]
    networks =[Momenet3]
    for model in networks:
        print(f"---TRAINING {model.model_name}---")
        model.train_all(epochs=EPOCHS, with_val=True)
        """load best model from a file"""
        #model.load_state_dict(torch.load('best_PointNet_model.pth'))
        """load best model from the atribute best_model (use when we train)"""
        model.load_state_dict(model.best_model)
        print(f"---TESTING {model.model_name}---")
        val = model.test_all()
        print('Test accuracy: %d %%' % val)