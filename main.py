from models import *
import utils.data_load as dl
import yaml
import os
import torch
from torch.utils.data import Dataset, DataLoader
import random
from utils.plotter import read_summaries
#from torch.utils.tensorboard import SummaryWriter
random.seed = 42


def main():
    #parameters and initilizations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device('cuda')
    print(device)
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
    #v_normals = True
    print("run order:")
    print('pointnet-normals, momenet-normals, momenet3-normals')
    print('pointnet-normals-lift_sin, momenet-normals-lift_sin, momenet3-normals-lift_sin')
    print('pointnet-normals-liftexp, momenet-normals-liftexp, momenet3-normals-liftexp')
    print('pointnet, momenet, momenet3')
    print('pointnet-lift_sin, momenet-lift_sin, momenet3-lift_sin')
    print('pointnet-liftexp, momenet-liftexp, momenet3-liftexp')

    #for v_normals in [False,True]:
    for v_normals in [True,False]:


        #orgenize the dataset and dataloader
        path=os.path.join('data','modelnet40_ply_hdf5_2048')

        train_ds = dl.PointCloudDataSet(path,numOfPoints = SAMPLING_POINTS ,v_normals=v_normals)
        valid_ds = dl.PointCloudDataSet(path,numOfPoints = SAMPLING_POINTS, valid=True,v_normals=v_normals)

        train_loader = DataLoader(dataset=train_ds, batch_size=TRAIN_BATCH, shuffle=True)
        valid_loader = DataLoader(dataset=valid_ds, batch_size=TEST_BATCH)

        #CREATE THE MODELS
        no_fun = lambda x: x
        log_fun = lambda x: -2*torch.log(x)
        sin_fun = lambda x: torch.sin(x)
        exp_fun = lambda x: torch.exp(-0.2*x)
        FUNCTIONS = [{'keep_dims': True, 'function': no_fun,'name': ""},
                     {'keep_dims': False, 'function': sin_fun,'name': "_sin_lift"},
                     {'keep_dims': False, 'function': exp_fun,'name': "_exp_lift"}]

        for lift_func in FUNCTIONS[1:]:
            pointnet = PointNet( train_loader, valid_loader, classes=NUM_CLASSES, lr=LR,alpha=alpha, v_normals=v_normals,lift_func=lift_func).to(device)
            Momenet  = Momentnet(train_loader, valid_loader, classes=NUM_CLASSES, lr=LR,moment_order=2, v_normals=v_normals,lift_func=lift_func).to(device)
            Momenet3 = Momentnet(train_loader, valid_loader, classes=NUM_CLASSES, lr=LR,moment_order=3, v_normals=v_normals,lift_func=lift_func).to(device)

            networks = [pointnet,Momenet, Momenet3]

            """now = datetime.now()
            date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
            summary_writer = SummaryWriter(os.path.join('config', 'results', date_time))"""
            for model in networks:
                print("---TRAINING {0}---".format(model.model_name))
                model.train_all(epochs=EPOCHS, with_val=True)
                """load best model from a file"""
                #model.load_state_dict(torch.load('best_PointNet_model.pth'))
                """load best model from the atribute best_model (use when we train)"""
                model.load_state_dict(model.best_model)
                print("---TESTING {0}---".format(model.model_name))
                val = model.test_all()
                print(f'Test accuracy:{val}%%')
def plot_graphs():
    read_summaries(train=False)
    read_summaries(train=True)
if __name__ == "__main__":
    #main()
    plot_graphs()
