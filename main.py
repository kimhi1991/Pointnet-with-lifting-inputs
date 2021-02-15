from models import *
from data_utils import *
import yaml


def transformation(vertex_normals = False):
    if vertex_normals:
        train_transforms = transforms.Compose([
            PointSampler(SAMPLING_POINTS),
            Normalize(),
            RandRotation_z(),
            RandomNoise(),
            add_vertex_normals(),
            ToTensor()
        ])
    else:
        train_transforms = transforms.Compose([
            PointSampler(SAMPLING_POINTS),
            Normalize(),
            RandRotation_z(),
            RandomNoise(),
            ToTensor()
        ])
    return train_transforms


if __name__ == "__main__":
    #parameters and initilizations
    random.seed = 42
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
    vertex_normals = False

    #orgenize the dataset and dataloader
    path = get_path(classes=NUM_CLASSES)
    folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path / dir)]
    classes = {folder: i for i, folder in enumerate(folders)}
    print(f"classes: {classes}")

    train_transforms = transformation(vertex_normals) #if we use the vertex_normals
    train_ds = PointCloudData(path, transform=train_transforms)
    valid_ds = PointCloudData(path, valid=True, folder='test')

    train_loader = DataLoader(dataset=train_ds, batch_size=TRAIN_BATCH, shuffle=True)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=TEST_BATCH)

    #CREATE THE MODELS
    pointnet = PointNet(train_loader, valid_loader,classes=NUM_CLASSES,lr = LR,alpha=alpha,
                        sampled_data=sampled_data).to(device)
    Momenet = Momentnet(train_loader, valid_loader, classes=NUM_CLASSES, lr=LR, alpha=alpha,
                        momentum_order=2,sampled_data=sampled_data).to(device)
    Momenet3 = Momentnet(train_loader, valid_loader, classes=NUM_CLASSES, lr=LR, alpha=alpha,
                        momentum_order=2,sampled_data=sampled_data).to(device)
    #networks = [pointnet, Momenet, Momenet3]
    networks =[Momenet]
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