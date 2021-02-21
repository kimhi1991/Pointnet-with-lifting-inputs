import os
import sys
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

TRAIN_FILES = 'train_files.txt'
TEST_FILES = 'test_files.txt'
CLASS_LIST = 'shape_names.txt'

# Download dataset for point cloud classification
"""
import zipfile as ZipAPIs
import torch
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
    www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    zipfile = os.path.basename(www)

    if os.name == 'nt':
        os.system('wget --no-check-certificate %s' % (www))
        with ZipAPIs.ZipFile(zipfile, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)

    else:
        os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))
"""

"""from path import Path

def get_path(classes=40,sampled=False):
    additianl = "-1024" if sampled else ""
    if classes == 40:
        return Path("data/ModelNet40"+additianl)
    return Path("data/ModelNet10"+additianl)
"""
#Augmentations
#TODO: add augmentation also to default
def default_transforms():
    return transforms.Compose([transforms.ToTensor()])


class PointCloudDataSet(Dataset):
    def __init__(self, data_base_path, numOfPoints, valid=False, transform=default_transforms(),v_normals=False):
        self.data_base_path = data_base_path
        self.numOfPoints = numOfPoints
        self.classes = {i:class_name for i,class_name in enumerate(getDataFiles(os.path.join(data_base_path, CLASS_LIST)))}

        if not valid:
            files_names_list = getDataFiles(os.path.join(data_base_path, TRAIN_FILES)) #'data/modelnet40_ply_hdf5_2048/train_files.txt' or 'data/modelnet40_ply_hdf5_2048/test_files.txt')
        else:
            files_names_list = getDataFiles(os.path.join(data_base_path, TEST_FILES))

        self.transforms = transform if not valid else default_transforms()

        self.data_list = []
        self.labels_list = []
        #Load all data
        for fn in range(len(files_names_list)):
            current_data, current_label = loadDataFile(files_names_list[fn],v_normals=v_normals)
            self.data_list.extend([current_data[i, 0:self.numOfPoints, :] for i in range(current_data.shape[0])])
            self.labels_list.extend(np.squeeze(current_label))

    def __len__(self):
        return(len(self.data_list))
    def __getitem__(self, idx):
        data = self.data_list[idx]
        label = self.labels_list[idx]

        #Transforms
        #data = self.transforms(data)


        return {'data': data, 'label': label}





def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data




def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def loadDataFile(h5_filename,v_normals = False):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    data =data[:,:,0:3]
    if v_normals:
        normals = f['normal'][:]
        normals =normals[:,:,0:3]
        data = np.concatenate((data,normals), axis=2)
    label = f['label'][:]
    return (data, label)




