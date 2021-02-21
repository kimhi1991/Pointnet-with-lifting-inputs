import numpy as np
import os
import sys
import torch
import trimesh
import tqdm
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
import scipy.spatial.distance
import math
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import itertools
import matplotlib.pyplot as plt
from path import Path
from models import *
from typing import Dict, Tuple


"""
    if sampled_data:
        import utils
        train_transforms = transforms.Compose([
            Normalize(),
            RandomNoise(),
        ])
        train_ds = utils.ModelNetDataset(path,train=True,transform=train_transforms)
        valid_ds = utils.ModelNetDataset(path, train=False, transform=train_transforms)

"""


torch_configs = dict(
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    type=torch.float32,
    float32=torch.float32,
    float64=torch.float64,
    int32=torch.int32,
    int64=torch.int64
)


def read_off(file):
    if 'OFF' != file.readline().strip():
        raise ('Not a valid OFF header')
    n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces

def visualize_rotate(data):
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames = []

    def rotate_z(x, y, z, theta):
        w = x + 1j * y
        return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
    fig = go.Figure(data=data,
                    layout=go.Layout(
                        updatemenus=[dict(type='buttons',
                                          showactive=False,
                                          y=1,
                                          x=0.8,
                                          xanchor='left',
                                          yanchor='bottom',
                                          pad=dict(t=45, r=10),
                                          buttons=[dict(label='Play',
                                                        method='animate',
                                                        args=[None, dict(frame=dict(duration=50, redraw=True),
                                                                         transition=dict(duration=0),
                                                                         fromcurrent=True,
                                                                         mode='immediate'
                                                                         )]
                                                        )
                                                   ]
                                          )
                                     ]
                    ),
                    frames=frames
                    )

    return fig

def pcshow(xs, ys, zs):
    data = [go.Scatter3d(x=xs, y=ys, z=zs,
                         mode='markers')]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=2,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()

def present_pc(ds_path, off_path ="bed/train/bed_0001.off" ):
    with open(ds_path /off_path , 'r') as f:
            verts, faces = read_off(f)
            x, y, z = np.array(verts).T
            pointcloud = PointSampler(3000)((verts, faces))
            norm_pointcloud = Normalize()(pointcloud)
            rot_pointcloud = RandRotation_z()(norm_pointcloud)
            noisy_rot_pointcloud = RandomNoise()(rot_pointcloud)
            pcshow(*noisy_rot_pointcloud.T)

def set_torch_configs(device: str, dtype: str = 'float32') -> None:
    """
    Sets torch device and dtype according to user's request.
    :param device:
    :return:
    """
    global torch_configs

    torch_configs['type'] = torch_configs[dtype]

    if device == 'cpu':
        torch_configs['dev'] = torch.device('cpu')

    if device == 'cuda' and not torch.cuda.is_available():
        print('cude is not available, using device: cpu.')

class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t - s) * pt2[i] + (1 - t) * pt3[i]
        return (f(0), f(1), f(2))

    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))

        sampled_faces = (random.choices(faces,
                                        weights=areas,
                                        cum_weights=None,
                                        k=self.output_size))

        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))

        return sampled_points

class Normalize(object):  # to unit sphare
    def __call__(self, pointcloud):
        #return pointcloud

        assert len(pointcloud.shape) == 2
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))
        return norm_pointcloud

class RandRotation_z(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[math.cos(theta), -math.sin(theta), 0],
                               [math.sin(theta), math.cos(theta), 0],
                               [0, 0, 1]])

        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud

class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))

        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud

class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        return torch.from_numpy(pointcloud)

def default_transforms():
    return transforms.Compose([
        Normalize(),
        ToTensor()
    ])






class PointCloudData(Dataset):
    def __init__(self, root_dir, valid=False, folder="train", transform=default_transforms()):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.files = []
        for category in self.classes.keys():
            new_dir = root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    sample['pcd_path'] = new_dir/file
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file):
        verts, faces = read_off(file)
        if self.transforms:
            pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f)
        return {'pointcloud': pointcloud,
                'category': self.classes[category]}

class ModelNetDataset(Dataset):

    def __init__(
            self,
            path: str,
            train: bool=True,
            transform=None,
            **kwargs
    ):
        """ isabelle.eleanore, chloe khan, lilyermak, yael kiselman,
        jessicakes33, oliviaculpo, misslynniemarie
        100 hottest instagram models to follow
        :param path: Path to dataset.
        :param train: Set true for train data, false for test data.
        :param transforms: A tuple with transforms to apply.
        :param kwargs: Args for transform.
        """
        super().__init__()
        if not os.path.exists(path) or not os.path.isdir(path):
            raise FileNotFoundError(path)

        self.mode = 'train' if train else 'test'
        self.transform = transform
        self.params = kwargs

        self.class_dict = dict()
        self.data = []
        for i, dir in enumerate(os.listdir(path)):
            self.class_dict.update({i:dir})

            data_path = os.path.join(path, dir, self.mode)

            for file in os.listdir(data_path):
                points = np.load(os.path.join(data_path, file))
                self.data.append((points, i))

    def labels(self) -> Dict:
        """
        :return: A dictionary with dataset labels and labels' names.
        """
        return self.class_dict

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Returns a labeled sample.
        :param index: Sample index.
        :return: A tuple (sample, label) containing the image and its class label.
        Raises a ValueError if index is out of range.
        """
        if index not in range(len(self.data)):
            raise ValueError

        points, label = self.data[index]

        if self.transform:
            points = self.transform(points, **self.params)

        points = torch_(torch.from_numpy(points))
        label  = torch_(torch.tensor(label)).type(torch.int64)

        return points, label

    def __len__(self):
        """
        :return: Number of samples in this dataset.
        """
        return len(self.data)

def torch_(x):
    """
    Converts torch object to device and dtype.
    :param x: any torch object.
    :return: x in the current available device and dtype.
    """
    return x.to(torch_configs['dev']).type(torch_configs['type'])

class HiddenPrints(object):

    def __init__(self, stdout: bool = True, stderr: bool = True):
        self._out = stdout
        self._err = stderr

    def __enter__(self):
        if self._out:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        if self._err:
            self._original_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._out:
            sys.stdout.close()
            sys.stdout = self._original_stdout
        if self._err:
            sys.stderr.close()
            sys.stderr = self._original_stderr

def sample(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
    """
    Sample num_points from shape's surface
    :param mesh: trimesh object with a shape.
    :param num_points: number of sampling points.
    :return: sampled points as numpy array.
    """
    with HiddenPrints():
        samples, _ = trimesh.sample.sample_surface_even(mesh, num_points)
        if samples.shape[0] < num_points:
            samples, _ = trimesh.sample.sample_surface(mesh, num_points)
    return samples

# function from https://deeplizard.com/learn/video/0LhiS6yu2qQ
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

