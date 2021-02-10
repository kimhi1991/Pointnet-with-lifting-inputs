import numpy as np
import os
import sys
import torch
import trimesh

torch_configs = dict(
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    type=torch.float32,
    float32=torch.float32,
    float64=torch.float64,
    int32=torch.int32,
    int64=torch.int64
)


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