import argparse
import meshio
import numpy as np
import os
import shutil
import trimesh

from utils import sample


def get_args() -> argparse.ArgumentParser:
    """
    Parse command-line arguments
    :return: argparser object with user opts.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--source', type=str, required=True, help='path to ModelNet data dir')
    parser.add_argument('--target', type=str, required=True, help='path to ModelNet processes data dir')
    parser.add_argument('--num-points', type=int, default=1024, help='#sampled points for each shape')

    opt = parser.parse_args()

    return opt


def process_dir(source: str, target: str, num_points: int) -> None:
    """
    Process files of a single dir.
     1. read file
     2. sample num_points of the shape/
     3. rotate for later convenient.
    :param source: source dir.
    :param target: target dir.
    :param num_points: number of points to sample.
    """
    for file in os.listdir(source):

        fsource = os.path.join(source, file)
        ftarget = os.path.join(target, file)

        try:
            fmesh = meshio.read(fsource)
        except meshio._exceptions.ReadError:
            continue

        mesh = trimesh.Trimesh(vertices=fmesh.points, faces=fmesh.cells_dict['triangle'])
        points = sample(mesh=mesh, num_points=num_points)
        points = np.transpose(points, (1, 0))  # rotate from (num_points,3) to (3,num_points)

        np.save(ftarget.split('.')[0], points)  # remove the '.off' suffix


if __name__ == '__main__':
    opts = get_args()

    if opts.source == opts.target:
        raise ValueError('source and target dirs must be different')

    spath = os.path.abspath(opts.source)
    tpath = os.path.abspath(opts.target)

    if not os.path.exists(spath):
        raise FileNotFoundError(spath)

    if os.path.exists(tpath):
        shutil.rmtree(tpath, ignore_errors=True)

    os.mkdir(tpath)

    for dir in os.listdir(spath):
        print('processing ' + os.path.join(spath, dir) + ' ... ', end='')

        os.mkdir(os.path.join(tpath, dir))

        source_dir = os.path.join(spath, dir, 'train')
        target_dir = os.path.join(tpath, dir, 'train')
        os.mkdir(target_dir)
        process_dir(source_dir, target_dir, opts.num_points)

        source_dir = os.path.join(spath, dir, 'test')
        target_dir = os.path.join(tpath, dir, 'test')
        os.mkdir(target_dir)
        process_dir(source_dir, target_dir, opts.num_points)

        print('done!')