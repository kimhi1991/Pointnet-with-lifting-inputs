import argparse
import numpy as np
import os
import shutil
import Mesh

def get_args() -> argparse.ArgumentParser:
    """
    Parse command-line arguments
    :return: argparser object with user opts.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='path to ModelNet data dir')
    parser.add_argument('--target', type=str, required=True, help='path to ModelNet processes data dir')
    opt = parser.parse_args()
    return opt


def process_dir(source: str, target: str) -> None:
    """
    Process files of a single dir.
    :param source: source dir.
    :param target: target dir.
    """
    for file in os.listdir(source):

        fsource = os.path.join(source, file)
        ftarget = os.path.join(target, file)
        mesh = Mesh.Mesh(fsource)
        Vertex_Normals = mesh.Vertex_Normals()
        Vertex_Normals = np.transpose(Vertex_Normals, (1, 0))
        np.save(ftarget.split('.')[0], Vertex_Normals)  # remove the '.off' suffix


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
        process_dir(source_dir, target_dir)

        source_dir = os.path.join(spath, dir, 'test')
        target_dir = os.path.join(tpath, dir, 'test')
        os.mkdir(target_dir)
        process_dir(source_dir, target_dir)

        print('done!')