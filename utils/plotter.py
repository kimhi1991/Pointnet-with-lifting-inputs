import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import scipy.spatial.distance
import math
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import itertools
from path import Path
from typing import Dict, Tuple

#===================DF FUNCT FOR GRAPHS=================#


def write_summerize(train,name, accuracy, epoch=None, batch=None, loss = None):
    if train:
        df = pd.DataFrame({'Epoch': [epoch], 'Batch': [batch], 'loss': [loss],
                           'train accuracy': [accuracy]})
        df.to_csv(os.path.join('figurs','train', name), mode='a', header=False,
                  index=False)  # orgenize
    else:
        df = pd.DataFrame({'valid accuracy': [accuracy]})
        df.to_csv(os.path.join('figurs', 'test', name), mode='a', header=False,
                  index=False)  # orgenize






def read_summaries(train=True):
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k', 'tab:orange', 'tab:brown', 'tab:blue']
    if train:
        path = os.path.join('figurs','train')
        fig_idx = 1
        for i, filename in enumerate(os.listdir(path)):
            color = colors[i%9]
            df = pd.read_csv(os.path.join(path,filename))
            sns.set()
            plt.figure(fig_idx)
            plt.title("Train Accuracy")
            plt.xlabel('time')
            timeline = 1. * df.values[:,0] + (df.values[:,1]/300.)
            loss = df.values[:,2]
            acc = df.values[:,3]
            plt.plot(timeline, acc, color=color, label=filename)
            plt.legend()
            plt.savefig(path + ' Accuracy graph.png')
            # plt.close(fig_idx)

            plt.figure(fig_idx + 1)
            plt.title("Train Loss")
            plt.xlabel('time')
            plt.plot(timeline, loss, color=color, label=filename)
            plt.legend()
            plt.savefig(path + ' Loss graph.png')
    else:
        path = os.path.join('figurs', 'test')
        fig_idx = 3
        for i, filename in enumerate(os.listdir(path)):
            color = colors[i%9]
            df = pd.read_csv(os.path.join(path, filename))
            sns.set()
            plt.figure(fig_idx)
            plt.title("Validation Accuracy")
            plt.xlabel('Epochs')

            acc = df.values
            timeline = np.linspace(0, len(acc)-1,len(acc))
            plt.plot(timeline, acc, color=color, label=filename)
            plt.legend()
            plt.savefig(path + ' Accuracy graph.png')
            # plt.close(fig_idx)

def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap,aspect='auto')
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
    #plt.show()
    path = os.path.join('figurs', 'train')
    date_time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    plt.savefig(path + ' confusion matrix'+date_time+'.png')



