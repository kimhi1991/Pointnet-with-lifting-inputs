import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd



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
    if train:
        path = os.path.join('figurs','train')
        fig_idx = 1
        colors = ['b','r','g','m','c','y','k']
        for i, filename in enumerate(os.listdir(path)):
            color = colors[i]
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
        colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
        for i, filename in enumerate(os.listdir(path)):
            color = colors[i]
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