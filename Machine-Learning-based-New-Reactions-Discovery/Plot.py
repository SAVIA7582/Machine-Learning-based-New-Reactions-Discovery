import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from IPython.core.pylabtools import figsize  # Internal ipython tool for setting figure size


def initialize():

    plt.rcParams['font.size'] = 24  # Set default font size

    sn.set(font_scale=2)

    pd.options.mode.chained_assignment = None  # No warnings about setting value on copy of slice
    pd.set_option('display.max_columns', 60)  # Display up to 60 columns of a dataframe


def draw_train_loss(loss_train):
    """
    Draw train loss plot
    :param loss_train: list of train_loss
    :return: None
    """

    initialize()

    plt.figure(1)
    plt.title('train loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(np.arange(len(loss_train)), loss_train, label='train')
    plt.legend(loc='upper right')
    plt.savefig('loss.png', dpi=600)  # 保存图片
    plt.show()


def draw_prediction(y_predict, y_test):
    """
    draw scatter plot with predict yield and actual yield
    :param y_predict: predict yield
    :param y_test: actual yield
    :return: None
    """

    sn.regplot(x=y_predict, y=y_test, line_kws={"lw": 2, 'ls': '--', 'color': 'black', "alpha": 0.7})
    plt.xlabel('Predicted yeild', color='blue')
    plt.ylabel('Measured yeild', color='blue')
