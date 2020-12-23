import cv2
import numpy as np 
import os
import argparse
import torch
import matplotlib.pyplot as plt


def plot_function(f, tx=None, ty=None, title=None, min=-2, max=2, figsize=(6,4)):
    print("ABC")
    x = torch.linspace(min,max)
    fig,ax = plt.subplots(figsize=figsize)
    ax.plot(x,f(x))
    if tx is not None: ax.set_xlabel(tx)
    if ty is not None: ax.set_ylabel(ty)
    if title is not None: ax.set_title(title)