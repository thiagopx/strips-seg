import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from skimage import measure
import os
import sys
from sklearn import mixture
from segmentKM import segment, join, save


if __name__ == '__main__':

    for line in open('imagens_erro/imagens_erro.txt'):
        front_fname, back_fname, _ = line.split()
        print(back_fname)
        front = cv2.imread(front_fname)[:, :, :: -1]
        back = cv2.imread(back_fname)[:, :, :: -1]
        #cv2.imread('bg_sample.jpeg')[:, :, :: -1]    
        #image = join(front, back, bg_sample, True, 300)
        #plt.imshow(bg_sample)
        #plt.show()
        # run in background
        strips, masks = segment(front, back, 50, 5, 3, 4, 2, 0, 4)
        id_ = front_fname.split('/')[-2]
        print(id_, len(strips))
        save(strips, masks, docname=id_, path='results-imagens-erro')
