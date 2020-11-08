# coding: utf-8 
import numpy as np 
import matplotlib.pyplot as plt 

''' CNN 필터 가중치 시각화 함수 ''' 

def filter_show(filters, nx=8, margin=3, scale=10): 
    """ c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py """ 
    FN, C, FH, FW = filters.shape     
    ny = int(np.ceil(FN / nx)) 
    fig = plt.figure() 
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.01, wspace=0.01)
    for i in range(FN): 
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[]) 
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest') 
    plt.show()

