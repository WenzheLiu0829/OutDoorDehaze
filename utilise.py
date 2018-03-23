import tensorflow as tf
import os
from operator import mul
import numpy as np 

def crop_image(x, y, width, height, stride, padding, I_haze):
    if x == 0 and y == 0:
          x_start = x*stride
          x_end = (x+1)*stride + padding
          y_start = y*stride
          y_end = (y+1)*stride + padding
          sub_I_haze = I_haze[x_start:x_end, y_start:y_end,:]
          sub_I_haze = np.lib.pad(sub_I_haze, ((padding,0),(padding,0),(0,0)), 'constant', constant_values=1)
    elif x == 0 and y != 0 and y != (width/stride):
          x_start = x*stride
          x_end = (x+1)*stride + padding
          y_start = y*stride - padding
          y_end = (y+1)*stride + padding
          sub_I_haze = I_haze[x_start:x_end, y_start:y_end,:]
          sub_I_haze = np.lib.pad(sub_I_haze, ((padding,0),(0,0),(0,0)), 'constant', constant_values=1)
    elif x == 0 and y == (width/stride):
          x_start = x*stride 
          x_end = (x+1)*stride + padding
          y_start = y*stride - padding
          y_end = width
          pad = (y+1)*stride - width + padding
          sub_I_haze = I_haze[x_start:x_end, y_start:y_end,:]
          sub_I_haze = np.lib.pad(sub_I_haze, ((padding,0),(0,pad),(0,0)), 'constant', constant_values=1)
    elif x != 0 and x != (height/stride) and y == 0:
          x_start = x*stride - padding
          x_end = (x+1)*stride + padding
          y_start = y*stride 
          y_end = (y+1)*stride + padding
          sub_I_haze = I_haze[x_start:x_end, y_start:y_end,:]
          sub_I_haze = np.lib.pad(sub_I_haze, ((0,0),(padding,0),(0,0)), 'constant', constant_values=1)
    elif x != 0 and x != (height/stride) and y == (width/stride):
          x_start = x*stride - padding 
          x_end = (x+1)*stride + padding
          y_start = y*stride - padding
          y_end = width
          pad = (y+1)*stride - width + padding
          sub_I_haze = I_haze[x_start:x_end, y_start:y_end,:]
          sub_I_haze = np.lib.pad(sub_I_haze, ((0,0),(0,pad),(0,0)), 'constant', constant_values=1)
    elif x == (height/stride) and y == 0:
          x_start = x*stride - padding
          x_end = height
          y_start = y*stride 
          y_end = (y+1)*stride + padding 
          pad = (x+1)*stride - height + padding
          sub_I_haze = I_haze[x_start:x_end, y_start:y_end,:]
          sub_I_haze = np.lib.pad(sub_I_haze, ((0,pad),(padding,0),(0,0)), 'constant', constant_values=1)
    elif x == (height/stride) and y != 0 and y != (width/stride):
          x_start = x*stride - padding
          x_end = height
          y_start = y*stride - padding
          y_end = (y+1)*stride + padding 
          pad = (x+1)*stride - height + padding
          sub_I_haze = I_haze[x_start:x_end, y_start:y_end,:]
          sub_I_haze = np.lib.pad(sub_I_haze, ((0,pad),(0,0),(0,0)), 'constant', constant_values=1)
    elif x == (height/stride) and y == (width/stride):
          x_start = x*stride - padding
          x_end = height
          y_start = y*stride - padding
          y_end = width
          pad = (y+1)*stride - width + padding
          padx = (x+1)*stride - height + padding
          sub_I_haze = I_haze[x_start:x_end, y_start:y_end,:]
          sub_I_haze = np.lib.pad(sub_I_haze, ((0,padx),(0,pad),(0,0)), 'constant', constant_values=1)
    else:
          x_start = x*stride - padding
          x_end = (x+1)*stride + padding 
          y_start = y*stride - padding
          y_end = (y+1)*stride + padding 
          sub_I_haze = I_haze[x_start:x_end, y_start:y_end,:]
        
    x_shape = stride + 2*padding
    y_shape = stride + 2*padding
    
    sub_I_haze = sub_I_haze.reshape([1, x_shape, y_shape, 3]) / 255.

    return sub_I_haze


