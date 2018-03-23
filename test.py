import numpy as np
import tensorflow as tf
import model
import os
import sys
import math
import cv2
from utilise import crop_image
import datetime

# assign GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# get the specified image resolution


def adaptiveEqualizeHist(img, clipLimit=1.5, tileGridSize=(6,6)):
    """
    Contrast Limited Adaptive Histogram Equalization: 

    Image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). 
    Then each of these blocks are histogram equalized as usual.
    So in a small area, histogram would confine to a small region (unless there is noise).
    If noise is there, it will be amplified. To avoid this, contrast limiting is applied. 
    If any histogram bin is above the specified contrast limit (by default 40 in OpenCV),
    those pixels are clipped and distributed uniformly to other bins before applying histogram equalization.
    After equalization, to remove artifacts in tile borders, bilinear interpolation is applied.
    
    Arguments:
    -----------
        img: np.ndarray
            Image to transform; (the grayscale image)
        clipLimit: pixels under this limit are clipped
        tileGridSize: the small region to calculate histogram locally.
    Returns:
    -----------
        clahe: np.ndarray
            The transformed output image
    """
    b, g, r = cv2.split(img)

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    blue = clahe.apply(b)
    green = clahe.apply(g)
    red = clahe.apply(r)

    return cv2.merge((blue, green, red))

def psnr(img1, img2):
    img1 = img1.astype('float32')
    img2 = img2.astype('float32')
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

haze_path = "./Hazy/"
clear_path =  "./GT/"

best_psnr = 0
best_ite = 0
totol_psnr = 0
psnr_list = []

config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True
with tf.Session(config=config_gpu) as sess:
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    CHANNEL_NUM = 3
    stride = 246
    padding = 5
    haze_img = tf.placeholder(tf.float32, [1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL_NUM])

    clear_img = model.EpMultiDenseNet(haze_img)

    print("processing image...")

    # load pre-trained model
    saver = tf.train.Saver()
    saver.restore(sess, "models/247000_iteration.ckpt")
    
    files = sorted(os.listdir(haze_path))
    gt_files = sorted(os.listdir(clear_path))
    totol_psnr = 0
    ave_psnr = 0
    for (file, gt_file) in zip(files, gt_files):
        print file 
        starttime = datetime.datetime.now()
        name = file.split(".")[0]
        haze = cv2.imread(haze_path + file)

        equa_haze = adaptiveEqualizeHist(haze)

        gt = cv2.imread(clear_path + gt_file)

        mheight = haze.shape[0]
        mwidth = haze.shape[1]
        
        mmheight = int(math.ceil(mheight/256.0)*256)
        mmwidth = int(math.ceil(mwidth/256.0)*256)

        I_haze = np.zeros((mmheight, mmwidth, 3))
        I_haze[mmheight-mheight:mmheight, mmwidth-mwidth:mmwidth, :] += equa_haze
        stack = []

        height = I_haze.shape[0]
        width = I_haze.shape[1]
       
        for x in range(height/stride + 1):
            for y in range(width/stride + 1):

                sub_I_haze = crop_image(x, y, width, height, stride, padding, I_haze)

                final_frame = sess.run(clear_img, feed_dict={haze_img:sub_I_haze})
                final_frame = final_frame[0]
                
                num_width = (width/stride + 1)
                if y != (num_width):
                    final_frame = final_frame[padding:(stride+padding), padding:(stride+padding), :]
                elif y == (num_width):
                    if width % stride == 0:
                      pad = stride + padding
                    else:
                      pad = width-(width/stride)*stride + padding
                    final_frame = final_frame[padding:(stride+padding), padding:pad, :]
                stack.append(final_frame)

        h_stack = []
        for x in range(height/stride+1):
            wid_len = width / stride + 1
            tmp = np.hstack((stack[m] for m in range(x*wid_len, (x+1)*wid_len)))
            h_stack.append(tmp)

        result = np.vstack((h_stack[p] for p in range(height/stride+1)))
        result = result[mmheight-mheight:mmheight, mmwidth-mwidth:mmwidth, :]
        
        result = result * 255
        result[result > 255] = 255
        result[result < 0] = 0
        result = result.astype(np.uint8)

        endtime = datetime.datetime.now()
        print("Running Time:", (endtime-starttime).seconds)

        cv2.imwrite("./results/" + str(name) + ".png", result)
           
        tmp_psnr = psnr(result, gt)
        print("psnr:", tmp_psnr)
            
        totol_psnr += tmp_psnr

    ave_psnr = totol_psnr / len(files)
    print("average_psnr:", ave_psnr) 