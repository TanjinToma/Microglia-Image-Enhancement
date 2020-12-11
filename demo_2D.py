from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from VBET_boosting import LDE_2D
from cane import cane_2d


if __name__=="__main__": 
    dir='./data/' 
    # read image from the directory
    im=io.imread(dir+'img.tif') 
   
    # normalize image
    im=im.astype(np.double)
    im=im/np.max(im[:])
    
    # perform VBET enhancement (2D)
    VE=LDE_2D(im*255)
    VE=VE.astype(np.double)
    VE=VE/255
    boosted_im=np.maximum(VE,im)
    smooth_degree=0.001 # hyperparameter in the optimization problem
    enhanced_im= cane_2d(boosted_im, smooth_degree)
    
    # Display enhanced and input image 
    fig = plt.figure()
    fig.add_subplot(121)
    plt.title("Input image", fontsize=10)
    plt.imshow(im, cmap='gray')
    plt.axis('off')
    fig.add_subplot(122)
    plt.title("VBET enhanced image", fontsize=10)
    plt.imshow(enhanced_im, cmap = 'gray')
    plt.axis('off')
    
