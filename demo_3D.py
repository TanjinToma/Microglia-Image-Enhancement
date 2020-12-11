
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from VBET_boosting import pseudo_3D_LDE
from cane import cane_3d



if __name__=="__main__": 
    
    dir='./data/' 
    # read 3D image stack from the directory
    imt=io.imread(dir+'img_stack.tif')
    
    # normalize input stack
    imt=imt.astype(np.double)
    imt=imt/np.max(imt[:])
    
    # perform VBET enhancement (3D) 
    boosted_stack=pseudo_3D_LDE(imt)
    mip_result=np.max(boosted_stack, axis=0)
    smooth_degree=0.002 # hyperparameter in the optimization problem
    enhanced_stack= cane_3d(boosted_stack, smooth_degree)
    
    
    # Display the maximum intensity projection (MIP) of the enhanced and input image stack
    fig = plt.figure()
    fig.add_subplot(121)
    plt.title("MIP of input image stack", fontsize=10)
    plt.imshow(np.max(imt, axis=0), cmap='gray')
    plt.axis('off')
    fig.add_subplot(122)
    plt.title("MIP of VBET-enhanced image stack", fontsize=10)
    plt.imshow(np.max(enhanced_stack, axis=0), cmap = 'gray')
    plt.axis('off')

    
    

    