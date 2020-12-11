import numpy as np
from numpy.fft import fft2, ifft2
from fun_psf2otf import psf2otf

def vessel_response(gxx,gyy,gxy,theta):
    R=gxx*(np.cos(np.deg2rad(theta)))**2 + gyy*(np.sin(np.deg2rad(theta)))**2 + gxy*(np.sin(np.deg2rad(2*theta)))
    return R

def compute_Hessian_of_Gaussian(X,Y,sigma):

    g=(1/(np.sqrt(2*np.pi)*sigma))*np.exp((-1)*(X**2+Y**2)/(2*sigma**2))
    
    gxx= np.multiply((X**2-sigma**2),g)/(sigma**4)
    gyy= np.multiply((Y**2-sigma**2),g)/(sigma**4)    
    gxy= np.multiply(np.multiply(X,Y),g)/(sigma**4)
    return gxx,gyy,gxy

def max_response_local_and_evidence_filters(I,d,theta,psi,sigma_var,sgn):    

    s = round(6*sigma_var);
    x= np.arange(-s,s+1,1)
    y= np.arange(-s,s+1,1)
    X, Y = np.meshgrid(x, y)
    
    gxx,gyy,gxy = compute_Hessian_of_Gaussian(X,Y,sigma_var)
     
    nr,nc=I.shape
    stack_psi=np.zeros([nr,nc,len(psi)])
    stack_theta=np.zeros([nr,nc,len(theta)])

    
    for ii in range(len(theta)):
        #print(ii)
        theta_var=theta[ii] 
        for kk in range(len(psi)):

            psi_var=psi[kk]
            X_d_f = X + d*np.cos(np.deg2rad(theta_var+psi_var))
            Y_d_f = Y + d*np.sin(np.deg2rad(theta_var+psi_var))

            gxx_d_f,gyy_d_f,gxy_d_f = compute_Hessian_of_Gaussian(X_d_f,Y_d_f, sigma_var)
            
            X_d_b = X - d*np.cos(np.deg2rad(theta_var+psi_var))
            Y_d_b = Y - d*np.sin(np.deg2rad(theta_var+psi_var))
        
            gxx_d_b,gyy_d_b,gxy_d_b= compute_Hessian_of_Gaussian(X_d_b,Y_d_b,sigma_var)
        
            R_d=vessel_response(gxx,gyy,gxy,theta_var); 
            R_f=vessel_response(gxx_d_f,gyy_d_f,gxy_d_f,theta_var+psi_var)
            R_b=vessel_response(gxx_d_b,gyy_d_b,gxy_d_b,theta_var+psi_var)
            
            F_R_d=psf2otf(((sgn*sigma_var**1.5)*R_d),I.shape)
            F_R_f=psf2otf(((sgn*sigma_var**1.5)*R_f),I.shape)
            F_R_b=psf2otf(((sgn*sigma_var**1.5)*R_b),I.shape)
            
            I_d=np.real(ifft2(fft2(I) * F_R_d))
            I_f=np.real(ifft2(fft2(I) * F_R_f))
            I_b=np.real(ifft2(fft2(I) * F_R_b))
            #I_d=I_d.astype(np.uint8); I_f=I_f.astype(np.uint8); I_b=I_b.astype(np.uint8);
            I_all = (I_d + 1*I_b + 1*I_f) 
            #I_all[I_all < 0] = 0
            stack_psi[:,:,kk] = I_all;
            
        stack_theta[:,:,ii] = np.amax(stack_psi, axis=2) 
 
    stack_final = np.amax(stack_theta, axis=2);
  
    return stack_final

def LDE_2D(im):   
    I=im
    vessel_type = 'bright' #vessel_type;        #-- 'bright' or 'dark' vessel
    theta = np.arange(-90,91,15) #detector_orientation_in_plane;  #-- orientation space for detectors   
    psi  = np.arange(-30,31,5)  #predictor_orientation_in_plane;    #-- orientation space for predictors
    smin=1;smax=2
    sigma_var=np.arange(smin,smax+1).tolist()              #-- scale space
    k     = 0.2 #offset_factor;            #-- d = k*sigma_var   
    if vessel_type=='dark':
        sgn = 1
    else:
        sgn = -1;
    
    nr,nc=I.shape
    
    stack=np.zeros([nr,nc,len(sigma_var)])
    
    #tt = time.time()
    for i in range(len(sigma_var)):
        #print(i)
        d=k*sigma_var[i]
        stack[:,:,i]=max_response_local_and_evidence_filters(I,d,theta,psi,sigma_var[i],sgn)
    enh_im=np.amax(stack, axis=2) 
    enh_im[enh_im > 255]=255; enh_im[enh_im < 0]=0; 
    enh_im=enh_im.astype(np.uint8)
    return enh_im


def pseudo_3D_LDE(imv):   
    boosted_stack=np.zeros(imv.shape)
    substack_thickness=np.min([10,imv.shape[0]]); # parameter substack thickness 5
    step=np.floor(substack_thickness*0.5) # oberlapping between substacks
       
    for i in range(0,imv.shape[0],int(step)):
        #print(i)
        substack=np.copy(imv[i:np.min([(i+substack_thickness),imv.shape[0]]),:,:]) # update 
        mip_substack=np.max(substack, axis=0)
        mip_ind=substack.argmax(axis=0)
        mip_LDE=LDE_2D(mip_substack*255)
        mip_LDE=(mip_LDE.astype(np.double))/255 # combine
        sub_new=substack
        
        for r in range(imv.shape[1]):
            for c in range(imv.shape[2]):
                if substack[mip_ind[r,c],r,c]!=0 and substack[mip_ind[r,c],r,c] < mip_LDE[r,c]:
                    sub_new[mip_ind[r,c],r,c]=mip_LDE[r,c]
        
        boosted_stack[i:np.min([(i+substack_thickness),imv.shape[0]]),:,:]=np.maximum(boosted_stack[i:np.min([(i+substack_thickness),imv.shape[0]]),:,:],sub_new)
        
    return boosted_stack
    