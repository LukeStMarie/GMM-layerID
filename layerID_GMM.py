"""
Processes optical images of thin flakes to distinguish layer thicknesses.
Usage: 
    Run training on folder of cropped images to produce catalog file.
    Run testing on target image, with catalog file as an argument.
    See examples at end of this file.
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.optimize import curve_fit
import os
import time
import cv2
import ntpath



# Poly function for fitting
def polyfun(data, a, b, c,d,e):
    x = data[0]
    y = data[1]
    return a + (b*x) + (c*x**2) + (d*y) + (e*y**2)   

def preprocess(img_file, save_dir="none"):
    """
    Use a GMM to create background mask, then uses that mask to generate
    a polynomial background surface, which it subtracts from the image

    Parameters
    ----------
    img_file : path to image to be processed
    
    save_dir : directory to save processed file in, or leave blank to skip saving

    """
    tic = time.perf_counter()
    
    # Load image and apply bilateral filter    
    img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)    
    img = img.astype(np.float32)/256
    img_bl = cv2.bilateralFilter(img,5,50,50)
          
    plt.figure()
    plt.imshow(img_bl)
    
    ## Flatten image
    img_array = np.asarray(img_bl)
    dims = img_array.shape
    img_array2D = img_array.reshape(dims[0]*dims[1],dims[2])

    # Put flattened image into GMM to determine background/foreground,
    # which wil be used as a mask
    n_clusters = 2;
    gmm = GaussianMixture(n_components=n_clusters,covariance_type='full',)
    gmm_pred = gmm.fit_predict(img_array2D)
       
    
    # GMM normally assigns each cluster a random color/label.
    # This sets whatever cluster has more pixels as the background cluster
    gmm_count = np.bincount(gmm_pred)
    idx = np.argsort(gmm_count)
    xdi = np.flip(idx)
    vals = np.arange(n_clusters)[xdi]
    gmm_mask = vals[gmm_pred]
    
    
    # This is an alternative method of determining which cluster is background.
    # This reorders the GMM clusters so the cluster corresponding to the
    # greenest pixels, which should the substrate, is cluster 0, and the next
    # greenest, which should be monolayer, is cluster 1, etc.
    
    # gmm_greens = gmm.means_[:,1]
    # idx = np.argsort(gmm_greens)
    # xdi = np.flip(idx)
    # vals = np.arange(n_clusters)[xdi]
    # gmm_mask = vals[gmm_pred]
    
    
    # This re-orders the GMM parameters to match the new cluster ordering.
    # Not actually needed, as this GMM isn't used again.
    
    # gmm.weights_ = np.array(gmm.weights_)[xdi]
    # gmm.means_ = np.array(gmm.means_)[xdi]
    # gmm.covariances_ = np.array(gmm.covariances_)[xdi]
    # gmm.precisions_ = np.array(gmm.precisions_)[xdi]
    # gmm.precisions_cholesky_ = np.array(gmm.precisions_cholesky_)[xdi]
    

    # Unflattens the mask to a 2D image
    img_gmm_mask = gmm_mask.reshape(dims[0],dims[1])
    
    # Show background/foreground mask
    #plt.figure()
    #plt.imshow(img_gmm_mask)
   
    

    # Uses the mask to fit a 2D polynomial surface to the background,
    # then subtracts the surface from the orginal image
    y_dim, x_dim, _ = img_bl.shape
    R = img_bl[:,:,0].flatten()
    G = img_bl[:,:,1].flatten()
    B = img_bl[:,:,2].flatten()
    X_, Y_ = np.meshgrid(np.arange(x_dim),np.arange(y_dim))
    X = X_.flatten()
    Y = Y_.flatten()
    sub_loc = ((img_gmm_mask.flatten())==0).nonzero()[0]
    Rsub = R[sub_loc]
    Gsub = G[sub_loc]
    Bsub = B[sub_loc]
    Xsub = X[sub_loc]
    Ysub = Y[sub_loc]
        
    Rparameters, Rcovariance = curve_fit(polyfun, [Ysub, Xsub], Rsub)
    Gparameters, Gcovariance = curve_fit(polyfun, [Ysub, Xsub], Gsub)
    Bparameters, Bcovariance = curve_fit(polyfun, [Ysub, Xsub], Bsub)
        
    Rfit = polyfun(np.array([Y, X]), *Rparameters)
    Gfit = polyfun(np.array([Y, X]), *Gparameters)
    Bfit = polyfun(np.array([Y, X]), *Bparameters)
    
    img_poly = np.dstack([(R-Rfit+1).reshape(y_dim,x_dim)/2,
                          (G-Gfit+1).reshape(y_dim,x_dim)/2,
                          (B-Bfit+1).reshape(y_dim,x_dim)/2])
    
    # Show subtracted image
    # plt.figure()
    # plt.imshow(img_poly)
    
    
    # Flattens the background-subtracted image
    img_array = np.asarray(img_poly)
    dims = img_array.shape
    img_array2D = img_array.reshape(dims[0]*dims[1],dims[2])   
    
    # Saves subtracted, flattened image if desired, else just returns it
    flake_name = ntpath.basename(img_file)
    flake_name = os.path.splitext(flake_name)[0] +'.npy'
    if(save_dir!="none"):
        path = os.path.join(save_dir, flake_name)
        np.save(path, img_array2D)

    toc = time.perf_counter()
    print("Processed ", flake_name, "in {} seconds.".format(toc-tic))

    return img_array2D


def training(img_dir, n_clusters, out_file):
    """
    Clusters pixels of processed images in "img_dir" and saves the
    clusters to a catalog in "out_dir".

    Parameters
    ----------
    img_dir : str
        Location of directory containing cropped training image files. 
    n_clusters : int
        Number of different layers thicknesses present in training images.
        Include zero layers (bare substrate) in this count.
    out_file : str
        Name and location of where to save the catalog npz file 
        (i.e., "...\\Graphene_on_SiO2_catalog_5layers.npz")
    """

    tic = time.perf_counter()
    img_array2D = np.empty((0,3))

    
    # Loops over each file in the specified directory.
    # Can work with images, or with npy files produced by running preprocess.
    # Concatenates preprocessed files together.
    for filename in os.listdir(img_dir):
        if filename.endswith(".npy"): 
            path = os.path.join(img_dir, filename)
            img_array2D = np.concatenate((img_array2D,np.load(path)))
        if filename.endswith(".jpg"):
            path = os.path.join(img_dir, filename)
            img_array2D = np.concatenate((img_array2D,preprocess(path)))

            

    
    ## Put preprocessed images into GMM
    gmm = GaussianMixture(n_components=n_clusters,covariance_type='full',)
    gmm.fit(img_array2D)
    
    # GMM normally assigns each cluster a random color/label.
    # This reorders the GMM clusters so the cluster corresponding to the
    # greenest pixels, which should the substrate, is cluster 0, and the next
    # greenest, which should be monolayer, is cluster 1, etc
    gmm_greens = gmm.means_[:,1]
    idx = np.argsort(gmm_greens)
    xdi = np.flip(idx)
    gmm.weights_ = np.array(gmm.weights_)[xdi]
    gmm.means_ = np.array(gmm.means_)[xdi]
    gmm.covariances_ = np.array(gmm.covariances_)[xdi]
    gmm.precisions_ = np.array(gmm.precisions_)[xdi]
    gmm.precisions_cholesky_ = np.array(gmm.precisions_cholesky_)[xdi]
    
    # Puts images through re-ordered GMM
    gmm_pred1 = gmm.predict(img_array2D)
    

    Rsub = img_array2D[:,0].flatten()
    Gsub = img_array2D[:,1].flatten()
    Bsub = img_array2D[:,2].flatten()

    
    ## Scatter plot of data points, colored according to nearest GMM fit.
    ## Uses results from GMM after reordering
    fig4 = plt.figure() 
    ax4 = fig4.add_subplot(111, projection='3d')
    ax4.scatter(Bsub, Gsub, Rsub, c=gmm_pred1, s=0.5, marker=',')
    ax4.set_xlabel('Blue')
    ax4.set_ylabel('Green')
    ax4.set_zlabel('Red')
    
    toc = time.perf_counter()
    print("{} seconds to run.".format(toc-tic))
    
    

    
    ## Export and Display
 
    in_file_dict = {}
    
    in_file_dict['weights'] = gmm.weights_
    in_file_dict['blue means'] = gmm.means_[:,0]
    in_file_dict['green means'] = gmm.means_[:,1]
    in_file_dict['red means'] = gmm.means_[:,2]
    in_file_dict['covariances'] = gmm.covariances_
    in_file_dict['precisions'] =  gmm.precisions_
    in_file_dict['Cholesky precisions'] = gmm.precisions_cholesky_
    
    
    
    plt.show(block=False)
    # out_file = out_dir + '\\master_catalog.npz'
    keep = input('Save this clustering? (y/n): ')
    if keep!='y':
        print('Clustering rejected. Check parameters and run script again.')
        return
    else:
        # try:
        #     os.mkdir(out_dir)
        #     print("folder '{}' created ".format(out_dir))
        # except FileExistsError:
        #     print("folder {} already exists".format(out_dir))
        try:
            os.remove(out_file)
        except FileNotFoundError:
            pass
        with open(out_file, 'wb') as f:
            np.savez(f, **in_file_dict)
        print(f'Clustering data saved to {out_file}.')
    
    
   # print(in_file_dict)
    
    
    return


def testing(img_dir, n_clusters, master_cat_file):
    """
    Identify thickness of flake "img_file" using "master_cat_file"

    Parameters
    ----------
    img_dir : str
        Location of folder of image files.
    n_clusters : int
        Number of layers to fit up to. Determine based on how many layers
        "master_cat_file" was trained to.
    master_cat_file : str
        Location of master catalog npz file for the same material/substrate as
        sample. (i.e., "...\\Graphene_on_SiO2_master_catalog.npz")
    """

    tic = time.perf_counter()

    

    ## Import master catalog values
    in_file_dict = dict(np.load(master_cat_file))
    
   # print(in_file_dict['weights'])
    
    ## Put averaged GMM params into a GMM
    gmm = GaussianMixture(n_components=n_clusters,covariance_type='full',)
    gmm.weights_ = in_file_dict['weights']
    gmm.covariances_ = in_file_dict['covariances']
    gmm.precisions_ = in_file_dict['precisions']
    gmm.precisions_cholesky_ = in_file_dict['Cholesky precisions']
    ## For the means, need to put them back in correct format
    nvals = len(in_file_dict['blue means'])
    meanstemp = np.zeros((nvals,3))
    for i in range(nvals):
        meanstemp[i,0] = in_file_dict['blue means'][i]
        meanstemp[i,1] = in_file_dict['green means'][i]
        meanstemp[i,2] = in_file_dict['red means'][i]
    gmm.means_ = meanstemp 
    
   
    
    for filename in os.listdir(img_dir):
        if filename.endswith(".jpg"):
            img_file = os.path.join(img_dir, filename)
            ## Original image
            img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
            img_array = np.asarray(img)
            dims = img_array.shape
            
            ## Import and pre-processing
            ## Image import
            img_array2D = preprocess(img_file)
            
            # Apply GMM to image
            gmm_pred = gmm.predict(img_array2D)
            
            ## Set pixels that didn't fit into any cluster to the highest value   
            score = gmm.score_samples(img_array2D)    
            gmm_pred = np.where(score<1,n_clusters,gmm_pred)
            
            # Unflatten image
            img_gmm_pred = gmm_pred.reshape(dims[0],dims[1])
            
            ## Plot layer map and color maps of original, pre-processed, and final images
            plt.figure()
            plt.imshow(img_gmm_pred)    
        
            # Can use this to plot "pixel clouds" of the images
            # R = img_array2D[:,0].flatten()
            # G = img_array2D[:,1].flatten()
            # B = img_array2D[:,2].flatten()            
            # ## Scatter plot of data points, of pre-processed image.
            # fig1 = plt.figure() 
            # ax1 = fig1.add_subplot(111, projection='3d')
            # ax1.scatter(B, G, R, s=0.5, c=img_array2D, marker=',')
            # ax1.set_xlabel('Blue')
            # ax1.set_ylabel('Green')
            # ax1.set_zlabel('Red')
            # ## Scatter plot of data points, after clustering
            # fig2 = plt.figure() 
            # ax2 = fig2.add_subplot(111, projection='3d')
            # ax2.scatter(B, G, R, c=gmm_pred, s=0.5, marker=',')
            # ax2.set_xlabel('Blue')
            # ax2.set_ylabel('Green')
            # ax2.set_zlabel('Red')
    

    
    toc = time.perf_counter()
    print("{} seconds to run.".format(toc-tic))
    
    
            
    return


""" Example usage """

# This runs training on images in a folder, creating a catalog that can
# discern between 4 different layer thicknesses (including bare substrate)

# args1 = {'img_dir': "CropImageSetB",
#         'n_clusters': 4,
#         'out_file': 'CropImageSetC/master_catalog_4.npz'}

# training(**args1)



# This uses the catalog created above to identify layers in an image.

args2 = {'img_dir': 'CropImageSetB',
        'n_clusters': 5,
        'master_cat_file': 'CropImageSetB/master_catalog_5.npz'}

testing(**args2)
