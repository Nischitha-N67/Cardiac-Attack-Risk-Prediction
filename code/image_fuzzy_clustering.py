import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
from numpy.random import randint, random
import scipy.stats
import math
import cv2
import imageio

from sklearn import cluster
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt
from scipy import ndimage
import os
from flask import url_for, current_app

# %matplotlib inline

# input fllename >> output 3d array
def read_img(filename, size):
    img_3d = imageio.imread(filename)
    # Downsample the image
    small = cv2.resize(img_3d, (0, 0), fx = size[0], fy = size[1])
    # Blurring effect to denoise
    blur = cv2.blur(small, (4, 4))
    return small, blur


# input 3d array >> output 2d array
def flatten_img(img_3d):
    x, y, z = img_3d.shape
    img_2d = img_3d.reshape(x*y, z)
    img_2d = np.array(img_2d, dtype = float)
    return img_2d


# input 2d array >> output 3d array
def recover_img(img_2d, X, Y, Z, ):
    img_2d = (img_2d * 255).astype(np.uint8)
    recover_img = img_2d.reshape(X, Y, Z)
    return recover_img


# input 2d array >> output estimated means, stds, pis
def initialization(img, k):
    try:
        # Try kmeans multiple times if needed
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                means, labels = kmeans2(img, k, minit='points')  # Use 'points' initialization
                if len(set(labels)) == k:  # Check if we have k clusters
                    break
            except Exception as e:
                print(f"K-means attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_attempts - 1:
                    raise
        
        means = np.array(means)
        
        # Calculate covariance matrices with regularization
        cov = []
        for i in range(k):
            cluster_points = img[labels == i]
            if len(cluster_points) > 1:
                cluster_cov = np.cov(cluster_points.T)
            else:
                # If cluster is empty or has one point, use identity matrix
                cluster_cov = np.eye(img.shape[1]) * 1e-3
            
            # Add regularization
            cluster_cov += np.eye(cluster_cov.shape[0]) * 1e-3
            cov.append(cluster_cov)
        
        cov = np.array(cov)
        
        # Calculate mixing coefficients (pis)
        cluster_sizes = np.array([np.sum(labels == i) for i in range(k)])
        # Add small constant to prevent zero probabilities
        cluster_sizes = cluster_sizes + 1e-10
        pis = cluster_sizes / np.sum(cluster_sizes)
        
        return means, cov, pis
        
    except Exception as e:
        print(f"Initialization failed: {str(e)}")
        raise


# E-Step: Update Parameters
# update the conditional pdf - prob that pixel i given class j
def update_responsibility(img, means, cov, pis, k):
    try:
        # Calculate responsibilities with error handling
        responsibilities = np.zeros((len(img), k))
        
        for j in range(k):
            try:
                pdf = scipy.stats.multivariate_normal.pdf(
                    img, 
                    mean=means[j], 
                    cov=cov[j] + np.eye(cov[j].shape[0]) * 1e-6,
                    allow_singular=True
                )
                responsibilities[:, j] = pis[j] * pdf
            except Exception as e:
                print(f"Error calculating responsibility for cluster {j}: {str(e)}")
                responsibilities[:, j] = 1e-10  # Small non-zero value
        
        # Normalize responsibilities
        row_sums = responsibilities.sum(axis=1)
        row_sums[row_sums == 0] = 1e-10  # Prevent division by zero
        responsibilities = responsibilities / row_sums[:, np.newaxis]
        
        return responsibilities
        
    except Exception as e:
        print(f"Error in update_responsibility: {str(e)}")
        raise


# update pi for each class of Gaussian model
def update_pis(responsibilities):
    pis = np.sum(responsibilities, axis = 0) / responsibilities.shape[0]
    return pis

# update means for each class of Gaussian model
def update_means(img, responsibilities):
    try:
        means = []
        class_n = responsibilities.shape[1]
        
        for j in range(class_n):
            weight = responsibilities[:, j]
            weight_sum = np.sum(weight)
            
            if weight_sum > 0:
                means_j = np.average(img, weights=weight, axis=0)
            else:
                # If cluster is empty, initialize with random point
                means_j = img[np.random.randint(len(img))]
            
            means.append(means_j)
            
        return np.array(means)
        
    except Exception as e:
        print(f"Error in update_means: {str(e)}")
        raise

# update covariance matrix for each class of Gaussian model
def update_covariance(img, responsibilities, means):
    try:
        cov = []
        class_n = responsibilities.shape[1]
        
        for j in range(class_n):
            weight = responsibilities[:, j]
            weight_sum = np.sum(weight)
            
            if weight_sum > 0:
                # Calculate weighted covariance
                diff = img - means[j]
                weighted_diff = diff * np.sqrt(weight[:, np.newaxis])
                cov_j = np.dot(weighted_diff.T, weighted_diff) / weight_sum
            else:
                # If cluster is empty, use identity matrix
                cov_j = np.eye(img.shape[1]) * 1e-3
            
            # Add regularization
            cov_j += np.eye(cov_j.shape[0]) * 1e-3
            cov.append(cov_j)
            
        return np.array(cov)
        
    except Exception as e:
        print(f"Error in update_covariance: {str(e)}")
        raise


# M-step: choose a label that maximise the likelihood
def update_labels(responsibilities):
    labels = np.argmax(responsibilities, axis = 1)
    return labels


def update_loglikelihood(img, means, cov, pis, k):
    try:
        pdf = np.array([pis[j] * scipy.stats.multivariate_normal.pdf(img, 
                                                                    mean=means[j], 
                                                                    cov=cov[j],
                                                                    allow_singular=True) for j in range(k)])
        log_ll = np.log(np.sum(pdf, axis=0))
        log_ll_sum = np.sum(log_ll)
        return log_ll_sum
    except Exception as e:
        print(f"Error in log likelihood calculation: {str(e)}")
        # Return a very small number instead of failing
        return -np.inf


def EM_cluster(img, k, error=10e-4, iter_n=9999):
    try:
        #  init setting
        cnt = 0
        likelihood_arr = []
        means_arr = []
        means, cov, pis = initialization(img, k)
        
        # Add small value to diagonal of initial covariance matrices
        for i in range(len(cov)):
            cov[i] = cov[i] + np.eye(cov[i].shape[0]) * 1e-6
        
        likelihood = 0
        new_likelihood = 2
        means_arr.append(means)
        responsibilities = update_responsibility(img, means, cov, pis, k)
        
        while (abs(likelihood - new_likelihood) > error) and (cnt != iter_n):
            start_dt = datetime.datetime.now()
            cnt += 1
            likelihood = new_likelihood
            
            try:
                # M-Step
                labels = update_labels(responsibilities)
                # E-step
                responsibilities = update_responsibility(img, means, cov, pis, k)
                means = update_means(img, responsibilities)
                cov = update_covariance(img, responsibilities, means)
                pis = update_pis(responsibilities)
                new_likelihood = update_loglikelihood(img, means, cov, pis, k)
                likelihood_arr.append(new_likelihood)
                
                end_dt = datetime.datetime.now()
                diff = relativedelta(end_dt, start_dt)
                print("iter: %s, time interval: %s:%s:%s:%s" % 
                      (cnt, diff.hours, diff.minutes, diff.seconds, diff.microseconds))
                print("log-likelihood = {}".format(new_likelihood))
                
                # Store means stat
                means_arr.append(means)
                
            except Exception as e:
                print(f"Error in iteration {cnt}: {str(e)}")
                break
                
        likelihood_arr = np.array(likelihood_arr)
        print('Converge at iteration {}'.format(cnt + 1))
        return labels, means, cov, pis, likelihood_arr, means_arr
        
    except Exception as e:
        print(f"Error in EM clustering: {str(e)}")
        raise

def get_pdf(y, means, cov, pis, k):
    pdf_arr = np.array([pis[j] * scipy.stats.multivariate_normal.pdf(y, mean=means[j], cov=cov[j]) for j in range(k)])
    pdf = np.sum(pdf_arr)
    return pdf

def plot_cluster_img(filename, k):
    # Ensure the static/images directory exists
    static_img_dir = os.path.join(current_app.root_path, 'static', 'images')
    if not os.path.exists(static_img_dir):
        os.makedirs(static_img_dir)
        print(f"Created directory: {static_img_dir}")

    # Define image paths
    orig_path = os.path.join(static_img_dir, 'orig_image.jpg')
    em_path = os.path.join(static_img_dir, 'em_image.jpg')
    
    print(f"Will save images to:\nOriginal: {orig_path}\nClustered: {em_path}")

    # Clean up old files if they exist
    for path in [orig_path, em_path]:
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"Removed existing file: {path}")
            except Exception as e:
                print(f"Error removing file {path}: {str(e)}")

    print(f"Processing new image: {filename} with k={k}")
    
    # Read and process image
    small_img, orig_img = read_img(filename=filename, size=(0.5, 0.5))
    x, y, z = orig_img.shape
    
    try:
        # Save original image
        cv2.imwrite(orig_path, cv2.cvtColor(small_img, cv2.COLOR_RGB2BGR))
        print(f"Saved original image to: {orig_path}")
        
        # Process image
        img = flatten_img(orig_img)
        labels, means, cov, pis, likelihood_arr, means_arr = EM_cluster(img, int(k), error=0.001, iter_n=10)
        em_img = recover_img(means[labels], X=x, Y=y, Z=z)
        
        # Save clustered image
        success = cv2.imwrite(em_path, cv2.cvtColor(em_img, cv2.COLOR_RGB2BGR))
        if success:
            print(f"Successfully saved clustered image to: {em_path}")
        else:
            print(f"Failed to save clustered image to: {em_path}")
            
    except Exception as ex:
        print(f"Error during image processing: {str(ex)}")
        raise

    # Verify files exist after processing
    for path in [orig_path, em_path]:
        if os.path.exists(path):
            print(f"Verified file exists: {path}")
        else:
            print(f"WARNING: File not found after processing: {path}")