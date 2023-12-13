# for finding and saving candidate centers
import numpy as np
from skimage import io
import skimage.filters
from skimage import measure
import scipy.stats as st
import pickle

# for checking centers and Gaussian fits
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from scipy.optimize import curve_fit

# function to take a pixel and make a step towards higher values pixels
def step(img_stack, position):
    z, x, y = img_stack.shape
    l = 1 # window to search
    cen_z = int(position[0])
    cen_x = int(position[1])
    cen_y = int(position[2])
    min_z = max(cen_z-l,0)
    max_z = min(cen_z+l+1,z)
    min_x = max(cen_x-l,0)
    max_x = min(cen_x+l+1,x)
    min_y = max(cen_y-l,0)
    max_y = min(cen_y+l+1,y)
    search_space = img_stack[min_z:max_z,min_x:max_x,min_y:max_y]
    delta = np.array(np.unravel_index(np.argmax(search_space), search_space.shape)) - [position[0]-min_z, position[1]-min_x, position[2]-min_y]
    return position + delta

def find_centers(im_full,peak_cutoff=0.0005,walker_cutoff=100):
    center_list = []
    counts_list = []
    peak_val_list = []
    not_tiny_list = []
    tiny_list = []
    points_init_list = []
    points_list = []
    for image in im_full:
        im = image
        # blur image to remove noise
        blurred_im = skimage.filters.gaussian(im)
        mu, sig = st.norm.fit(blurred_im.flatten())
        # even if you can't get an image with exactly 3 components, we can pick out the 2nd and 3rd largest components which are the cells
        blob_image = measure.label(np.average(blurred_im, axis=0) > (mu), background=0)
        counts = []
        for i in np.unique(blob_image):
            counts.append(np.count_nonzero(blob_image == i))
        labels = np.argsort(counts)[-3:-1]
        filter_mask = (blob_image != labels[0]) & (blob_image != labels[1])
        # create another image with slightly less filter
        blurred_im = skimage.filters.gaussian(im,sigma=[1,1,1])
        for blurred_img in blurred_im:
            # apply cell mask from before
            blurred_img[filter_mask] = 0
            # "normalize" by cytoplasm color
            blurred_img -= np.median(blurred_img[blurred_img !=0])
            # all negatives clip to 0
            np.clip(blurred_img, 0, 1, blurred_img)
        # use all KT pixels as starting points
        mu, sig = st.norm.fit(blurred_im.flatten())
        points_init = np.argwhere(blurred_im>(mu+3*sig)).astype(np.float32)
        points = np.argwhere(blurred_im>(mu+3*sig)).astype(np.float32)
        for i in range(len(points)):
            points[i] = step(blurred_im, points[i])
        # take many steps until convergence
        for _ in range(15):
            for i in range(len(points)):
                points[i] = step(blurred_im, points[i])
        points=points.astype(np.int64)
        centers, counts = np.unique(points, axis=0, return_counts=True)
        not_tiny_idx = []
        tiny_idx = []
        peak_vals = [blurred_im[centers[i][0]][centers[i][1]][centers[i][2]] for i in range(len(centers))]
        for idx, peak in enumerate(peak_vals):
            if peak>peak_cutoff and counts[idx]>walker_cutoff:
                not_tiny_idx.append(idx)
            else:
                tiny_idx.append(idx)
        not_tiny_list.append(not_tiny_idx)
        tiny_list.append(tiny_idx)
        center_list.append(centers)
        counts_list.append(counts)
        peak_val_list.append(peak_vals)
        points_init_list.append(points_init)
        points_list.append(points)
    results_dict = {'not_tiny_list': not_tiny_list,
                   'tiny_list': tiny_list,
                   'center_list': center_list,
                   'counts_list': counts_list,
                   'peak_val_list':peak_val_list,
                   'points_init_list':points_init_list,
                   'points_list':points_list}
    return results_dict

def generate_blurred_im(im):
    blurred_im = skimage.filters.gaussian(im)
    mu, sig = st.norm.fit(blurred_im.flatten())
    # even if you can't get an image with exactly 3 components, we can pick out the 2nd and 3rd largest components which are the cells
    blob_image = measure.label(np.average(blurred_im, axis=0) > (mu), background=0)
    counts = []
    for i in np.unique(blob_image):
        counts.append(np.count_nonzero(blob_image == i))
    labels = np.argsort(counts)[-3:-1]
    filter_mask = (blob_image != labels[0]) & (blob_image != labels[1])
    blurred_im = skimage.filters.gaussian(im,sigma=[1,1,1])
    for blurred_img in blurred_im:
        # apply cell mask from before
        blurred_img[filter_mask] = 0
        # "normalize" by cytoplasm color
        blurred_img -= np.median(blurred_img[blurred_img !=0])
        # all negatives clip to 0
        np.clip(blurred_img, 0, 1, blurred_img)
    return blurred_im

def pixel_to_nm(point):
    z, x, y = point
    return [500*z,90*x,90*y]

def calc_included_points(points_init, points, center):
    included_points = points_init[np.sum(points==center, axis=1)==3]
    return included_points.astype(int)

def calc_point_vals(points, im):
    included_vals = [im[point[0]][point[1], point[2]] for point in points]
    return np.array(included_vals)

def coord_lims(points):
    zlim = [np.min(points[:,0]), np.max(points[:,0])]
    xlim = [np.min(points[:,1]), np.max(points[:,1])]
    ylim = [np.min(points[:,2]), np.max(points[:,2])]
    return xlim, ylim, zlim

def gauss_3d_var_center(coords, mu_x, mu_y, mu_z, A0, A1, sig_x, sig_y, sig_z):
    x = coords[:,1]
    y = coords[:,2]
    z = coords[:,0]
    xgauss = np.exp(-(x-mu_x)**2/(2*sig_x**2))
    ygauss = np.exp(-(y-mu_y)**2/(2*sig_y**2))
    zgauss = np.exp(-(z-mu_z)**2/(2*sig_z**2))
    result = A0+A1*1/(sig_x*sig_y*sig_z*(2*np.pi)**3/2)*xgauss*ygauss*zgauss
    return result

def fit_gauss_3d(points_list,points_init_list,not_tiny_list,blurred_im):
    points_init = points_init_list[idx]
    points = points_list[idx]
    centers = center_list[idx]
    blurred_im = blurred_im_list[idx]
    not_tiny_idx = not_tiny_list[idx]
    included_points_list = []
    included_vals_list = []
    for center in centers[not_tiny_idx]:
        included_points = calc_included_points(points_init, points, center)
        included_points = np.append(included_points,center).reshape(len(np.append(included_points,center))//3,3)
        included_points_list.append(included_points)
        included_vals_list.append(calc_point_vals(included_points, blurred_im))
    popt_list_var_center = np.zeros((len(centers[not_tiny_idx]), 8))
    pcov_list_var_center = np.zeros((len(centers[not_tiny_idx]), 8, 8))
    for i in range(len(centers[not_tiny_idx])): 
        center = centers[not_tiny_idx][i]
        xlim, ylim, zlim = coord_lims(included_points_list[i])
        guess = [center[1], center[2], center[0], np.min(included_vals_list[i]), 1.5, 2, 2, 2]
        bounds = ([xlim[0], ylim[0], zlim[0], 0, 0, 0, 0, 0], 
              [xlim[1], ylim[1], zlim[1], np.max(included_vals_list[i]), 
               100, 10, 10, 10])
        popt, pcov = curve_fit(gauss_3d_var_center, included_points_list[i], included_vals_list[i], bounds=bounds, p0=guess)
        popt_list_var_center[i] = popt
        pcov_list_var_center[i] = pcov
    df_var_center = pd.DataFrame(popt_list_var_center)
    df_var_center.columns = ['x_1', 'y_1', 'z_1',
                         'background', 'A1',
                         'sig_x1', 'sig_y1', 'sig_z1']
    df_var_center['int_intensity'] = df_var_center['A1']*df_var_center['sig_x1']*df_var_center['sig_y1']*df_var_center['sig_z1']
    return df_var_center
