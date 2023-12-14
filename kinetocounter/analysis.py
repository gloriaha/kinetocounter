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
    """Finds brightest neighboring pixel including itself

    Parameters
    ----------
    img_stack : ndarray
        z, x, y order
    position : list
        3D coordinate of pixel (z,x,y)

    Returns
    -------
    position + delta : list
        3D coordinate of brightest neighbor including itself
    """
    z, x, y = img_stack.shape
    l = 1  # window to search
    cen_z = int(position[0])
    cen_x = int(position[1])
    cen_y = int(position[2])
    min_z = max(cen_z - l, 0)
    max_z = min(cen_z + l + 1, z)
    min_x = max(cen_x - l, 0)
    max_x = min(cen_x + l + 1, x)
    min_y = max(cen_y - l, 0)
    max_y = min(cen_y + l + 1, y)

    # define search space of neighboring pixels
    search_space = img_stack[min_z:max_z, min_x:max_x, min_y:max_y]

    # define step direction
    delta = np.array(np.unravel_index(np.argmax(search_space), search_space.shape)) - [
        position[0] - min_z,
        position[1] - min_x,
        position[2] - min_y,
    ]
    return position + delta


def find_centers(im_full, peak_cutoff=0.0005, walker_cutoff=100):
    """Finds candidate kinetochore centers using pixel brightness ascent

    Parameters
    ----------
    im_full : ndarray
        t, z, x, y order
    peak_cutoff : float
        intensity cutoff for candidate center
    walker_cutoff : int
        minimum number of walkers for candidate center

    Returns
    -------
    result_dict : dict
        center finding results
    """
    center_list = []
    counts_list = []
    peak_val_list = []
    not_tiny_list = []
    tiny_list = []
    points_init_list = []
    points_list = []
    for image in im_full:
        # blur image to remove noise
        blurred_im = generate_blurred_im(image)

        # use all KT pixels as starting points
        mu, sig = st.norm.fit(blurred_im.flatten())
        points_init = np.argwhere(blurred_im > (mu + 3 * sig)).astype(np.float32)
        points = np.argwhere(blurred_im > (mu + 3 * sig)).astype(np.float32)
        for i in range(len(points)):
            points[i] = step(blurred_im, points[i])

        # take many steps until convergence
        for _ in range(15):
            for i in range(len(points)):
                points[i] = step(blurred_im, points[i])
        points = points.astype(np.int64)
        centers, counts = np.unique(points, axis=0, return_counts=True)
        not_tiny_idx = []
        tiny_idx = []
        peak_vals = [
            blurred_im[centers[i][0]][centers[i][1]][centers[i][2]]
            for i in range(len(centers))
        ]

        # only keep centers that are bright enough and that have enough walkers that ended up there
        for idx, peak in enumerate(peak_vals):
            if peak > peak_cutoff and counts[idx] > walker_cutoff:
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

    return {
        "not_tiny_list": not_tiny_list,
        "tiny_list": tiny_list,
        "center_list": center_list,
        "counts_list": counts_list,
        "peak_val_list": peak_val_list,
        "points_init_list": points_init_list,
        "points_list": points_list,
    }


def generate_blurred_im(im):
    """Generates blurred version of image

    Parameters
    ----------
    im : ndarray
        z, x, y order

    Returns
    -------
    blurred_im : ndarray
        z, x, y order
    """
    blurred_im = skimage.filters.gaussian(im)
    mu, sig = st.norm.fit(blurred_im.flatten())
    blob_image = measure.label(np.average(blurred_im, axis=0) > (mu), background=0)
    counts = []

    for i in np.unique(blob_image):
        counts.append(np.count_nonzero(blob_image == i))

    # even if you can't get an image with exactly 3 components, we can pick out the 2nd and 3rd largest components which are the cells
    labels = np.argsort(counts)[-3:-1]
    filter_mask = (blob_image != labels[0]) & (blob_image != labels[1])
    blurred_im = skimage.filters.gaussian(im, sigma=[1, 1, 1])

    # apply cell mask from before
    filter_mask = np.broadcast_to(filter_mask[np.newaxis, :, :], blurred_im.shape)
    blurred_im[filter_mask] = 0
    # "normalize" by cytoplasm color
    blurred_im -= np.median(blurred_im[blurred_im != 0])
    # all negatives clip to 0
    np.clip(blurred_im, 0, 1, blurred_im)

    return blurred_im


def pixel_to_nm(point, x_width=90, y_width=90, z_width=500):
    """Converts pixel coordinates to nanometer coordinates

    Parameters
    ----------
    point : list
        z, x, y order

    Returns
    -------
    point_in_nm : list
        z, x, y order
    """
    z, x, y = point
    point_in_nm = [z_width * z, x_width * x, y_width * y]
    return point_in_nm


def cluster_centers(center_list, not_tiny_list, num_clusters):
    """Clusters candidate centers using k-means

    Parameters
    ----------
    center_list : list
        i, z, x, y order
    not_tiny_list : list
        indices of candidate centers
    num_clusters : int
        number of clusters for k-means

    Returns
    -------
    kmeans_list : list
        for each timepoint, KMeans result
    kt_counts : ndarray
        for each timepoint, the number of centers in each cluster
    """
    kmeans_list = []
    kt_counts = np.zeros((len(center_list), num_clusters))
    for i in range(len(center_list)):
        # calculate coordinates in real space
        nm_list = np.array([pixel_to_nm(center) for center in center_list[i]])
        kmeans = KMeans(n_clusters=2, random_state=0).fit(nm_list)

        if len(center_list[i][not_tiny_list[i]]) > 0:
            # perform kmeans
            kmeans_list.append(
                KMeans(n_clusters=num_clusters, random_state=0).fit(
                    nm_list[not_tiny_list[i]]
                )
            )

            # calculate how many centers are in each cluster
            kt_counts[i] = [
                np.sum(kmeans_list[i].labels_ == j) for j in range(num_clusters)
            ]
            print("time index", i)
            print("kinetochore counts: ", kt_counts[i])
        else:
            kmeans_list.append([])
    return kmeans_list, kt_counts


def calc_included_points(points_init, points, center):
    """Finds area of pixels for fitting candidate center

    Parameters
    ----------
    points_init : ndarray
        initial kinetochore threshold pixel coordinates (i, z, x, y)
    points : ndarray
        final pixel coordinates after walking (i, z, x, y)
    center : ndarray
        candidate center coordinate (z, x, y)

    Returns
    -------
    included_points : ndarray
        coordinates of initial pixels that ended up at center (i, z, x, y)
    """
    included_points = points_init[np.sum(points == center, axis=1) == 3].astype(int)
    return included_points


def calc_point_vals(included_points, im):
    """Finds intensity values of pixels given coordinates

    Parameters
    ----------
    included_points : ndarray
        coordinates of initial pixels that ended up at center (i, z, x, y)
    im : ndarray
        image (z, x, y)

    Returns
    -------
    included_vals : ndarray
        intensity values of included points
    """
    included_vals = np.array(
        [im[point[0]][point[1], point[2]] for point in included_points]
    )
    return included_vals


def coord_lims(points):
    """Finds coordinate limits in 3 dimensions given list of pixel coordinates

    Parameters
    ----------
    points : ndarray
        coordinates of pixels (i, z, x, y)

    Returns
    -------
    xlim : list
        minimum and maximum x values of points
    ylim : list
        minimum and maximum y values of points
    zlim : list
        minimum and maximum z values of points
    """
    zlim = [np.min(points[:, 0]), np.max(points[:, 0])]
    xlim = [np.min(points[:, 1]), np.max(points[:, 1])]
    ylim = [np.min(points[:, 2]), np.max(points[:, 2])]
    return xlim, ylim, zlim


def gauss_3d_var_center(coords, mu_x, mu_y, mu_z, A0, A1, sig_x, sig_y, sig_z):
    """Function for 3D Gaussian with variable center coordinate

    Parameters
    ----------
    coords : ndarray
        for each point, coordinates in z, x, y
    mu_x : float
        center coordinate in x
    mu_y : float
        center coordinate in y
    mu_z : float
        center coordinate in z
    A0 : float
        background intensity value
    A1 : float
        peak intensity amplitude value
    sig_x : float
        width of Gaussian in x
    sig_y : float
        width of Gaussian in y
    sig_z : float
        width of Gaussian in z

    Returns
    -------
    result : ndarray
        for each point, intensity values predicted by Gaussian
    """
    x = coords[:, 1]
    y = coords[:, 2]
    z = coords[:, 0]
    xgauss = np.exp(-((x - mu_x) ** 2) / (2 * sig_x**2))
    ygauss = np.exp(-((y - mu_y) ** 2) / (2 * sig_y**2))
    zgauss = np.exp(-((z - mu_z) ** 2) / (2 * sig_z**2))
    result = (
        A0
        + A1
        * 1
        / (sig_x * sig_y * sig_z * (2 * np.pi) ** 3 / 2)
        * xgauss
        * ygauss
        * zgauss
    )
    return result


def fit_gauss_3d(
    idx, center_list, points_list, points_init_list, not_tiny_list, blurred_im_list
):
    """For a given timepoint, fits 3D Gaussian to each center point

    Parameters
    ----------
    idx : int
        timepoint index
    center_list : list
        for every timepoint, all final center coordinates (t, i, z, x, y)
    points_list : list
        for every timepoint, final pixel coordinates after walking (t, i, z, x, y)
    points_init_list : list
        for every timepoint, initial pixel coordinates before walking (t, i, z, x, y)
    not_tiny_list : list
        for every timepoint, candidate center coordinates that pass thresholds (t, i)
    blurred_im_list : list
        blurred images

    Returns
    -------
    gauss_df : Pandas dataframe
        results of Gaussian fitting
    """
    points_init = points_init_list[idx]
    points = points_list[idx]
    centers = center_list[idx]
    blurred_im = blurred_im_list[idx]
    not_tiny_idx = not_tiny_list[idx]
    included_points_list = []
    included_vals_list = []

    for center in centers[not_tiny_idx]:
        # figure out which points to do Gaussian fit on
        included_points = calc_included_points(points_init, points, center)
        included_points = np.append(included_points, center).reshape(
            len(np.append(included_points, center)) // 3, 3
        )
        included_points_list.append(included_points)
        included_vals_list.append(calc_point_vals(included_points, blurred_im))

    popt_list_var_center = np.zeros((len(centers[not_tiny_idx]), 8))
    pcov_list_var_center = np.zeros((len(centers[not_tiny_idx]), 8, 8))

    for i in range(len(centers[not_tiny_idx])):
        # perform 3D Gaussian fit
        center = centers[not_tiny_idx][i]
        xlim, ylim, zlim = coord_lims(included_points_list[i])
        guess = [
            center[1],
            center[2],
            center[0],
            np.min(included_vals_list[i]),
            1.5,
            2,
            2,
            2,
        ]
        bounds = (
            [xlim[0], ylim[0], zlim[0], 0, 0, 0, 0, 0],
            [xlim[1], ylim[1], zlim[1], np.max(included_vals_list[i]), 100, 10, 10, 10],
        )
        popt, pcov = curve_fit(
            gauss_3d_var_center,
            included_points_list[i],
            included_vals_list[i],
            bounds=bounds,
            p0=guess,
        )
        popt_list_var_center[i] = popt
        pcov_list_var_center[i] = pcov

    gauss_df = pd.DataFrame(popt_list_var_center)
    gauss_df.columns = [
        "x_1",
        "y_1",
        "z_1",
        "background",
        "A1",
        "sig_x1",
        "sig_y1",
        "sig_z1",
    ]
    # calculate integrated intensities
    gauss_df["int_intensity"] = (
        gauss_df["A1"] * gauss_df["sig_x1"] * gauss_df["sig_y1"] * gauss_df["sig_z1"]
    )
    return gauss_df
