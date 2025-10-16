"""
Non Negative Matrix Factorization implementation
Based on "Autofluorescence Removal by Non-Negative Matrix Factorization",Woolfe and al. 2011, IEEE.

Notation from paper : 
A = observation_matrix
E = exposure_time_matrix
C = system_output_matrix (factorization solution)
B = linear_coef_matrix
D = dark_current_matrix
"""

from typing import TypedDict
import numpy as np
from scipy.optimize import nnls, lsq_linear
from scipy.ndimage import gaussian_filter

class matrix_dict(TypedDict) :
    observed_matrix : np.ndarray
    exposure_matrix : np.ndarray
    decomposed_signal_matrix : np.ndarray
    linear_coef_matrix : np.ndarray
    dark_current_matrix : np.ndarray

def NMF(
    images : list[np.ndarray],
    exposure_time : int | list[int],
    gaussian_kernel : int | tuple[int],
    max_iteration : int
):
    
    matrix_dict = _initialize_matrix(
        images,
        exposure_time
    )

    iter = 0

    Y = np.dot(
        1/matrix_dict['exposure_matrix'],
        matrix_dict['observed_matrix']
    )
    while iter < max_iteration :
        iter +=1

        # TODO linear_coef_matrix, dark_current_matrix = _estimate_coef_and_darkcurrent(

        #   )

def _initialize_matrix(
    images : list[np.ndarray],
    exposure_time : int | list[int]
)-> matrix_dict :
    if isinstance(images, list) :
        if len(images) < 2 : raise ValueError("Matrix factorization needs at least two images : signal and extra channel")
        if not all([isinstance(im,np.ndarray) for im in images]) : raise TypeError("All list element must be ndarrays.")
        length_list = [im.size for im in images]
        if min(length_list) != max(length_list) : raise ValueError("Passed images must have same shape")
    elif isinstance(images, np.ndarray) :
        images = [images]
    else :
        raise TypeError("Wrong type for images. Expected list or ndarray got {}".format(type(images)))
    
    if isinstance(exposure_time, list) :
        if not all([isinstance(im,int) for im in images]) : raise TypeError("All list element must be ints.")
    elif isinstance(exposure_time, int) :
        exposure_time = [exposure_time]
    else :
        raise TypeError("Wrong type for exposure time. Expected int or list got {}".format(type(exposure_time)))

    # Def A matrix : observed signal matrix
    flatten_images = [im.flatten() for im in images]
    observed_matrix = np.array(flatten_images, dtype=np.float64)

    images_number, pixel_number =  observed_matrix.shape

    # Def E matrix : Exposure time matrix
    exposure_matrix = np.diag(exposure_time)

    #Init matrix C : system output
    decomposed_signal_matrix = np.ones(shape=(2,pixel_number), dtype=np.float64)

    #Init matrix B : linear coefficient
    linear_coef_matrix = np.ones(shape=(images_number,2),dtype=np.float64)

    #Init matrix D : Dark current
    dark_current_matrix : np.ndarray = np.ones(shape=(images_number,pixel_number))

    res : matrix_dict = {
    'observed_matrix' : observed_matrix,
    'exposure_matrix' : exposure_matrix,
    'decomposed_signal_matrix' : decomposed_signal_matrix,
    'linear_coef_matrix' : linear_coef_matrix,
    'dark_current_matrix' : dark_current_matrix
    }

    return res

def _launch_optimization(
    observation_matrix : np.ndarray,
    linear_coef_matrix : np.ndarray,
    dark_current_matrix : np.ndarray,
    exposure_time_matrix : np.ndarray,
    target_matrix : np.ndarray

) :

    pass

def _estimate_coef_and_darkcurrent(
    Y : np.ndarray,
    linear_coef_matrix : np.ndarray,
    dark_current_matrix : np.ndarray,
    target_matrix : np.ndarray
) :
    """
    # STEP 1 : Holding C constant estimate B and D with known A and E where N is necleted (higher is E the more negligible it becomes)
        * Aim: Transform problem A = E(BC+D) to E⁻1A = BC+D
                                             Y = XC'
                      where Y = E⁻¹A and X is B with an extra last column containing d_i and C' is C with an extra line full of ones.

            And pass this to nnls or lsq_square that solves Ax=b
                                                   with b = Y and A = C'
        so returned solution vector x corresponds to X
    """ 
    
    images_number, pixel_number = Y.shape

    C = np.vstack(
        (target_matrix, np.ones(shape=(1,pixel_number), dtype=np.float64)),
        dtype=np.float64)
    
    
    nnls_estimate = [
        nnls(A=C.T, b=Y[line,:])
        for line in range(0, images_number)]
    
    x,residuals = zip(*nnls_estimate)

    X = np.array(x, dtype=np.float64)

    estimate_coef_matrix, estimate_dark_current_matrix = X[:,:-1], X[:,-1]


    return estimate_coef_matrix, estimate_dark_current_matrix

def _estimate_target_matrix(
    Y : np.ndarray,
    linear_coef_matrix : np.ndarray,
    dark_current_matrix : np.ndarray,
) :
    """
    # Step 2 : Estimate factorization matrix (target) holding C constant.
    More straightforwad we transform eq by substracting D on both sides with previously defined Y.
    """
    
    image_number, pixel_number = Y.shape
    corrected_signal = Y-dark_current_matrix

    nnls_estimate = [
        nnls(A=linear_coef_matrix.T, b= corrected_signal[:,pixel])
        for pixel in range(pixel_number)]
    
    x,residuals = zip(*nnls_estimate)
    estimated_target_matrix = np.array(
        x, dtype=np.float64
    )

    return estimated_target_matrix

def _apply_gaussian_filter_on_target_matrix(
    target_matrix : np.ndarray,
    gaussian_kernel : int | tuple[int],
    image_shape : tuple[int]
) :
    assert target_matrix.shape[0] == 2, "target matrix should have only two component (signal_AF, signal_true)"

    for image_idx in range(2) :
        flat_image = target_matrix[image_idx,:]
        smoothed_image = gaussian_filter(
            input= flat_image.reshape(image_shape),
            sigma=gaussian_kernel
        )
        target_matrix[image_idx,:] = smoothed_image.flatten()
    
    return target_matrix

def assess_convergence() :
    pass