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

from typing import TypedDict, cast
import numpy as np
from scipy.optimize import nnls
from scipy.ndimage import gaussian_filter 
from scipy.linalg import norm
from sklearn.decomposition import NMF

#TODO : change list comprehension to pre-allocate + for loop. Should not impact performance since nnls is already a C function.

class matrix_dict(TypedDict) :
    decomposed_signal_matrix : np.ndarray
    linear_coef_matrix : np.ndarray
    dark_current_matrix : np.ndarray
    normalised_observed_matrix : np.ndarray

class residuals_dict(TypedDict) :
    linear_coef_matrix : list
    dark_current_matrix : list
    decomposed_signal_matrix : list


def remove_autofluorescence_NMF(
    images : list[np.ndarray],
    exposure_time : int | list[int],
    gaussian_kernel : int | tuple[int],
    max_iteration : int
) -> tuple[np.ndarray, np.ndarray, residuals_dict]:
    """
    Remove autofluorescent signal from an acquisition performed in different colors. The Non negative matrix factorization described in *Woolfe and al.2011*
    is implemented with linearly decreasing gaussian kernel.

    # Parameters 
        images : list of equally shaped images from same field of view with different acquisition wavelength.  
        exposure_time : int or list of int with exposure time corresponding to images acquisitions.  
        gaussian_kernel : Initial value for gaussian kernel.
        max_iteration : maximum number of iteration before stopping algorithm, will stop earlier if convergence is reached beforehand.

    # Returns
        signal : True signal from images.
        autofluorescence : signal coming from autofluorescence noise.
    """

    #Images integrity check
    if isinstance(images, list) :
        if len(images) < 2 : raise ValueError("Matrix factorization needs at least two images : signal and extra channel")
        if not all([isinstance(im,np.ndarray) for im in images]) : raise TypeError("All list element must be ndarrays.")
        
        #shape check
        if all(images[0].shape == im.shape for im in images) :
            shape = images[0].shape
        else :
            raise ValueError

    else :
        raise TypeError("Wrong type for images. Expected list or ndarray got {}".format(type(images)))
    
    if isinstance(exposure_time, list) :
        if not all([isinstance(im,int) for im in images]) : raise TypeError("All list element must be ints.")
    elif isinstance(exposure_time, int) :
        exposure_time = [exposure_time]*len(images)
    else :
        raise TypeError("Wrong type for exposure time. Expected int or list got {}".format(type(exposure_time)))
    
    

    matrix_dict = _initialize_matrix(
        images,
        exposure_time
    )

    iter = 0

    Y = matrix_dict['normalised_observed_matrix']

    sigmas = np.linspace(gaussian_kernel, 0, num=max_iteration)
    residuals : residuals_dict = {
        'decomposed_signal_matrix' : [],
        'dark_current_matrix' : [],
        'linear_coef_matrix' : []
    }
    while iter < max_iteration :
        sigma = sigmas[iter]
        iter +=1

        estimate_coef_dark = _estimate_coef_and_darkcurrent(
            Y=Y,
            linear_coef_matrix= matrix_dict['linear_coef_matrix'],
            dark_current_matrix= matrix_dict['dark_current_matrix'],
            target_matrix=matrix_dict['decomposed_signal_matrix']
        )

        estimate_factorized_signal = _estimate_target_matrix(
            Y=Y,
            linear_coef_matrix=estimate_coef_dark['linear_coef_matrix'],
            dark_current_matrix=estimate_coef_dark['dark_current_matrix'],
        )

        _apply_gaussian_filter_on_target_matrix(
            target_matrix=estimate_factorized_signal,
            gaussian_kernel=sigma,
            image_shape=shape,
        )

        new_matrix_dict = matrix_dict.copy()
        new_matrix_dict['dark_current_matrix'] = estimate_coef_dark['dark_current_matrix']
        new_matrix_dict['linear_coef_matrix'] = estimate_coef_dark['linear_coef_matrix']
        new_matrix_dict['decomposed_signal_matrix'] = estimate_factorized_signal

        has_converged, residuals = _acess_convergence(
            previous_matrix_dict= matrix_dict,
            current_matrix_dict= new_matrix_dict,
            matrix_residuals=residuals
        )

        matrix_dict.update(new_matrix_dict)

        if has_converged : break
    
    signal, autofluorescence = _extract_resulting_signals(
        decomposed_signal_matrix=matrix_dict['decomposed_signal_matrix'],
        shape=shape
    )

    return signal, autofluorescence, residuals


def _initialize_matrix(
    images : list[np.ndarray],
    exposure_time : int | list[int]
)-> matrix_dict :

    # Def A matrix : observed signal matrix
    flatten_images = [im.flatten() for im in images]
    observed_matrix = np.array(flatten_images, dtype=np.float32)

    images_number, pixel_number =  observed_matrix.shape

    # Def E matrix : Exposure time matrix
    exposure_matrix = np.diag(exposure_time)

    Y = np.dot(
        np.linalg.inv(exposure_matrix),
        observed_matrix,
    )

    model = NMF(n_components=cast(str,2), init='nndsvda', max_iter=50)
    decomposed_signal_matrix = model.fit_transform(Y)   # shape (m,2) ~ B
    linear_coef_matrix = model.components_        # shape (2,n) ~ C

    #Init matrix D : Dark current
    dark_current_matrix : np.ndarray = np.ones(shape=(images_number,1))

    res : matrix_dict = {
    'normalised_observed_matrix' : Y,
    'decomposed_signal_matrix' : decomposed_signal_matrix,
    'linear_coef_matrix' : linear_coef_matrix,
    'dark_current_matrix' : dark_current_matrix
    }

    return res

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
        (target_matrix, np.ones(shape=(1,pixel_number), dtype=np.float32)),
        dtype=np.float32).T
    
    
    nnls_estimate = [
        nnls(A=C, b=Y[line,:])
        for line in range(0, images_number)]
    
    x,residuals = zip(*nnls_estimate)

    X = np.array(x, dtype=np.float32)

    estimate_coef_matrix, estimate_dark_current_matrix = X[:,:-1], X[:,-1]

    res = {
        'dark_current_matrix' : estimate_dark_current_matrix,
        'linear_coef_matrix' : estimate_coef_matrix,
    }

    return res

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
    corrected_signal = Y-np.repeat(dark_current_matrix, Y.shape[1],axis=1)

    nnls_estimate = [
        nnls(A=linear_coef_matrix.T, b= corrected_signal[:,pixel])
        for pixel in range(pixel_number)]
    
    x,residuals = zip(*nnls_estimate)
    estimated_target_matrix = np.array(
        x, dtype=np.float32
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

def _acess_convergence(
    previous_matrix_dict : matrix_dict,
    current_matrix_dict : matrix_dict,
    matrix_residuals : residuals_dict
    ) :
    
    for key in matrix_residuals.keys() :
        new_residual = _compute_error(
            current_matrix_dict[key],
            previous_matrix_dict[key]
        )
        matrix_residuals[key].append(new_residual)
    
    return False, matrix_residuals

def _compute_error(
    obj : np.ndarray,
    obj_prev : np.ndarray,
    delta = 1e-12
) :
    
    error = norm(obj-obj_prev) / norm(obj_prev) + delta
    return error

def _extract_resulting_signals(
    decomposed_signal_matrix : np.ndarray,
    shape : tuple[int],
    ) :

    assert decomposed_signal_matrix.shape[0] == 0, "Unexpected shape for decomposed_signal : ".format(decomposed_signal_matrix.shape)

    signal = decomposed_signal_matrix[0,:].reshape(shape)
    autofluorescence = decomposed_signal_matrix[1,:].reshape(shape)

    return signal, autofluorescence

