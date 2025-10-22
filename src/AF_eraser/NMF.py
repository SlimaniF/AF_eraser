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
from tqdm import tqdm
import numpy as np
from numpy.linalg import pinv
from scipy.optimize import nnls
from scipy.ndimage import gaussian_filter 
from scipy.linalg import decomp, norm
from sklearn.decomposition import NMF

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
    print("shape : ", shape)
    while iter < max_iteration :
        signal, autofluorescence = _extract_resulting_signals(
        decomposed_signal_matrix=matrix_dict['decomposed_signal_matrix'],
        shape=shape
    )
        print("iter : ", iter)
        sigma = sigmas[iter]
        iter +=1

        estimate_coef_dark = _estimate_coef_and_darkcurrent(
            Y=Y,
            target_matrix=matrix_dict['decomposed_signal_matrix']
        )

        estimate_factorized_signal = _estimate_target_matrix(
            Y=Y,
            linear_coef_matrix=estimate_coef_dark['linear_coef_matrix'],
            dark_current_matrix=estimate_coef_dark['dark_current_matrix'],
        )

        estimate_factorized_signal = _apply_gaussian_filter_on_target_matrix(
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
    
    print("shape : ", shape)
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

    init_background = observed_matrix[1,:]
    init_signal = images[0].flatten() - init_background
    init_signal[init_signal < 0] = 0

    init_background = init_background.astype(np.float32)
    init_signal = init_signal.astype(np.float32)

    decomposed_signal_matrix = np.array([init_signal, init_background], dtype=np.float32)

    #Init matrix D : Dark current
    dark_current_matrix : np.ndarray = np.empty(shape=(images_number,1))

    res : matrix_dict = {
    'normalised_observed_matrix' : Y,
    'decomposed_signal_matrix' : decomposed_signal_matrix,
    'linear_coef_matrix' : np.empty(shape=(images_number,2)),
    'dark_current_matrix' : dark_current_matrix
    }

    return res

def _estimate_coef_and_darkcurrent(
    Y : np.ndarray,
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

    X = np.zeros(shape=(images_number,3), dtype=np.float32)
    for line in tqdm(range(images_number), desc="B and D estimate") :
        X[line,:],_ = nnls(A=C, b=Y[line,:])
    
    estimate_coef_matrix, estimate_dark_current_matrix = X[:,:-1], X[:,-1]

    res = {
        'dark_current_matrix' : estimate_dark_current_matrix,
        'linear_coef_matrix' : estimate_coef_matrix,
    }

    return res

def _estimate_target_matrix_brut_nnls(
    Y : np.ndarray,
    linear_coef_matrix : np.ndarray,
    dark_current_matrix : np.ndarray,
) :
    """
    # Step 2 : Estimate factorization matrix (target) holding C constant.
    More straightforwad we transform eq by substracting D on both sides with previously defined Y.
    """
    
    image_number, pixel_number = Y.shape
    corrected_signal = Y-dark_current_matrix[:,None]

    estimated_target_matrix = np.zeros(shape=(2,pixel_number), dtype= np.float32)
    for pixel in tqdm(range(pixel_number), desc="target estimate") :
        estimated_target_matrix[:,pixel], _ = nnls(A=linear_coef_matrix, b= corrected_signal[:,pixel])


    return estimated_target_matrix

def _estimate_target_matrix(
    Y : np.ndarray,
    linear_coef_matrix : np.ndarray,
    dark_current_matrix : np.ndarray,
) :
    image_number, pixel_number = Y.shape
    Y_coor = Y-dark_current_matrix[:,None]
    BtransposedY_coor = linear_coef_matrix.T @ Y_coor
    
    A = linear_coef_matrix.T @ linear_coef_matrix
    A_inv = pinv(A)

    C = A_inv @ BtransposedY_coor # unconstrained solution, we need to enforce non negativity.
    assert C.shape == (2,pixel_number)
    
    positive_threshold = -1e-12
    if (C > positive_threshold).all() : 
        return C
    else : # Then enforce non negativity that yields minimal residuals 

        b1_transposedb1 = A[0,0]
        b2_transposedb2 = A[1,1]

        c1 = (linear_coef_matrix.T[0,:] @ Y_coor ) / b1_transposedb1 # solution for c2 = 0; shaped (1,pixel_number)
        c2 = (linear_coef_matrix.T[1,:] @ Y_coor ) / b2_transposedb2 # solution for c1 = 0
        c1[c1 <= positive_threshold] = 0
        c2[c2 <= positive_threshold] = 0

        c1 = c1.squeeze()
        c2 = c2.squeeze()

        must_enforce_nnegativity_mask = (C <= positive_threshold).any(axis=0)
        matrix_indices = np.indices(C.shape)[1,0,:]

        for pixel_idx in matrix_indices[must_enforce_nnegativity_mask] :
            residual_c1 =cast(float,norm(c1[pixel_idx]*linear_coef_matrix[:,0] - Y_coor[:,pixel_idx])) 
            residual_c2 =cast(float,norm(c2[pixel_idx]*linear_coef_matrix[:,1] - Y_coor[:,pixel_idx])) 
            residual_0 = cast(float,norm(0 - Y_coor[:,pixel_idx]))
            min_residual = np.argmin([residual_0, residual_c1, residual_c2,])

            if min_residual == 0:
                C[:,pixel_idx] = [0,0]
            elif min_residual == 1 :
                C[:,pixel_idx] = [c1[pixel_idx],0]
            elif min_residual == 2 :
                C[:, pixel_idx] =[0,c2[pixel_idx]]
        
        return C



def _apply_gaussian_filter_on_target_matrix(
    target_matrix : np.ndarray,
    gaussian_kernel : int | tuple[int],
    image_shape : tuple[int]
) :
    assert target_matrix.shape[0] == 2, "target matrix should have only two component (signal_AF, signal_true)"

    for image_idx,kernel in zip(range(2), [0.5,gaussian_kernel]) :
        flat_image = target_matrix[image_idx,:]
        smoothed_image = gaussian_filter(
            input= flat_image.reshape(image_shape),
            sigma=kernel
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
    
    error = norm(obj-obj_prev) / (norm(obj_prev) + delta)
    return error

def compute_residuals() :
    pass

def _extract_resulting_signals(
    decomposed_signal_matrix : np.ndarray,
    shape : tuple[int],
    ) :

    assert decomposed_signal_matrix.shape[0] == 2, "Unexpected shape for decomposed_signal : ".format(decomposed_signal_matrix.shape)

    print('decomposed_signal_matrix, shape : ', decomposed_signal_matrix.shape)

    signal = decomposed_signal_matrix[0,:].reshape(shape)
    autofluorescence = decomposed_signal_matrix[1,:].reshape(shape)

    return signal, autofluorescence

