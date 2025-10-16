"""
Non Negative Matrix Factorization implementation
"""

import numpy as np
from scipy.optimize import nnls, lsq_linear

def NMF(
    images : list[np.ndarray],
    exposure_time : int | list[int],
    gaussian_kernel : int | tuple[int]
) :
    pass

def _initialize_matrix(
    images : list[np.ndarray],
    exposure_time : int | list[int]
) :   
    if len(images) < 2 : raise ValueError("Matrix factorization needs at least two images : signal, extra channel")
    pass

def _estimate_error() :
    pass

def _new_iteration() :
    pass

def _launch_optimization(
    observation_matrix : np.ndarray,
    linear_coef_matrix : np.ndarray,
    dark_current_matrix : np.ndarray,
    exposure_time_matrix : np.ndarray,
    target_matrix : np.ndarray

) :

    #Aim: Transform problem A = E(BC+D) to E⁻1A = BC+D
    #                                      Y = XC'
    #               where Y = E⁻¹A and X is B with an extra last column containing d_i and C' is C with an extra line full of ones.

    #     And pass this to nnls or lsq_square that solves Ax=b
    #                                            with b = Y and A = C'   
    
    images_number, pixel_number = observation_matrix.shape

    Y = np.dot(1/exposure_time_matrix,observation_matrix)
    X = np.hstack(
        (linear_coef_matrix, dark_current_matrix[:,0]), 
        dtype=int
        )
    C = np.vstack(
        (target_matrix, np.ones(1,pixel_number))
        )