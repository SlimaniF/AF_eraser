import numpy as np
from scipy.stats import mode
from sklearn.linear_model import RANSACRegressor, LinearRegression


def remove_autofluorescence_RANSACfit(
    signal : np.ndarray,
    background : np.ndarray,
    max_trials : int
) :
    if not isinstance(signal, np.ndarray) : raise TypeError(f"Expected numpy ndarray for signal got {signal.shape}")
    if not isinstance(background, np.ndarray) : raise TypeError(f"Expected numpy ndarray for background got {background.shape}")
    if signal.shape != background.shape : raise ValueError(f"signal and background must have identical shapes. {signal.shape} - {background.shape}")

    background = _match_intensities(signal, background)
    ransac_estimator = _fit_backgrounds(signal, background, min_samples=0.25, max_trials=max_trials)
    new_background = ransac_estimator.predict(background.ravel().reshape(-1,1))
    new_background = new_background.reshape(signal.shape)
    new_background = new_background.astype(signal.dtype)

    new_signal = _substract_backgrounds(
        signal,
        background=new_background
    ).astype(signal.dtype)

    return new_signal, ransac_estimator.score

def _match_intensities(
    signal : np.ndarray,
    background : np.ndarray
) :
    background_mark = int(mode(background.flatten()).mode)
    signal_mark = int(mode(signal.flatten()).mode)
    shift = background_mark - signal_mark
    
    res = np.subtract(background, shift, dtype=np.int64)    
    res[res < 0] = 0
    res = res.astype(background.dtype)

    return res

def _fit_backgrounds(
    signal : np.ndarray,
    background : np.ndarray,
    min_samples = 0.25,
    max_trials = 500,
) : 
    
    X = background.reshape(-1,1)
    y = signal.flatten()

    RANSAC_estimator = RANSACRegressor(
    estimator=LinearRegression(),
    # residual_threshold=inlier_threshold,
    # min_samples=0.25,
    max_trials=max_trials
    )

    RANSAC_estimator.fit(X,y)

    return RANSAC_estimator

    

def _substract_backgrounds(
    signal : np.ndarray,
    background : np.ndarray
) :
    mask = signal >= background
    signal[mask] -= background[mask]
    signal[~mask] = 0

    return signal 

