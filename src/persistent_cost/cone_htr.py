
import numpy as np
from scipy.spatial.distance import squareform
from persistent_cost.utils.utils import htr, births_deaths_to_dgm, conematrix
from persistent_cost.cone import kercoker_bars


def cone_pipeline_htr(dX, dY, f, maxdim=1, cone_eps=0.0, tol=1e-11, threshold=3, return_extra=False):
    """
    Compute the total persistence diagram using the cone algorithm with HTR.

    Parameters
    ----------
    dX : np.array
        Distance matrix of the source space in condensed form.
    dY : np.array
        Distance matrix of the target space in condensed form.
    f : np.array
        Function values.
    maxdim : int, optional
        Maximum homology dimension to compute, by default 1
    cone_eps : float, optional
        Epsilon value for the cone construction, by default 0.0
    tol : float, optional
        Tolerance for numerical comparisons, by default 1e-11
    threshold : float, optional
        Maximum distance for Rips complex, by default 3
    return_extra : bool, optional
        Whether to return extra information, by default False
    
    Returns
    -------
    tuple
        (dgm_coker, dgm_ker, dgm_cone, dgm_X, dgm_Y) or with D and missing if return_extra=True
    """
    
    D = conematrix(dX, dY, f, cone_eps)
    
    # Compute persistence using htr with the new signature
    # htr returns (births, deaths, dims)
    births_X, deaths_X, dims_X = htr(distance_matrix=squareform(dX), threshold=threshold, maxdim=maxdim)
    births_Y, deaths_Y, dims_Y = htr(distance_matrix=squareform(dY), threshold=threshold, maxdim=maxdim)
    births_cone, deaths_cone, dims_cone = htr(distance_matrix=D, threshold=threshold, maxdim=maxdim)
    
    dgm_X = births_deaths_to_dgm(births_X, deaths_X, dims_X, maxdim)
    dgm_Y = births_deaths_to_dgm(births_Y, deaths_Y, dims_Y, maxdim)
    dgm_cone = births_deaths_to_dgm(births_cone, deaths_cone, dims_cone, maxdim)

    dgm_coker, dgm_ker, missing = kercoker_bars(
        dgm_cone, dgm_X, dgm_Y, cone_eps, tol)
    
    if return_extra:
        return dgm_coker, dgm_ker, dgm_cone, dgm_X, dgm_Y, D, missing
    return dgm_coker, dgm_ker, dgm_cone, dgm_X, dgm_Y
