import numpy as np

def sparse_grasp_flip_reward(
    achieved_goal: dict,
    desired_goal: dict,
    pos_tol: float,
    ori_tol: float
) -> float:
    """
    Returns 1.0 if both position and orientation errors
    are within tolerance; otherwise 0.0.
    """
    # Position error
    ag_p = np.array(achieved_goal["position"])
    dg_p = np.array(desired_goal["position"])
    
    # Ensure these are 1D arrays - flatten any extra dimensions
    ag_p = ag_p.flatten()
    dg_p = dg_p.flatten()
    
    pos_err = np.linalg.norm(ag_p - dg_p)

    # Orientation error (quaternion angular distance)
    ag_q = np.array(achieved_goal["orientation"])
    dg_q = np.array(desired_goal["orientation"])
    
    # Ensure these are 1D arrays - flatten any extra dimensions
    ag_q = ag_q.flatten()
    dg_q = dg_q.flatten()
    
    # Ensure quaternions are normalized
    ag_q = ag_q / np.linalg.norm(ag_q)
    dg_q = dg_q / np.linalg.norm(dg_q)
    
    # Compute dot product using element-wise multiplication and sum
    # This is more robust than np.dot for handling any remaining shape issues
    dot_product = np.abs(np.sum(ag_q * dg_q))
    
    # Clamp to avoid numerical errors
    dot_product = np.clip(dot_product, 0.0, 1.0)
    
    # Angular distance: angle = 2 * arccos(|dot_product|)
    ori_err = 2.0 * np.arccos(dot_product)

    return 1.0 if (pos_err < pos_tol and ori_err < ori_tol) else 0.0

