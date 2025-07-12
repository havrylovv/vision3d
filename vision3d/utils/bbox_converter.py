"""Utility functions for ordering unstructured 3D bounding boxes and converting them between different formats."""

import numpy as np
import torch
from torch import Tensor
from typing import Union
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Union


def reorder_corners_pca(boxes: Union[Tensor, np.ndarray]) -> torch.Tensor:
    """
    Reorders the corners of a batch of 3D bounding boxes to enforce a consistent
    orientation and ordering using PCA alignment and Hungarian matching.
    Order: 
        - 0: Bottom-left-down
        - 1: Bottom-right-down
        - 2: Bottom-right-up
        - 3: Bottom-left-up
        - 4: Top-left-down
        - 5: Top-right-down
        - 6: Top-right-up
        - 7: Top-left-up    

    Args:
        boxes (torch.Tensor): Tensor of shape (B, 8, 3) representing a batch of B boxes,
            each with 8 corner points in 3D space.

    Returns:
        torch.Tensor: Tensor of shape (B, 8, 3) containing reordered box corners.
    """
    assert boxes.ndim == 3 and boxes.shape[1:] == (8, 3), "Input tensor must have shape (B, 8, 3)"
    
    # Move to CPU and convert to NumPy for PCA operations
    if isinstance(boxes, torch.Tensor):
        boxes_np = boxes.detach().cpu().numpy()
    elif not isinstance(boxes, np.ndarray): 
        boxes_np = boxes
    else: 
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")
        
    B = boxes_np.shape[0]
    reordered_boxes_np = np.zeros_like(boxes_np)

    for i in range(B):
        corners = boxes_np[i]
        center = corners.mean(axis=0)
        centered = corners - center

        # Compute covariance matrix of centered points for PCA
        cov = np.cov(centered.T)

        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Sort eigenvectors by descending eigenvalue magnitude
        eigvecs = eigvecs[:, np.argsort(-eigvals)]

        # Enforce right-handed coordinate system
        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, -1] *= -1

        # Transform corners to PCA-aligned coordinate system
        aligned_corners = centered @ eigvecs

        # Determine bottom and top faces by Z coordinate in PCA space
        z_min, z_max = aligned_corners[:, 2].min(), aligned_corners[:, 2].max()
        z_mid = (z_min + z_max) / 2
        bottom_mask = aligned_corners[:, 2] < z_mid
        top_mask = ~bottom_mask

        bottom_corners = aligned_corners[bottom_mask]
        top_corners = aligned_corners[top_mask]
        bottom_indices = np.where(bottom_mask)[0]
        top_indices = np.where(top_mask)[0]

        # Sort bottom face corners counter-clockwise based on angle around centroid
        bottom_center = bottom_corners.mean(axis=0)
        bottom_relative = bottom_corners[:, :2] - bottom_center[:2]
        bottom_angles = np.arctan2(bottom_relative[:, 1], bottom_relative[:, 0])
        bottom_order = np.argsort(bottom_angles)

        # Sort top face corners counter-clockwise with the same approach
        top_center = top_corners.mean(axis=0)
        top_relative = top_corners[:, :2] - top_center[:2]
        top_angles = np.arctan2(top_relative[:, 1], top_relative[:, 0])
        top_order = np.argsort(top_angles)

        # Combine bottom and top face indices to form final ordering
        reordered_indices = np.concatenate([
            bottom_indices[bottom_order],
            top_indices[top_order]
        ])

        # Verify counter-clockwise orientation of bottom face in original space
        ordered_bottom = corners[reordered_indices[:4]]
        v1 = ordered_bottom[1] - ordered_bottom[0]
        v2 = ordered_bottom[2] - ordered_bottom[1]
        cross = np.cross(v1, v2)

        vertical_axis = eigvecs[:, 2]  # PCA Z-axis in original space
        if np.dot(cross, vertical_axis) < 0:
            # If orientation is clockwise, reverse order of both faces
            reordered_indices[:4] = reordered_indices[:4][::-1]
            reordered_indices[4:] = reordered_indices[4:][::-1]

        reordered_boxes_np[i] = corners[reordered_indices]

    # Convert result back to torch tensor
    reordered_boxes = torch.from_numpy(reordered_boxes_np).to(boxes.device)
    return reordered_boxes

def corners_to_obb(corners: Union[Tensor, np.ndarray]) -> torch.Tensor:
    """
    Convert 8 corners of 3D bbox to parametric representation of Oriented Bounding Boxes. 
    Structure: [center, size, quaternion]
    
    Args:
        corners: (B, 8, 3) array of corner coordinates
        
    Returns:
        obb: (B, 10) array [center_x, center_y, center_z, size_x, size_y, size_z, qx, qy, qz, qw]
    """
    if isinstance(corners, torch.Tensor):
        device = corners.device
        corners = corners.detach().cpu().numpy()
    elif isinstance(corners, np.ndarray):
        device = torch.device('cpu')
    else:
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

    if corners.ndim == 2:
        corners = corners[np.newaxis, ...]  
    
    B = corners.shape[0]
    obb = np.zeros((B, 10))
    
    # Calculate centers
    centers = np.mean(corners, axis=1)  # (B, 3)
    obb[:, 0:3] = centers
    
    # Get edge vectors for each batch
    x_vecs = corners[:, 1] - corners[:, 0]  # (B, 3)
    y_vecs = corners[:, 3] - corners[:, 0]  # (B, 3)
    z_vecs = corners[:, 4] - corners[:, 0]  # (B, 3)
    
    # Calculate dimensions
    widths = np.linalg.norm(x_vecs, axis=1)   # (B,)
    heights = np.linalg.norm(y_vecs, axis=1)  # (B,)
    depths = np.linalg.norm(z_vecs, axis=1)   # (B,)
    
    obb[:, 3] = widths
    obb[:, 4] = heights
    obb[:, 5] = depths
    
    # Calculate rotation matrices
    rotation_matrices = np.zeros((B, 3, 3))
    
    for i in range(B):
        x_unit = x_vecs[i] / widths[i] if widths[i] > 0 else np.array([1, 0, 0])
        y_unit = y_vecs[i] / heights[i] if heights[i] > 0 else np.array([0, 1, 0])
        z_unit = z_vecs[i] / depths[i] if depths[i] > 0 else np.array([0, 0, 1])

        x_unit, y_unit, z_unit = orthonormalize_basis(x_unit, y_unit, z_unit)
        rotation_matrix = np.column_stack([x_unit, y_unit, z_unit])
        
        # Fix determinant sign if needed
        if np.linalg.det(rotation_matrix) < 0:
            rotation_matrix[:, 2] *= -1  # flip z axis
        
        rotation_matrices[i] = rotation_matrix

    
    # Convert rotation matrices to quaternions using scipy
    quaternions = R.from_matrix(rotation_matrices).as_quat()  # (B, 4) [x, y, z, w]
    obb[:, 6:10] = quaternions

    return torch.from_numpy(obb).to(device)

def obb_to_corners(obb: Union[Tensor, np.ndarray]) -> torch.Tensor:
    """
    Convert Oriented Bounding Box to 8 corners representatoin. Ensures specific corner ordering.
    Order:
        - 0: Bottom-left-down   
        - 1: Bottom-right-down
        - 2: Bottom-right-up
        - 3: Bottom-left-up
        - 4: Top-left-down
        - 5: Top-right-down
        - 6: Top-right-up
        - 7: Top-left-up
    
    Args:
        obb: (B, 10) array [center_x, center_y, center_z, size_x, size_y, size_z, qx, qy, qz, qw]
        
    Returns:
        corners: (B, 8, 3) array of corner coordinates
    """
    if isinstance(obb, torch.Tensor):
        device = obb.device
        obb = obb.detach().cpu().numpy()
    elif isinstance(obb, np.ndarray):
        device = torch.device('cpu')
    else:
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")
    
    if obb.ndim == 1:
        obb = obb[np.newaxis, ...] 
    
    B = obb.shape[0]
    
    # Extract parameters
    centers = obb[:, 0:3]      # (B, 3)
    sizes = obb[:, 3:6]        # (B, 3)
    quaternions = obb[:, 6:10] # (B, 4)     #  [qx, qy, qz, qw]
    
    # Convert quaternions to rotation matrices using scipy
    rotation_matrices = R.from_quat(quaternions).as_matrix()  # (B, 3, 3)
    
    # Half dimensions
    half_sizes = sizes / 2  # (B, 3)
    
    # Local corner coordinates (before rotation and translation)
    local_corners = np.array([
        [-1, -1, -1],  # 0: min_x, min_y, min_z
        [+1, -1, -1],  # 1: max_x, min_y, min_z
        [+1, +1, -1],  # 2: max_x, max_y, min_z
        [-1, +1, -1],  # 3: min_x, max_y, min_z
        [-1, -1, +1],  # 4: min_x, min_y, max_z
        [+1, -1, +1],  # 5: max_x, min_y, max_z
        [+1, +1, +1],  # 6: max_x, max_y, max_z
        [-1, +1, +1],  # 7: min_x, max_y, max_z
    ])  # (8, 3)
    
    # Scale by half sizes
    scaled_corners = local_corners[np.newaxis, :, :] * half_sizes[:, np.newaxis, :]  # (B, 8, 3)
    
    # Apply rotation and translation
    corners = np.zeros((B, 8, 3))
    for i in range(B):
        # Rotate corners
        rotated_corners = (rotation_matrices[i] @ scaled_corners[i].T).T  # (8, 3)
        # Translate to center
        corners[i] = rotated_corners + centers[i]
    
    return torch.from_numpy(corners).to(device)

def orthonormalize_basis(x, y, z):
    """Orthonormalize a set of 3 vectors using the Gram-Schmidt process."""
    x = x / np.linalg.norm(x)
    y = y - np.dot(y, x) * x
    y = y / np.linalg.norm(y)
    z = z - np.dot(z, x) * x - np.dot(z, y) * y
    z = z / np.linalg.norm(z)
    return x, y, z

"""
Old example usage 
original_bbox3d = bbox3d.copy()  # Keep original for comparison
reordered_bboxes = reorder_corners_pca(original_bbox3d)
obb = corners_to_obb(reordered_bboxes)
print(f"obb Shape: {obb.shape}")
reconstructed_obb = obb_to_corners(obb)
print(f"Reconstructed Corners Shape: {reconstructed_obb.shape}")

compare_bboxes(reordered_bboxes, reconstructed_obb, colors=['blue', 'red'])

"""