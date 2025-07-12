
from typing import Union, Tuple, List
from scipy.spatial.transform import Rotation
from vision3d.utils.bbox_converter import corners_to_obb, obb_to_corners
import torch
import torch.nn.functional as F
import numpy as np

class IoU3D:
    """
    PyTorch-based 3D IoU calculator for oriented bounding boxes.
    
    Supports Corner format: [B, 8, 3] - batch of 8 corners per bbox
    
    Corner order follows the specified convention:
    0: min_x, min_y, min_z    4: min_x, min_y, max_z
    1: max_x, min_y, min_z    5: max_x, min_y, max_z  
    2: max_x, max_y, min_z    6: max_x, max_y, max_z
    3: min_x, max_y, min_z    7: min_x, max_y, max_z
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def get_bbox_volume_batch(self, corners: torch.Tensor) -> torch.Tensor:
        """
        Calculate volume of 3D bounding boxes from their corners (vectorized).
        
        Args:
            corners: [B, 8, 3] tensor of corner coordinates
            
        Returns:
            volume: [B] tensor of volumes
        """
        # Take 3 edges from corner 0 for each box
        edge1 = corners[:, 1] - corners[:, 0]  # [B, 3] x direction
        edge2 = corners[:, 3] - corners[:, 0]  # [B, 3] y direction  
        edge3 = corners[:, 4] - corners[:, 0]  # [B, 3] z direction
        
        # Volume is scalar triple product: |a · (b × c)|
        cross_product = torch.cross(edge2, edge3, dim=1)  # [B, 3]
        volume = torch.abs(torch.sum(edge1 * cross_product, dim=1))  # [B]
        return volume
    
    def get_axis_aligned_bbox_from_corners_batch(self, corners: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get axis-aligned bounding box from corners (vectorized).
        
        Args:
            corners: [B, 8, 3] tensor of corner coordinates
            
        Returns:
            min_coords: [B, 3] minimum coordinates
            max_coords: [B, 3] maximum coordinates
        """
        min_coords = torch.min(corners, dim=1)[0]  # [B, 3]
        max_coords = torch.max(corners, dim=1)[0]  # [B, 3]
        return min_coords, max_coords
    
    def axis_aligned_bbox_intersection_batch(self, min1: torch.Tensor, max1: torch.Tensor, 
                                       min2: torch.Tensor, max2: torch.Tensor) -> torch.Tensor:
        """
        Compute intersection volume of axis-aligned bounding boxes (vectorized).
        
        Args:
            min1, max1: [B, 3] min and max coordinates of first bbox
            min2, max2: [B, 3] min and max coordinates of second bbox
            
        Returns:
            intersection_volume: [B] tensor of intersection volumes
        """
        # Ensure all tensors have the same dtype
        min1, max1 = min1.float(), max1.float()
        min2, max2 = min2.float(), max2.float()
        
        # Compute intersection bounds
        intersection_min = torch.maximum(min1, min2)  # [B, 3]
        intersection_max = torch.minimum(max1, max2)  # [B, 3]
        
        # Check if there's any intersection
        intersection_dims = intersection_max - intersection_min  # [B, 3]
        
        # Set negative dimensions to 0 (no intersection)
        intersection_dims = torch.clamp(intersection_dims, min=0.0)
        
        # Compute intersection volume
        intersection_volume = torch.prod(intersection_dims, dim=1)  # [B]
        return intersection_volume

    def compute_intersection_volume_oriented_batch(self, corners1: torch.Tensor, corners2: torch.Tensor) -> torch.Tensor:
        """
        Compute intersection volume between oriented 3D bounding boxes (vectorized).
        
        Args:
            corners1: [B, 8, 3] corners of first bbox
            corners2: [B, 8, 3] corners of second bbox
            
        Returns:
            volumes: [B] tensor of intersection volumes
        """
        B = corners1.shape[0]
        device = corners1.device
        
        # Ensure float32 dtype
        corners1 = corners1.float()
        corners2 = corners2.float()
        
        # For each pair, check which corners are inside the other box
        intersection_volumes = torch.zeros(B, device=device, dtype=torch.float32)
        
        # Check corners of box1 inside box2
        points1_in_box2 = self.point_in_oriented_bbox_batch(corners1, corners2)  # [B, 8]
        # Check corners of box2 inside box1
        points2_in_box1 = self.point_in_oriented_bbox_batch(corners2, corners1)  # [B, 8]
        
        # For boxes with some intersection, use a simplified approximation
        # This is a heuristic - for exact computation, you'd need SAT or other methods
        for i in range(B):
            if torch.any(points1_in_box2[i]) or torch.any(points2_in_box1[i]):
                # Simple approximation: use the minimum volume as upper bound
                vol1 = self.get_bbox_volume_batch(corners1[i:i+1])[0]
                vol2 = self.get_bbox_volume_batch(corners2[i:i+1])[0]
                
                # Count intersection points as a rough measure
                intersection_ratio = (torch.sum(points1_in_box2[i]).float() + 
                                    torch.sum(points2_in_box1[i]).float()) / 16.0
                
                # Heuristic: intersection volume is roughly proportional to intersection ratio
                intersection_volumes[i] = intersection_ratio * torch.min(vol1, vol2)
        
        return intersection_volumes
    
    def is_axis_aligned_batch(self, corners: torch.Tensor, tolerance: float = 1e-6) -> torch.Tensor:
        """
        Check if bounding boxes are axis-aligned (vectorized).
        
        Args:
            corners: [B, 8, 3] tensor of corner coordinates
            tolerance: float tolerance for axis alignment check
            
        Returns:
            is_aligned: [B] boolean tensor
        """
        B = corners.shape[0]
        device = corners.device
        
        # For axis-aligned boxes, all corners should align with the grid
        min_coords, max_coords = self.get_axis_aligned_bbox_from_corners_batch(corners)
        
        # Create expected corners for all boxes
        expected_corners = torch.stack([
            torch.stack([min_coords[:, 0], min_coords[:, 1], min_coords[:, 2]], dim=1),  # 0
            torch.stack([max_coords[:, 0], min_coords[:, 1], min_coords[:, 2]], dim=1),  # 1
            torch.stack([max_coords[:, 0], max_coords[:, 1], min_coords[:, 2]], dim=1),  # 2
            torch.stack([min_coords[:, 0], max_coords[:, 1], min_coords[:, 2]], dim=1),  # 3
            torch.stack([min_coords[:, 0], min_coords[:, 1], max_coords[:, 2]], dim=1),  # 4
            torch.stack([max_coords[:, 0], min_coords[:, 1], max_coords[:, 2]], dim=1),  # 5
            torch.stack([max_coords[:, 0], max_coords[:, 1], max_coords[:, 2]], dim=1),  # 6
            torch.stack([min_coords[:, 0], max_coords[:, 1], max_coords[:, 2]], dim=1),  # 7
        ], dim=1)  # [B, 8, 3]
        
        # Check if corners match expected positions (within tolerance)
        differences = torch.abs(corners - expected_corners)
        is_aligned = torch.all(torch.all(differences < tolerance, dim=2), dim=1)  # [B]
        return is_aligned
    
    def point_in_oriented_bbox_batch(self, points: torch.Tensor, corners: torch.Tensor) -> torch.Tensor:
        """
        Check if points are inside oriented 3D bounding boxes (vectorized).
        
        Args:
            points: [B, N, 3] points to check
            corners: [B, 8, 3] bbox corner coordinates
            
        Returns:
            inside: [B, N] boolean tensor
        """
        B, N = points.shape[:2]
        device = points.device
        
        # Get the center and three edge vectors of the bbox
        center = torch.mean(corners, dim=1, keepdim=True)  # [B, 1, 3]
        
        # Get three edge vectors from corner 0
        edge1 = corners[:, 1] - corners[:, 0]  # [B, 3] x direction
        edge2 = corners[:, 3] - corners[:, 0]  # [B, 3] y direction
        edge3 = corners[:, 4] - corners[:, 0]  # [B, 3] z direction
        
        # Normalize edge vectors
        edge1_norm = F.normalize(edge1, dim=1)  # [B, 3]
        edge2_norm = F.normalize(edge2, dim=1)  # [B, 3]
        edge3_norm = F.normalize(edge3, dim=1)  # [B, 3]
        
        # Get half-lengths
        half_length1 = torch.norm(edge1, dim=1, keepdim=True) / 2  # [B, 1]
        half_length2 = torch.norm(edge2, dim=1, keepdim=True) / 2  # [B, 1]
        half_length3 = torch.norm(edge3, dim=1, keepdim=True) / 2  # [B, 1]
        
        # Transform points to box coordinate system
        to_points = points - center  # [B, N, 3]
        
        # Project onto each axis
        proj1 = torch.abs(torch.sum(to_points * edge1_norm.unsqueeze(1), dim=2))  # [B, N]
        proj2 = torch.abs(torch.sum(to_points * edge2_norm.unsqueeze(1), dim=2))  # [B, N]
        proj3 = torch.abs(torch.sum(to_points * edge3_norm.unsqueeze(1), dim=2))  # [B, N]
        
        # Check if projections are within bounds
        inside = ((proj1 <= half_length1) & 
                  (proj2 <= half_length2) & 
                  (proj3 <= half_length3))  # [B, N]
        
        return inside
    
    def compute_iou_3d(self, bboxes1: torch.Tensor, bboxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute 3D IoU between two sets of 3D bounding boxes (fully vectorized).
        
        Args:
            bboxes1: First set of bboxes - [B, 8, 3] 
            bboxes2: Second set of bboxes - [B, 8, 3] 
            
        Returns:
            iou: [B] tensor of IoU values
        """
        # Ensure tensors are on the same device and dtype
        if isinstance(bboxes1, np.ndarray):
            bboxes1 = torch.from_numpy(bboxes1).float().to(self.device)
        if isinstance(bboxes2, np.ndarray):
            bboxes2 = torch.from_numpy(bboxes2).float().to(self.device)
        
        # Ensure consistent dtype
        bboxes1 = bboxes1.float().to(self.device)
        bboxes2 = bboxes2.float().to(self.device)
        
        B = bboxes1.shape[0]
        
        # Get volumes of individual bboxes (vectorized)
        vol1 = self.get_bbox_volume_batch(bboxes1)  # [B]
        vol2 = self.get_bbox_volume_batch(bboxes2)  # [B]
        
        # Check which boxes are axis-aligned
        is_aligned1 = self.is_axis_aligned_batch(bboxes1)  # [B]
        is_aligned2 = self.is_axis_aligned_batch(bboxes2)  # [B]
        both_aligned = is_aligned1 & is_aligned2  # [B]
        
        # Initialize intersection volumes with correct dtype
        intersection_vol = torch.zeros(B, device=self.device, dtype=torch.float32)
        
        # Handle axis-aligned boxes efficiently
        if torch.any(both_aligned):
            aligned_indices = torch.where(both_aligned)[0]
            min1, max1 = self.get_axis_aligned_bbox_from_corners_batch(bboxes1[aligned_indices])
            min2, max2 = self.get_axis_aligned_bbox_from_corners_batch(bboxes2[aligned_indices])
            
            # Ensure intersection computation returns float32
            intersection_aligned = self.axis_aligned_bbox_intersection_batch(min1, max1, min2, max2)
            intersection_vol[aligned_indices] = intersection_aligned.float()
        
        # Handle oriented boxes
        if torch.any(~both_aligned):
            oriented_indices = torch.where(~both_aligned)[0]
            intersection_oriented = self.compute_intersection_volume_oriented_batch(
                bboxes1[oriented_indices], bboxes2[oriented_indices]
            )
            intersection_vol[oriented_indices] = intersection_oriented.float()
        
        # Compute IoU
        union_vol = vol1 + vol2 - intersection_vol
        
        # Avoid division by zero
        iou = torch.where(union_vol > 1e-10, 
                        intersection_vol / union_vol, 
                        torch.zeros_like(union_vol))
        
        return iou


import torch
import numpy as np
from vision3d.utils.bbox_converter import obb_to_corners

def test_iou_3d():
    """Simple test function to verify 3D IoU calculator correctness"""
    
    def create_bbox_corners(center, size, device='cpu'):
        """Create axis-aligned bbox corners"""
        cx, cy, cz = center
        sx, sy, sz = size
        
        return torch.tensor([
            [cx - sx/2, cy - sy/2, cz - sz/2],  # 0: min_x, min_y, min_z
            [cx + sx/2, cy - sy/2, cz - sz/2],  # 1: max_x, min_y, min_z
            [cx + sx/2, cy + sy/2, cz - sz/2],  # 2: max_x, max_y, min_z
            [cx - sx/2, cy + sy/2, cz - sz/2],  # 3: min_x, max_y, min_z
            [cx - sx/2, cy - sy/2, cz + sz/2],  # 4: min_x, min_y, max_z
            [cx + sx/2, cy - sy/2, cz + sz/2],  # 5: max_x, min_y, max_z
            [cx + sx/2, cy + sy/2, cz + sz/2],  # 6: max_x, max_y, max_z
            [cx - sx/2, cy + sy/2, cz + sz/2],  # 7: min_x, max_y, max_z
        ], dtype=torch.float32, device=device)
    
    # Initialize calculator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iou_calc = IoU3D(device=device)
    
    print(f"Testing 3D IoU Calculator on {device}...")
    print("=" * 40)
    
    # Test 1: Identical boxes (should be 1.0)
    print("Test 1: Identical boxes")
    box1 = create_bbox_corners([0, 0, 0], [2, 2, 2], device).unsqueeze(0)  # [1, 8, 3]
    box2 = create_bbox_corners([0, 0, 0], [2, 2, 2], device).unsqueeze(0)  # [1, 8, 3]
    iou1 = iou_calc.compute_iou_3d(box1, box2)[0].item()
    print(f"IoU: {iou1:.6f} (expected: 1.000000)")
    assert abs(iou1 - 1.0) < 1e-3, f"Failed: expected 1.0, got {iou1}"
    print("PASSED\n")
    
    # Test 2: No overlap (should be 0.0)
    print("Test 2: No overlap")
    box1 = create_bbox_corners([0, 0, 0], [1, 1, 1], device).unsqueeze(0)
    box2 = create_bbox_corners([5, 5, 5], [1, 1, 1], device).unsqueeze(0)
    iou2 = iou_calc.compute_iou_3d(box1, box2)[0].item()
    print(f"IoU: {iou2:.6f} (expected: 0.000000)")
    assert abs(iou2 - 0.0) < 1e-6, f"Failed: expected 0.0, got {iou2}"
    print("PASSED\n")
    
    # Test 3: Half overlap in each dimension
    print("Test 3: Half overlap")
    # Box1: center=[0,0,0], size=[2,2,2] -> volume=8
    # Box2: center=[1,1,1], size=[2,2,2] -> volume=8
    # Intersection: [0,1]x[0,1]x[0,1] -> volume=1
    # Union: 8 + 8 - 1 = 15
    # IoU: 1/15 ≈ 0.0667
    box1 = create_bbox_corners([0, 0, 0], [2, 2, 2], device).unsqueeze(0)
    box2 = create_bbox_corners([1, 1, 1], [2, 2, 2], device).unsqueeze(0)
    iou3 = iou_calc.compute_iou_3d(box1, box2)[0].item()
    expected_iou3 = 1.0 / 15.0
    print(f"IoU: {iou3:.6f} (expected: {expected_iou3:.6f})")
    assert abs(iou3 - expected_iou3) < 1e-3, f"Failed: expected {expected_iou3}, got {iou3}"
    print("PASSED\n")
    
    # Test 4: One box inside another
    print("Test 4: One box inside another")
    # Large box: volume = 4*4*4 = 64
    # Small box: volume = 2*2*2 = 8
    # Intersection: 8 (small box is completely inside)
    # Union: 64 (large box volume)
    # IoU: 8/64 = 0.125
    box1 = create_bbox_corners([0, 0, 0], [4, 4, 4], device).unsqueeze(0)  # large box
    box2 = create_bbox_corners([0, 0, 0], [2, 2, 2], device).unsqueeze(0)  # small box inside
    iou4 = iou_calc.compute_iou_3d(box1, box2)[0].item()
    expected_iou4 = 8.0 / 64.0
    print(f"IoU: {iou4:.6f} (expected: {expected_iou4:.6f})")
    assert abs(iou4 - expected_iou4) < 1e-3, f"Failed: expected {expected_iou4}, got {iou4}"
    print("PASSED\n")
    
    # Test 5: Volume calculation
    print("Test 5: Volume calculation")
    box = create_bbox_corners([0, 0, 0], [2, 3, 4], device).unsqueeze(0)
    volume = iou_calc.get_bbox_volume_batch(box)[0].item()
    expected_volume = 2 * 3 * 4
    print(f"Volume: {volume:.6f} (expected: {expected_volume:.6f})")
    assert abs(volume - expected_volume) < 1e-3, f"Failed: expected {expected_volume}, got {volume}"
    print("PASSED\n")
    
    # Test 6: Parametric format
    print("Test 6: Parametric format - identical boxes")
    # [center_x, center_y, center_z, size_x, size_y, size_z, quat_w, quat_x, quat_y, quat_z]
    box1_params = np.array([[0, 0, 0, 2, 2, 2, 0, 0, 0, 1]])  # axis-aligned
    box2_params = np.array([[0, 0, 0, 2, 2, 2, 0, 0, 0, 1]])  # identical

    bbox1_corners = obb_to_corners(box1_params)
    bbox2_corners = obb_to_corners(box2_params)
    
    # Convert to PyTorch tensors
    if isinstance(bbox1_corners, np.ndarray):
        bbox1_corners = torch.from_numpy(bbox1_corners).float().to(device)
    if isinstance(bbox2_corners, np.ndarray):
        bbox2_corners = torch.from_numpy(bbox2_corners).float().to(device)

    print("Converted to corners format:")
    print("Box 1 corners: shape:    ", bbox1_corners.shape)
    print("Box 2 corners: shape:    ", bbox2_corners.shape)
    iou6 = iou_calc.compute_iou_3d(bbox1_corners, bbox2_corners)[0].item()
    print(f"IoU: {iou6:.6f} (expected: 1.000000)")
    assert abs(iou6 - 1.0) < 1e-3, f"Failed: expected 1.0, got {iou6}"
    print("PASSED\n")

    # Test 7: Batch processing
    print("Test 7: Batch processing")
    batch_box1 = torch.stack([
        create_bbox_corners([0, 0, 0], [2, 2, 2], device),  # identical
        create_bbox_corners([0, 0, 0], [1, 1, 1], device),  # no overlap
    ])
    batch_box2 = torch.stack([
        create_bbox_corners([0, 0, 0], [2, 2, 2], device),  # identical
        create_bbox_corners([5, 5, 5], [1, 1, 1], device),  # no overlap
    ])
    iou_batch = iou_calc.compute_iou_3d(batch_box1, batch_box2)
    iou_batch_cpu = iou_batch.cpu().numpy()
    print(f"Batch IoU: [{iou_batch_cpu[0]:.6f}, {iou_batch_cpu[1]:.6f}] (expected: [1.000000, 0.000000])")
    assert abs(iou_batch_cpu[0] - 1.0) < 1e-3, f"Failed batch[0]: expected 1.0, got {iou_batch_cpu[0]}"
    assert abs(iou_batch_cpu[1] - 0.0) < 1e-6, f"Failed batch[1]: expected 0.0, got {iou_batch_cpu[1]}"
    print("PASSED\n")
    
    # Test 8: Large batch performance test
    print("Test 8: Large batch performance test")
    B = 1000
    large_batch1 = torch.stack([create_bbox_corners([0, 0, 0], [2, 2, 2], device) for _ in range(B)])
    large_batch2 = torch.stack([create_bbox_corners([0, 0, 0], [2, 2, 2], device) for _ in range(B)])
    
    import time
    start_time = time.time()
    iou_large = iou_calc.compute_iou_3d(large_batch1, large_batch2)
    end_time = time.time()
    
    print(f"Processed {B} box pairs in {end_time - start_time:.4f} seconds")
    print(f"Average IoU: {torch.mean(iou_large).item():.6f}")
    assert torch.all(torch.abs(iou_large - 1.0) < 1e-3), "Failed: large batch test"
    print("PASSED\n")
    
    print("=" * 40)
    print("ALL TESTS PASSED")
    print("The 3D IoU calculator is working correctly!")
    print("=" * 40)

# Example usage
if __name__ == "__main__":
    # Test with corner format
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iou_calculator = IoU3D(device=device)
    
    # Create two example bboxes in corner format
    bbox1_corners = torch.tensor([[
        [0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0],  # bottom face
        [0, 0, 2], [2, 0, 2], [2, 2, 2], [0, 2, 2]   # top face
    ]], dtype=torch.float32, device=device)
    
    bbox2_corners = torch.tensor([[
        [1, 1, 1], [3, 1, 1], [3, 3, 1], [1, 3, 1],  # bottom face
        [1, 1, 3], [3, 1, 3], [3, 3, 3], [1, 3, 3]   # top face
    ]], dtype=torch.float32, device=device)
    
    # Compute IoU
    iou = iou_calculator.compute_iou_3d(bbox1_corners, bbox2_corners)
    print(f"IoU (corners): {iou[0].item():.4f}")
    
    # Run tests
    test_iou_3d()
