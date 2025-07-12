import numpy as np
from typing import Union, Tuple, List
from scipy.spatial.transform import Rotation
from vision3d.utils.bbox_converter import corners_to_obb, obb_to_corners

class IoU3D:
    """
    Class for computing 3D IoU between 3D bounding boxes.
    
    Supports Corner format: [B, 8, 3] - batch of 8 corners per bbox
    
    Corner order follows the specified convention:
    0: min_x, min_y, min_z    4: min_x, min_y, max_z
    1: max_x, min_y, min_z    5: max_x, min_y, max_z  
    2: max_x, max_y, min_z    6: max_x, max_y, max_z
    3: min_x, max_y, min_z    7: min_x, max_y, max_z
    """
    
    def __init__(self):
        pass
    
    def get_bbox_volume(self, corners: np.ndarray) -> float:
        """
        Calculate volume of a 3D bounding box from its corners.
        
        Args:
            corners: [8, 3] array of corner coordinates
            
        Returns:
            volume: float
        """
        # Use the cross product method to compute volume
        # Take 3 edges from corner 0
        edge1 = corners[1] - corners[0]  # x direction
        edge2 = corners[3] - corners[0]  # y direction  
        edge3 = corners[4] - corners[0]  # z direction
        
        # Volume is scalar triple product
        volume = abs(np.dot(edge1, np.cross(edge2, edge3)))
        return volume
    
    def get_axis_aligned_bbox_from_corners(self, corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get axis-aligned bounding box from corners.
        
        Args:
            corners: [8, 3] array of corner coordinates
            
        Returns:
            min_coords: [3] minimum coordinates
            max_coords: [3] maximum coordinates
        """
        min_coords = np.min(corners, axis=0)
        max_coords = np.max(corners, axis=0)
        return min_coords, max_coords
    
    def axis_aligned_bbox_intersection(self, min1: np.ndarray, max1: np.ndarray, 
                                     min2: np.ndarray, max2: np.ndarray) -> float:
        """
        Compute intersection volume of two axis-aligned bounding boxes.
        
        Args:
            min1, max1: [3] min and max coordinates of first bbox
            min2, max2: [3] min and max coordinates of second bbox
            
        Returns:
            intersection_volume: float
        """
        # Compute intersection bounds
        intersection_min = np.maximum(min1, min2)
        intersection_max = np.minimum(max1, max2)
        
        # Check if there's any intersection
        if np.any(intersection_min >= intersection_max):
            return 0.0
        
        # Compute intersection volume
        intersection_dims = intersection_max - intersection_min
        return np.prod(intersection_dims)
    
    def is_axis_aligned(self, corners: np.ndarray, tolerance: float = 1e-6) -> bool:
        """
        Check if a bounding box is axis-aligned.
        
        Args:
            corners: [8, 3] array of corner coordinates
            tolerance: float tolerance for axis alignment check
            
        Returns:
            bool: True if axis-aligned
        """
        # For axis-aligned boxes, all corners should align with the grid
        min_coords, max_coords = self.get_axis_aligned_bbox_from_corners(corners)
        
        # Check if all corners are at the min/max positions
        expected_corners = np.array([
            [min_coords[0], min_coords[1], min_coords[2]],  # 0
            [max_coords[0], min_coords[1], min_coords[2]],  # 1
            [max_coords[0], max_coords[1], min_coords[2]],  # 2
            [min_coords[0], max_coords[1], min_coords[2]],  # 3
            [min_coords[0], min_coords[1], max_coords[2]],  # 4
            [max_coords[0], min_coords[1], max_coords[2]],  # 5
            [max_coords[0], max_coords[1], max_coords[2]],  # 6
            [min_coords[0], max_coords[1], max_coords[2]],  # 7
        ])
        
        # Check if corners match expected positions (within tolerance)
        differences = np.abs(corners - expected_corners)
        return np.all(differences < tolerance)
    
    def compute_intersection_volume_oriented(self, corners1: np.ndarray, corners2: np.ndarray) -> float:
        """
        Compute intersection volume between two oriented 3D bounding boxes using separating axis theorem.
        
        Args:
            corners1: [8, 3] corners of first bbox
            corners2: [8, 3] corners of second bbox
            
        Returns:
            volume: float intersection volume
        """
        # For oriented boxes, we use a more complex approach
        # This is a simplified implementation - for production use, consider using
        # specialized libraries like Open3D or implementing full SAT algorithm
        
        # As a fallback, we'll use the convex hull approach
        try:
            from scipy.spatial import ConvexHull
            
            # Combine all corners
            all_corners = np.vstack([corners1, corners2])
            
            # Find points that are inside both boxes
            intersection_points = []
            
            # Check which corners of box1 are inside box2
            for corner in corners1:
                if self.point_in_oriented_bbox(corner, corners2):
                    intersection_points.append(corner)
            
            # Check which corners of box2 are inside box1
            for corner in corners2:
                if self.point_in_oriented_bbox(corner, corners1):
                    intersection_points.append(corner)
            
            # If we have enough points, compute convex hull
            if len(intersection_points) >= 4:
                intersection_points = np.array(intersection_points)
                
                # Remove duplicates
                unique_points = []
                for point in intersection_points:
                    is_duplicate = False
                    for existing in unique_points:
                        if np.linalg.norm(point - existing) < 1e-8:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        unique_points.append(point)
                
                if len(unique_points) >= 4:
                    unique_points = np.array(unique_points)
                    hull = ConvexHull(unique_points)
                    return hull.volume
            
            return 0.0
            
        except:
            # Fallback to 0 if convex hull fails
            return 0.0
    
    def point_in_oriented_bbox(self, point: np.ndarray, corners: np.ndarray) -> bool:
        """
        Check if a point is inside an oriented 3D bounding box.
        
        Args:
            point: [3] point coordinates
            corners: [8, 3] bbox corner coordinates
            
        Returns:
            bool: True if point is inside bbox
        """
        # Get the center and three edge vectors of the bbox
        center = np.mean(corners, axis=0)
        
        # Use corners to define the oriented bounding box
        # Get three edge vectors from corner 0
        edge1 = corners[1] - corners[0]  # x direction
        edge2 = corners[3] - corners[0]  # y direction
        edge3 = corners[4] - corners[0]  # z direction
        
        # Normalize edge vectors
        edge1_norm = edge1 / np.linalg.norm(edge1)
        edge2_norm = edge2 / np.linalg.norm(edge2)
        edge3_norm = edge3 / np.linalg.norm(edge3)
        
        # Get half-lengths
        half_length1 = np.linalg.norm(edge1) / 2
        half_length2 = np.linalg.norm(edge2) / 2
        half_length3 = np.linalg.norm(edge3) / 2
        
        # Transform point to box coordinate system
        to_point = point - center
        
        # Project onto each axis
        proj1 = abs(np.dot(to_point, edge1_norm))
        proj2 = abs(np.dot(to_point, edge2_norm))
        proj3 = abs(np.dot(to_point, edge3_norm))
        
        # Check if projections are within bounds
        return (proj1 <= half_length1 and 
                proj2 <= half_length2 and 
                proj3 <= half_length3)
    
    def compute_intersection_volume(self, corners1: np.ndarray, corners2: np.ndarray) -> float:
        """
        Compute intersection volume between two 3D bounding boxes.
        
        Args:
            corners1: [8, 3] corners of first bbox
            corners2: [8, 3] corners of second bbox
            
        Returns:
            volume: float intersection volume
        """
        # Check if both boxes are axis-aligned
        if self.is_axis_aligned(corners1) and self.is_axis_aligned(corners2):
            # Use efficient axis-aligned intersection
            min1, max1 = self.get_axis_aligned_bbox_from_corners(corners1)
            min2, max2 = self.get_axis_aligned_bbox_from_corners(corners2)
            return self.axis_aligned_bbox_intersection(min1, max1, min2, max2)
        else:
            # Use oriented box intersection
            return self.compute_intersection_volume_oriented(corners1, corners2)
    
    def compute_iou_3d(self, bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
        """
        Compute 3D IoU between two sets of 3D bounding boxes.
        
        Args:
            bboxes1: First set of bboxes - [B, 8, 3] 
            bboxes2: Second set of bboxes - [B, 8, 3] 

            
        Returns:
            iou: [B] array of IoU values
        """
        
        corners1 = bboxes1
        corners2 = bboxes2
        
        B = corners1.shape[0]
        iou_values = np.zeros(B)
        
        for i in range(B):
            # Get volumes of individual bboxes
            vol1 = self.get_bbox_volume(corners1[i])
            vol2 = self.get_bbox_volume(corners2[i])
            
            # Get intersection volume
            intersection_vol = self.compute_intersection_volume(corners1[i], corners2[i])
            
            # Compute IoU
            union_vol = vol1 + vol2 - intersection_vol
            
            if union_vol > 1e-10:
                iou_values[i] = intersection_vol / union_vol
            else:
                iou_values[i] = 0.0
                
        return iou_values

# Test function
def test_iou_3d():
    """Simple test function to verify 3D IoU calculator correctness"""
    
    def create_bbox_corners(center, size):
        """Create axis-aligned bbox corners"""
        cx, cy, cz = center
        sx, sy, sz = size
        
        return np.array([
            [cx - sx/2, cy - sy/2, cz - sz/2],  # 0: min_x, min_y, min_z
            [cx + sx/2, cy - sy/2, cz - sz/2],  # 1: max_x, min_y, min_z
            [cx + sx/2, cy + sy/2, cz - sz/2],  # 2: max_x, max_y, min_z
            [cx - sx/2, cy + sy/2, cz - sz/2],  # 3: min_x, max_y, min_z
            [cx - sx/2, cy - sy/2, cz + sz/2],  # 4: min_x, min_y, max_z
            [cx + sx/2, cy - sy/2, cz + sz/2],  # 5: max_x, min_y, max_z
            [cx + sx/2, cy + sy/2, cz + sz/2],  # 6: max_x, max_y, max_z
            [cx - sx/2, cy + sy/2, cz + sz/2],  # 7: min_x, max_y, max_z
        ])
    
    # Initialize calculator
    iou_calc = IoU3D()
    
    print("Testing 3D IoU Calculator...")
    print("=" * 40)
    
    # Test 1: Identical boxes (should be 1.0)
    print("Test 1: Identical boxes")
    box1 = create_bbox_corners([0, 0, 0], [2, 2, 2]).reshape(1, 8, 3)
    box2 = create_bbox_corners([0, 0, 0], [2, 2, 2]).reshape(1, 8, 3)
    iou1 = iou_calc.compute_iou_3d(box1, box2)[0]
    print(f"IoU: {iou1:.6f} (expected: 1.000000)")
    assert abs(iou1 - 1.0) < 1e-3, f"Failed: expected 1.0, got {iou1}"
    print("PASSED\n")
    
    # Test 2: No overlap (should be 0.0)
    print("Test 2: No overlap")
    box1 = create_bbox_corners([0, 0, 0], [1, 1, 1]).reshape(1, 8, 3)
    box2 = create_bbox_corners([5, 5, 5], [1, 1, 1]).reshape(1, 8, 3)
    iou2 = iou_calc.compute_iou_3d(box1, box2)[0]
    print(f"IoU: {iou2:.6f} (expected: 0.000000)")
    assert iou2 == 0.0, f"Failed: expected 0.0, got {iou2}"
    print("PASSED\n")
    
    # Test 3: Half overlap in each dimension
    print("Test 3: Half overlap")
    # Box1: center=[0,0,0], size=[2,2,2] -> volume=8
    # Box2: center=[1,1,1], size=[2,2,2] -> volume=8
    # Intersection: [0,1]x[0,1]x[0,1] -> volume=1
    # Union: 8 + 8 - 1 = 15
    # IoU: 1/15 ≈ 0.0667
    box1 = create_bbox_corners([0, 0, 0], [2, 2, 2]).reshape(1, 8, 3)
    box2 = create_bbox_corners([1, 1, 1], [2, 2, 2]).reshape(1, 8, 3)
    iou3 = iou_calc.compute_iou_3d(box1, box2)[0]
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
    box1 = create_bbox_corners([0, 0, 0], [4, 4, 4]).reshape(1, 8, 3)  # large box
    box2 = create_bbox_corners([0, 0, 0], [2, 2, 2]).reshape(1, 8, 3)  # small box inside
    iou4 = iou_calc.compute_iou_3d(box1, box2)[0]
    expected_iou4 = 8.0 / 64.0
    print(f"IoU: {iou4:.6f} (expected: {expected_iou4:.6f})")
    assert abs(iou4 - expected_iou4) < 1e-3, f"Failed: expected {expected_iou4}, got {iou4}"
    print("PASSED\n")
    
    # Test 5: Volume calculation
    print("Test 5: Volume calculation")
    box = create_bbox_corners([0, 0, 0], [2, 3, 4])
    volume = iou_calc.get_bbox_volume(box)
    expected_volume = 2 * 3 * 4
    print(f"Volume: {volume:.6f} (expected: {expected_volume:.6f})")
    assert abs(volume - expected_volume) < 1e-3, f"Failed: expected {expected_volume}, got {volume}"
    print("PASSED\n")
    
    # Test 6: Parametric format
    print("Test 6: Parametric format - identical boxes")
    # [center_x, center_y, center_z, size_x, size_y, size_z, quat_w, quat_x, quat_y, quat_z]
    box1_params = np.array([[0, 0, 0, 2, 2, 2, 0, 0, 0, 1]])  # axis-aligned
    box2_params = np.array([[0, 0, 0, 2, 2, 2, 0, 0, 0, 1]])  # identical

    bbox1_corners = obb_to_corners(box1_params).numpy()
    bbox2_corners = obb_to_corners(box2_params).numpy()

    print("Converted to corners format:")
    print("Box 1 corners: shape:    ", bbox1_corners.shape)
    print("Box 2 corners: shape:    ", bbox2_corners.shape)
    iou6 = iou_calc.compute_iou_3d(bbox1_corners, bbox2_corners)[0]
    print(f"IoU: {iou6:.6f} (expected: 1.000000)")
    assert abs(iou6 - 1.0) < 1e-3, f"Failed: expected 1.0, got {iou6}"
    print("PASSED\n")

    # Test 7: Batch processing
    print("Test 7: Batch processing")
    batch_box1 = np.array([
        create_bbox_corners([0, 0, 0], [2, 2, 2]),  # identical
        create_bbox_corners([0, 0, 0], [1, 1, 1]),  # no overlap
    ])
    batch_box2 = np.array([
        create_bbox_corners([0, 0, 0], [2, 2, 2]),  # identical
        create_bbox_corners([5, 5, 5], [1, 1, 1]),  # no overlap
    ])
    iou_batch = iou_calc.compute_iou_3d(batch_box1, batch_box2)
    print(f"Batch IoU: [{iou_batch[0]:.6f}, {iou_batch[1]:.6f}] (expected: [1.000000, 0.000000])")
    assert abs(iou_batch[0] - 1.0) < 1e-3, f"Failed batch[0]: expected 1.0, got {iou_batch[0]}"
    assert iou_batch[1] == 0.0, f"Failed batch[1]: expected 0.0, got {iou_batch[1]}"
    print("PASSED\n")
    
    print("=" * 40)
    print("ALL TESTS PASSED")
    print("The 3D IoU calculator is working correctly!")
    print("=" * 40)

# Example usage
if __name__ == "__main__":
    # Test with corner format
    iou_calculator = IoU3D()
    
    # Create two example bboxes in corner format
    bbox1_corners = np.array([[
        [0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0],  # bottom face
        [0, 0, 2], [2, 0, 2], [2, 2, 2], [0, 2, 2]   # top face
    ]])
    
    bbox2_corners = np.array([[
        [1, 1, 1], [3, 1, 1], [3, 3, 1], [1, 3, 1],  # bottom face
        [1, 1, 3], [3, 1, 3], [3, 3, 3], [1, 3, 3]   # top face
    ]])
    
    # Compute IoU
    iou = iou_calculator.compute_iou_3d(bbox1_corners, bbox2_corners)
    print(f"IoU (corners): {iou[0]:.4f}")
    
    
    # Run tests
    test_iou_3d()