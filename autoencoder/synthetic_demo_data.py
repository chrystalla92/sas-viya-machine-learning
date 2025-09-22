"""
Synthetic Data Generator for Autoencoder Demonstration

This module generates simple synthetic 2D patterns that can be used to demonstrate
the autoencoder functionality without requiring real image datasets like MNIST.

The patterns are designed to be:
- Simple geometric shapes (circles, rectangles, lines)  
- Easy to visualize and understand
- Challenging enough to show autoencoder reconstruction capabilities
- Scalable in complexity
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import torch


class SyntheticPatternGenerator:
    """Generator for simple 2D synthetic patterns."""
    
    def __init__(self, image_size: int = 28, seed: Optional[int] = 42):
        """
        Initialize the pattern generator.
        
        Args:
            image_size: Size of generated square images (image_size x image_size)
            seed: Random seed for reproducible generation
        """
        self.image_size = image_size
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def generate_circle(self, center: Optional[Tuple[int, int]] = None, 
                       radius: Optional[int] = None, 
                       thickness: int = 1) -> np.ndarray:
        """
        Generate a circle pattern.
        
        Args:
            center: Circle center (x, y). If None, random center is chosen
            radius: Circle radius. If None, random radius is chosen
            thickness: Thickness of the circle line
            
        Returns:
            2D numpy array with circle pattern
        """
        image = np.zeros((self.image_size, self.image_size))
        
        if center is None:
            center = (
                np.random.randint(self.image_size // 4, 3 * self.image_size // 4),
                np.random.randint(self.image_size // 4, 3 * self.image_size // 4)
            )
        
        if radius is None:
            max_radius = min(center[0], center[1], 
                           self.image_size - center[0], 
                           self.image_size - center[1]) - 2
            radius = np.random.randint(3, max(4, max_radius))
        
        # Create circle using distance formula
        y, x = np.ogrid[:self.image_size, :self.image_size]
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        
        # Create circle with thickness
        circle_mask = (dist_from_center >= radius - thickness/2) & \
                     (dist_from_center <= radius + thickness/2)
        image[circle_mask] = 1.0
        
        return image
    
    def generate_rectangle(self, top_left: Optional[Tuple[int, int]] = None,
                          width: Optional[int] = None,
                          height: Optional[int] = None,
                          filled: bool = False) -> np.ndarray:
        """
        Generate a rectangle pattern.
        
        Args:
            top_left: Top-left corner (x, y). If None, random position chosen
            width: Rectangle width. If None, random width chosen
            height: Rectangle height. If None, random height chosen
            filled: Whether rectangle is filled or just outline
            
        Returns:
            2D numpy array with rectangle pattern
        """
        image = np.zeros((self.image_size, self.image_size))
        
        if top_left is None:
            top_left = (
                np.random.randint(2, self.image_size // 2),
                np.random.randint(2, self.image_size // 2)
            )
        
        if width is None:
            max_width = self.image_size - top_left[0] - 2
            width = np.random.randint(4, max(5, max_width))
            
        if height is None:
            max_height = self.image_size - top_left[1] - 2
            height = np.random.randint(4, max(5, max_height))
        
        x1, y1 = top_left
        x2, y2 = min(x1 + width, self.image_size - 1), min(y1 + height, self.image_size - 1)
        
        if filled:
            image[y1:y2, x1:x2] = 1.0
        else:
            # Draw outline
            image[y1:y2, x1] = 1.0  # Left edge
            image[y1:y2, x2-1] = 1.0  # Right edge  
            image[y1, x1:x2] = 1.0  # Top edge
            image[y2-1, x1:x2] = 1.0  # Bottom edge
            
        return image
    
    def generate_diagonal_line(self, start: Optional[Tuple[int, int]] = None,
                              end: Optional[Tuple[int, int]] = None,
                              thickness: int = 1) -> np.ndarray:
        """
        Generate a diagonal line pattern.
        
        Args:
            start: Start point (x, y). If None, random start chosen
            end: End point (x, y). If None, random end chosen
            thickness: Line thickness
            
        Returns:
            2D numpy array with line pattern
        """
        image = np.zeros((self.image_size, self.image_size))
        
        if start is None:
            start = (
                np.random.randint(2, self.image_size // 2),
                np.random.randint(2, self.image_size // 2)
            )
            
        if end is None:
            end = (
                np.random.randint(self.image_size // 2, self.image_size - 2),
                np.random.randint(self.image_size // 2, self.image_size - 2)
            )
        
        # Draw line using Bresenham-like algorithm
        x1, y1 = start
        x2, y2 = end
        
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        x, y = x1, y1
        while True:
            for t in range(-thickness//2, thickness//2 + 1):
                for t2 in range(-thickness//2, thickness//2 + 1):
                    px, py = x + t, y + t2
                    if 0 <= px < self.image_size and 0 <= py < self.image_size:
                        image[py, px] = 1.0
            
            if x == x2 and y == y2:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
                
        return image
    
    def generate_cross(self, center: Optional[Tuple[int, int]] = None,
                      size: Optional[int] = None,
                      thickness: int = 2) -> np.ndarray:
        """
        Generate a cross pattern.
        
        Args:
            center: Cross center (x, y). If None, random center chosen
            size: Cross arm length. If None, random size chosen
            thickness: Cross line thickness
            
        Returns:
            2D numpy array with cross pattern
        """
        image = np.zeros((self.image_size, self.image_size))
        
        if center is None:
            center = (self.image_size // 2, self.image_size // 2)
            
        if size is None:
            max_size = min(center[0], center[1], 
                          self.image_size - center[0],
                          self.image_size - center[1]) - 1
            size = np.random.randint(3, max(4, max_size))
        
        cx, cy = center
        
        # Horizontal line
        x1, x2 = max(0, cx - size), min(self.image_size, cx + size)
        y1, y2 = max(0, cy - thickness//2), min(self.image_size, cy + thickness//2)
        image[y1:y2+1, x1:x2+1] = 1.0
        
        # Vertical line  
        x1, x2 = max(0, cx - thickness//2), min(self.image_size, cx + thickness//2)
        y1, y2 = max(0, cy - size), min(self.image_size, cy + size)
        image[y1:y2+1, x1:x2+1] = 1.0
        
        return image
    
    def generate_random_dots(self, num_dots: Optional[int] = None,
                           dot_size: int = 1) -> np.ndarray:
        """
        Generate random dots pattern.
        
        Args:
            num_dots: Number of dots. If None, random number chosen
            dot_size: Size of each dot
            
        Returns:
            2D numpy array with dots pattern
        """
        image = np.zeros((self.image_size, self.image_size))
        
        if num_dots is None:
            num_dots = np.random.randint(5, 20)
        
        for _ in range(num_dots):
            x = np.random.randint(dot_size, self.image_size - dot_size)
            y = np.random.randint(dot_size, self.image_size - dot_size)
            
            image[y-dot_size:y+dot_size+1, x-dot_size:x+dot_size+1] = 1.0
            
        return image
    
    def generate_dataset(self, num_samples: int = 1000,
                        pattern_types: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a dataset with multiple pattern types.
        
        Args:
            num_samples: Total number of samples to generate
            pattern_types: List of pattern types to include. 
                          Available: ['circle', 'rectangle', 'line', 'cross', 'dots']
                          If None, all types are used
            
        Returns:
            Tuple of (images, labels) where:
            - images: (num_samples, image_size*image_size) flattened images
            - labels: (num_samples,) pattern type labels  
        """
        if pattern_types is None:
            pattern_types = ['circle', 'rectangle', 'line', 'cross', 'dots']
        
        images = []
        labels = []
        
        pattern_generators = {
            'circle': self.generate_circle,
            'rectangle': lambda: self.generate_rectangle(filled=np.random.choice([True, False])),
            'line': self.generate_diagonal_line,
            'cross': self.generate_cross,
            'dots': self.generate_random_dots
        }
        
        for i in range(num_samples):
            # Choose random pattern type
            pattern_type = np.random.choice(pattern_types)
            pattern_label = pattern_types.index(pattern_type)
            
            # Generate pattern
            image = pattern_generators[pattern_type]()
            
            # Add some noise
            noise = np.random.normal(0, 0.05, image.shape)
            image = np.clip(image + noise, 0, 1)
            
            # Flatten image
            flattened_image = image.flatten()
            
            images.append(flattened_image)
            labels.append(pattern_label)
        
        return np.array(images), np.array(labels)


def visualize_samples(images: np.ndarray, labels: np.ndarray, 
                     pattern_types: List[str], 
                     image_size: int = 28,
                     num_samples: int = 10,
                     figsize: Tuple[int, int] = (12, 4)) -> None:
    """
    Visualize sample images from the dataset.
    
    Args:
        images: Dataset images (flattened)
        labels: Dataset labels
        pattern_types: List of pattern type names
        image_size: Size of square images
        num_samples: Number of samples to show
        figsize: Figure size for plotting
    """
    fig, axes = plt.subplots(2, num_samples // 2, figsize=figsize)
    axes = axes.flatten()
    
    # Select diverse samples
    unique_labels = np.unique(labels)
    sample_indices = []
    
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        if len(label_indices) > 0:
            sample_indices.append(np.random.choice(label_indices))
    
    # Fill remaining with random samples
    while len(sample_indices) < num_samples:
        sample_indices.append(np.random.randint(0, len(images)))
    
    sample_indices = sample_indices[:num_samples]
    
    for i, idx in enumerate(sample_indices):
        image = images[idx].reshape(image_size, image_size)
        label = labels[idx]
        
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'{pattern_types[label]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Demonstration of synthetic data generation
    print("Generating synthetic patterns...")
    
    generator = SyntheticPatternGenerator(image_size=28, seed=42)
    
    # Generate small dataset for demonstration
    images, labels = generator.generate_dataset(
        num_samples=500, 
        pattern_types=['circle', 'rectangle', 'line', 'cross', 'dots']
    )
    
    print(f"Generated {len(images)} samples")
    print(f"Image shape: {images.shape}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Visualize samples
    pattern_names = ['circle', 'rectangle', 'line', 'cross', 'dots']
    visualize_samples(images, labels, pattern_names, num_samples=10)
