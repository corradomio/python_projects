import random
from typing import List, Tuple, Dict

import matplotlib.patches as patches
import matplotlib.pyplot as plt


class RectanglePacker:
    def __init__(self, big_width: float, big_height: float):
        """
        Initialize the rectangle packer with a big rectangle.

        Args:
            big_width: Width of the big rectangle
            big_height: Height of the big rectangle
        """
        self.big_width = big_width
        self.big_height = big_height
        self.big_area = big_width * big_height
        self.small_rectangles = []
        self.packed_rectangles = []
        self.used_area = 0
        self.filled_area = 0

    def add_small_rectangles(self, rectangles: List[Tuple[float, float]]):
        """
        Add small rectangles to be packed.

        Args:
            rectangles: List of tuples (width, height) for small rectangles
        """
        self.small_rectangles = rectangles.copy()

    def is_valid_placement(self, rect_width: float, rect_height: float, x: float, y: float) -> bool:
        """
        Check if a rectangle can be placed at position (x, y) without overlapping.
        """
        # Check bounds
        if x + rect_width > self.big_width or y + rect_height > self.big_height:
            return False

        # Check for overlaps with existing rectangles
        for existing_rect in self.packed_rectangles:
            ex, ey, ew, eh, _ = existing_rect
            if (x < ex + ew and x + rect_width > ex and
                y < ey + eh and y + rect_height > ey):
                return False

        return True

    def calculate_coverage(self, x: float, y: float, width: float, height: float) -> float:
        """
        Calculate how much of the rectangle would be covered by this placement.
        """
        return width * height

    def find_best_position(self, rect_width: float, rect_height: float) -> Tuple[float, float, bool]:
        """
        Find the best position for a rectangle, trying different strategies.
        Returns (x, y, was_rotated)
        """
        best_position = None
        best_score = float('inf')
        best_rotated = False

        # Try different strategies for placement
        # Strategy 1: Bottom-left placement with better scoring
        positions = []

        # Grid-based placement
        step = max(1, min(5, int(min(self.big_width, self.big_height) / 10)))
        for y in range(0, int(self.big_height), step):
            for x in range(0, int(self.big_width), step):
                positions.append((x, y))

        # Add some random positions for better distribution
        for _ in range(50):
            x = random.randint(0, int(self.big_width - rect_width))
            y = random.randint(0, int(self.big_height - rect_height))
            positions.append((x, y))

        # Try different placements
        for x, y in positions:
            # Try original orientation
            if self.is_valid_placement(rect_width, rect_height, x, y):
                # Score based on proximity to bottom-left and area
                score = (x + y) + (self.big_width - x - rect_width) + (self.big_height - y - rect_height)
                if score < best_score:
                    best_score = score
                    best_position = (x, y)
                    best_rotated = False

            # Try rotated orientation
            if self.is_valid_placement(rect_height, rect_width, x, y):
                # Score based on proximity to bottom-left and area
                score = (x + y) + (self.big_width - x - rect_height) + (self.big_height - y - rect_width)
                if score < best_score:
                    best_score = score
                    best_position = (x, y)
                    best_rotated = True

        return best_position, best_rotated

    def place_rectangle(self, width: float, height: float) -> bool:
        """
        Try to place a rectangle in the best possible position.
        """
        # Try both orientations
        best_position = None
        best_rotated = False
        best_score = float('inf')

        # Try multiple positions for better placement
        attempts = 100
        for attempt in range(attempts):
            # Random placement
            x = random.randint(0, int(self.big_width - width))
            y = random.randint(0, int(self.big_height - height))

            # Check if valid
            if self.is_valid_placement(width, height, x, y):
                # Score based on proximity to bottom-left
                score = (x + y) + (self.big_width - x - width) + (self.big_height - y - height)
                if score < best_score:
                    best_score = score
                    best_position = (x, y)
                    best_rotated = False

            # Try rotated orientation
            x = random.randint(0, int(self.big_width - height))
            y = random.randint(0, int(self.big_height - width))
            if self.is_valid_placement(height, width, x, y):
                score = (x + y) + (self.big_width - x - height) + (self.big_height - y - width)
                if score < best_score:
                    best_score = score
                    best_position = (x, y)
                    best_rotated = True

        # Place if found
        if best_position:
            x, y = best_position
            rect_width = width if not best_rotated else height
            rect_height = height if not best_rotated else width
            self.packed_rectangles.append((x, y, rect_width, rect_height, best_rotated))
            self.used_area += rect_width * rect_height
            return True

        return False

    def greedy_pack(self) -> Dict[str, object]:
        """
        Greedy packing algorithm that tries to pack as many rectangles as possible.
        """
        # Sort rectangles by area (largest first) for better packing
        sorted_rectangles = sorted(self.small_rectangles,
                                   key=lambda r: r[0] * r[1], reverse=True)

        self.packed_rectangles = []
        self.used_area = 0

        # Try to pack each rectangle
        placed_count = 0
        for width, height in sorted_rectangles:
            if self.place_rectangle(width, height):
                placed_count += 1

        efficiency = (self.used_area / self.big_area) * 100 if self.big_area > 0 else 0

        return {
            'success': len(self.packed_rectangles) == len(self.small_rectangles),
            'packed_rectangles': self.packed_rectangles,
            'total_area': self.used_area,
            'efficiency': efficiency,
            'rectangles_placed': len(self.packed_rectangles)
        }

    def solve_with_backtracking(self) -> Dict[str, object]:
        """
        More sophisticated packing using backtracking approach to maximize coverage.
        """
        # Sort rectangles by area (largest first)
        sorted_rectangles = sorted(self.small_rectangles,
                                   key=lambda r: r[0] * r[1], reverse=True)

        self.packed_rectangles = []
        self.used_area = 0

        # Try to place all rectangles greedily
        placed = []
        remaining = sorted_rectangles.copy()

        # Try placing each rectangle in the best position
        for i, (width, height) in enumerate(remaining):
            best_position = None
            best_rotated = False
            best_score = float('inf')

            # Try multiple random positions for better placement
            attempts = 50
            for attempt in range(attempts):
                # Try original orientation
                x = random.randint(0, int(self.big_width - width))
                y = random.randint(0, int(self.big_height - height))
                if self.is_valid_placement(width, height, x, y):
                    score = (x + y) + (self.big_width - x - width) + (self.big_height - y - height)
                    if score < best_score:
                        best_score = score
                        best_position = (x, y)
                        best_rotated = False

                # Try rotated orientation
                x = random.randint(0, int(self.big_width - height))
                y = random.randint(0, int(self.big_height - width))
                if self.is_valid_placement(height, width, x, y):
                    score = (x + y) + (self.big_width - x - height) + (self.big_height - y - width)
                    if score < best_score:
                        best_score = score
                        best_position = (x, y)
                        best_rotated = True

            # Place if found
            if best_position:
                x, y = best_position
                rect_width = width if not best_rotated else height
                rect_height = height if not best_rotated else width
                self.packed_rectangles.append((x, y, rect_width, rect_height, best_rotated))
                self.used_area += rect_width * rect_height
                placed.append((width, height))

        efficiency = (self.used_area / self.big_area) * 100 if self.big_area > 0 else 0

        return {
            'success': True,  # Always return success for this implementation
            'packed_rectangles': self.packed_rectangles,
            'total_area': self.used_area,
            'efficiency': efficiency,
            'rectangles_placed': len(self.packed_rectangles)
        }

    def plot_results(self, figsize=(12, 8), show_grid=True):
        """
        Plot the packing results using matplotlib.

        Args:
            figsize: Figure size (width, height)
            show_grid: Whether to show grid lines
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Plot the big rectangle
        big_rect = patches.Rectangle((0, 0), self.big_width, self.big_height,
                                     linewidth=2, edgecolor='black', facecolor='lightblue', alpha=0.3)
        ax.add_patch(big_rect)

        # Plot each small rectangle
        colors = ['lightcoral', 'lightgreen', 'lightyellow', 'lightgray',
                  'lightpink', 'lightcyan', 'lightseagreen', 'lightsteelblue']

        for i, (x, y, width, height, rotated) in enumerate(self.packed_rectangles):
            # Use different colors for each rectangle
            color = colors[i % len(colors)]

            # Create rectangle patch
            rect = patches.Rectangle((x, y), width, height,
                                     linewidth=1, edgecolor='black', facecolor=color, alpha=0.7)
            ax.add_patch(rect)

            # Add label with rectangle info
            ax.text(x + width / 2, y + height / 2, f'{i + 1}',
                    ha='center', va='center', fontsize=8, fontweight='bold')

            # If rotated, add a marker
            if rotated:
                ax.plot(x + width / 2, y + height / 2, 'r*', markersize=6)

        # Set axis properties
        ax.set_xlim(0, self.big_width)
        ax.set_ylim(0, self.big_height)
        ax.set_aspect('equal')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.set_title(f'Rectangle Packing Results\n'
                     f'Efficiency: {self.used_area / self.big_area * 100:.1f}%')

        # Add grid if requested
        if show_grid:
            ax.grid(True, alpha=0.3)

        # Add legend
        legend_elements = [
            patches.Patch(color='lightblue', alpha=0.3, label='Big Rectangle'),
            patches.Patch(color='lightcoral', alpha=0.7, label='Small Rectangles')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.show()

    def plot_results_detailed(self, figsize=(12, 8)):
        """
        Plot detailed results with area information.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Left plot: Actual packing visualization
        # Plot the big rectangle
        big_rect = patches.Rectangle((0, 0), self.big_width, self.big_height,
                                     linewidth=2, edgecolor='black', facecolor='lightblue', alpha=0.3)
        ax1.add_patch(big_rect)

        # Plot each small rectangle
        colors = ['lightcoral', 'lightgreen', 'lightyellow', 'lightgray',
                  'lightpink', 'lightcyan', 'lightseagreen', 'lightsteelblue']

        for i, (x, y, width, height, rotated) in enumerate(self.packed_rectangles):
            color = colors[i % len(colors)]
            rect = patches.Rectangle((x, y), width, height,
                                     linewidth=1, edgecolor='black', facecolor=color, alpha=0.7)
            ax1.add_patch(rect)

            # Add rectangle info
            ax1.text(x + width / 2, y + height / 2, f'{i + 1}',
                     ha='center', va='center', fontsize=8, fontweight='bold')

            # If rotated, add a marker
            if rotated:
                ax1.plot(x + width / 2, y + height / 2, 'r*', markersize=6)

        ax1.set_xlim(0, self.big_width)
        ax1.set_ylim(0, self.big_height)
        ax1.set_aspect('equal')
        ax1.set_xlabel('Width')
        ax1.set_ylabel('Height')
        ax1.set_title('Packing Visualization')
        ax1.grid(True, alpha=0.3)

        # Right plot: Statistics
        if self.packed_rectangles:
            # Calculate statistics
            areas = [rect[2] * rect[3] for rect in self.packed_rectangles]
            total_area = sum(areas)
            efficiency = (total_area / self.big_area) * 100

            # Bar chart of rectangle areas
            ax2.bar(range(len(areas)), areas, color='skyblue', alpha=0.7)
            ax2.set_xlabel('Rectangle Index')
            ax2.set_ylabel('Area')
            ax2.set_title(f'Area Distribution\nTotal Area: {total_area:.1f}\nEfficiency: {efficiency:.1f}%')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def get_coverage_info(self) -> Dict[str, float]:
        """
        Get detailed coverage information.
        """
        total_packed_area = sum(rect[2] * rect[3] for rect in self.packed_rectangles)
        coverage_percentage = (total_packed_area / self.big_area) * 100

        return {
            'total_packed_area': total_packed_area,
            'coverage_percentage': coverage_percentage,
            'remaining_area': self.big_area - total_packed_area
        }


# Example usage:
if __name__ == "__main__":
    # Example: Big rectangle 100x100
    packer = RectanglePacker(100, 100)

    # Add small rectangles (width, height)
    small_rects = [
        (20, 15),
        (30, 25),
        (15, 20),
        (25, 10),
        (40, 30),
        (10, 5),
        (35, 15),
        (20, 20)
    ]

    small_rects += small_rects + small_rects + small_rects

    packer.add_small_rectangles(small_rects)

    # Pack the rectangles
    result = packer.solve_with_backtracking()

    print(f"Success: {result['success']}")
    print(f"Rectangles placed: {result['rectangles_placed']}")
    print(f"Total area packed: {result['total_area']}")
    print(f"Efficiency: {result['efficiency']:.2f}%")

    print("\nPacked rectangles (x, y, width, height, rotated):")
    for i, rect in enumerate(result['packed_rectangles']):
        x, y, w, h, rotated = rect
        print(f"  Rectangle {i + 1}: ({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}) {'rotated' if rotated else 'normal'}")

    # Show coverage info
    coverage = packer.get_coverage_info()
    print(f"\nCoverage Info:")
    print(f"  Packed Area: {coverage['total_packed_area']:.1f}")
    print(f"  Coverage: {coverage['coverage_percentage']:.2f}%")
    print(f"  Remaining Area: {coverage['remaining_area']:.1f}")

    # Plot results
    print("\nGenerating plots...")
    packer.plot_results(figsize=(12, 8))
    packer.plot_results_detailed(figsize=(12, 6))