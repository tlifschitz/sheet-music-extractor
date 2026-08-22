import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

def line_angle(line):
    x1, y1, x2, y2 = line[0]
    return np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

def line_distance(line1, line2):
    # Calculate average distance between midpoints
    x1_1, y1_1, x2_1, y2_1 = line1[0]
    x1_2, y1_2, x2_2, y2_2 = line2[0]
    
    mid1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
    mid2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
    
    return np.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)

def merge_two_lines(line1, line2):
    return line1
    # Merge two lines by extending endpoints
    x1_1, y1_1, x2_1, y2_1 = line1[0]
    x1_2, y1_2, x2_2, y2_2 = line2[0]
    
    # Find the extreme points
    all_points = [(x1_1, y1_1), (x2_1, y2_1), (x1_2, y1_2), (x2_2, y2_2)]
    
    # For horizontal lines, sort by x-coordinate
    angle = line_angle(line1)
    if abs(angle) < 45 or abs(angle) > 135:  # More horizontal
        all_points.sort(key=lambda p: p[0])
        return np.array([[all_points[0][0], all_points[0][1], 
                        all_points[-1][0], all_points[-1][1]]])
    else:  # More vertical
        all_points.sort(key=lambda p: p[1])
        return np.array([[all_points[0][0], all_points[0][1], 
                        all_points[-1][0], all_points[-1][1]]])


def merge_similar_lines(lines, distance_threshold=20, angle_threshold=5):
    """
    Merge lines that are close together and have similar angles
    
    Args:
        lines: Array of lines from HoughLinesP
        distance_threshold: Maximum distance between lines to merge (pixels)
        angle_threshold: Maximum angle difference to merge (degrees)
    
    Returns:
        merged_lines: Array of merged lines
    """
    if lines is None or len(lines) == 0:
        return lines
    
    merged_lines = []
    used = set()
    
    for i, line1 in enumerate(lines):
        if i in used:
            continue
            
        current_line = line1
        angle1 = line_angle(line1)
        
        # Find all lines to merge with this one
        to_merge = [i]
        
        for j, line2 in enumerate(lines[i+1:], i+1):
            if j in used:
                continue
                
            angle2 = line_angle(line2)
            
            # Check if angles are similar
            angle_diff = abs(angle1 - angle2)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            if angle_diff <= angle_threshold:
                # Check if lines are close enough
                if line_distance(current_line, line2) <= distance_threshold:
                    to_merge.append(j)
                    current_line = merge_two_lines(current_line, line2)
        
        # Mark all merged lines as used
        for idx in to_merge:
            used.add(idx)
        
        merged_lines.append(current_line)
    
    return np.array(merged_lines)

def extract_vertical_lines(img):
    """
    Extract only straight lines from sheet music using Hough Line Transform
    
    Args:
        image_path (str): Path to input sheet music image
        output_path (str): Path to save output image (optional)
    
    Returns:
        tuple: (original_image, lines_only_image, detected_lines)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Optional: Morphological operations to clean up edges
    kernel = np.ones((2,2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Apply Hough Line Transform
    # Parameters explanation:
    # - rho: distance resolution in pixels
    # - theta: angle resolution in radians
    # - threshold: minimum number of votes (intersections in Hough space)
    # - minLineLength: minimum line length
    # - maxLineGap: maximum allowed gap between line segments
    
    lines = cv2.HoughLinesP(
        edges,
        rho=1,                    # 1 pixel resolution
        theta=np.pi/180/10,          # 1 degree resolution
        threshold=120,            # minimum votes
        minLineLength=100,         # minimum line length
        maxLineGap=100            # maximum gap between line segments
    )
    
    # Create a blank image for drawing lines
    lines_img = np.zeros_like(img)
    
    # Create a white background version
    lines_img_white = np.ones_like(img) * 255
    vertical_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = line_angle(line)
            if abs(angle - 90) < 15 or abs(angle + 90) < 15:  # Vertical lines
                vertical_lines.append(line)
            
    vertical_lines = merge_similar_lines(vertical_lines)
    for line in vertical_lines:
        x1, y1, x2, y2 = line[0]
        # Draw on black background
        cv2.line(lines_img, (x1, y1), (x2, y2), (250,0,0), 2)
        # Draw on white background (black lines)
        cv2.line(lines_img_white, (x1, y1), (x2, y2), (0, 0, 0), 2)

    return img, lines_img, lines_img_white, vertical_lines

def display_results(original, lines_colored, lines_clean, lines_data):
    """
    Display the original image and extracted lines side by side
    """
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Sheet Music')
    plt.axis('off')
    
    # Lines on black background (colored)
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(lines_colored, cv2.COLOR_BGR2RGB))
    plt.title('Detected Lines (Colored by Direction)')
    plt.axis('off')
    
    # Lines on white background (clean)
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(lines_clean, cv2.COLOR_BGR2RGB))
    plt.title('Extracted Lines Only')
    plt.axis('off')
    
    # Statistics
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.8, f"Total lines detected: {len(lines_data) if lines_data is not None else 0}", 
             fontsize=12, transform=plt.gca().transAxes)
    
    if lines_data is not None:
        # Analyze line orientations
        horizontal_count = 0
        vertical_count = 0
        other_count = 0
        
        for line in lines_data:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            if abs(angle) < 15 or abs(angle) > 165:
                horizontal_count += 1
            elif abs(angle - 90) < 15 or abs(angle + 90) < 15:
                vertical_count += 1
            else:
                other_count += 1
        
        plt.text(0.1, 0.6, f"Horizontal lines: {horizontal_count}", 
                fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.4, f"Vertical lines: {vertical_count}", 
                fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.2, f"Other angles: {other_count}", 
                fontsize=12, transform=plt.gca().transAxes)
    
    plt.title('Line Detection Statistics')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to process sheet music image
    """
    # Staff image to analyse. Generate one with:
    #   python video2sheet.py <video> --dump-bars bars/
    input_path = sys.argv[1] if len(sys.argv) > 1 else "1.png"

    try:
        # Read the image
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Could not load image from {input_path}")
        

        # Extract lines
        original, lines_colored, lines_clean, lines_data = extract_vertical_lines(img)
        
        # Display results
        display_results(original, lines_colored, lines_clean, lines_data)
        
        print("Line extraction completed successfully!")
        print(f"Detected {len(lines_data) if lines_data is not None else 0} lines")
        
    except Exception as e:
        print(f"Error processing image: {e}")
        print("Make sure the image path is correct and the file exists.")


if __name__ == "__main__":
    main()