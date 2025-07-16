# CoCscreenshot.py
#
# A utility to capture the screen, filter the checkerboard pattern, and then
# use a complex computer vision algorithm to reconstruct a "true" diamond overlay
# for perfect visual calibration.

import cv2
import numpy as np
from PIL import ImageGrab
import time
import os
import pyautogui
import traceback

def filter_checkerboard_pattern(gray_image):
    """
    Analyzes a grayscale image and isolates the checkerboard pattern by
    identifying its two distinct grayscale value ranges.

    Args:
        gray_image (numpy.ndarray): The input grayscale image.

    Returns:
        numpy.ndarray: A binary mask where white pixels represent the checkerboard pattern.
    """
    print("-> Analyzing grayscale image to find checkerboard pattern...")
    
    # These two specific grayscale ranges correspond to the
    # light and dark squares of the deployable checkerboard grid.
    lower_light_square = 145
    upper_light_square = 178
    lower_dark_square = 175
    upper_dark_square = 202

    # Create a mask for each component of the checkerboard
    light_square_mask = cv2.inRange(gray_image, lower_light_square, upper_light_square)
    dark_square_mask = cv2.inRange(gray_image, lower_dark_square, upper_dark_square)
    
    # Combine the two masks to get the complete checkerboard pattern
    checkerboard_mask = cv2.bitwise_or(light_square_mask, dark_square_mask)
    
    print("-> Successfully filtered for two-tone grayscale pattern.")
    
    return checkerboard_mask

def position_camera():
    """
    Automatically zooms out and centers the camera using pyautogui mouse controls.
    """
    print("-> Positioning camera...")
    screen_w, screen_h = pyautogui.size()
    center_x, center_y = screen_w // 2, screen_h // 2

    # 1. Zoom out fully
    print("--> Zooming out...")
    pyautogui.moveTo(center_x, center_y)
    pyautogui.scroll(-20000) # Use a very large scroll value to ensure it's fully zoomed out
    time.sleep(1)

    # 2. Drag to position camera to the top-right
    print("--> Dragging to top-right corner...")
    pyautogui.moveTo(1800, 435)
    pyautogui.dragTo(1000, 1120, duration=0.7, button='left')
    time.sleep(0.5)

    # 3. Drag to center the view
    print("--> Centering camera...")
    pyautogui.moveTo(1200, 1080) # Adjusted start point for a more reliable drag
    pyautogui.dragTo(1300, 880, duration=0.5, button='left')
    time.sleep(1)
    print("-> Camera positioned.")


def generate_calibration_image(output_filename="checkerboard_calibration.png"):
    """
    Captures the screen, filters the grid, calculates a median diamond, and then
    processes the interior to create a smooth, solid base mask.
    """
    print("Capturing screen...")
    try:
        screenshot_pil = ImageGrab.grab()
        screenshot_np = np.array(screenshot_pil)
        screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
        screenshot_gray = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2GRAY)
        
        height, width = screenshot_gray.shape
        
        # Get the initial noisy, dotted checkerboard mask
        filtered_mask = filter_checkerboard_pattern(screenshot_gray)

        # --- Step 1: Calculate the Median Diamond ---
        print("-> Calculating median diamond for accurate boundary...")
        # Define the hardcoded "Expected" diamond based on screen percentages
        y_offset, x_offset, expansion_offset = -65, 35, 172
        top_point_expected = (int(width * 0.5) + x_offset, int(height * 0.08) + y_offset - expansion_offset)
        left_point_expected = (int(width * 0.12) + x_offset - expansion_offset, int(height * 0.5) + y_offset)
        bottom_point_expected = (int(width * 0.5) + x_offset, int(height * 0.95) + y_offset + expansion_offset)
        right_point_expected = (int(width * 0.85) + x_offset + expansion_offset, int(height * 0.5) + y_offset)
        expected_diamond_pts = np.array([top_point_expected, right_point_expected, bottom_point_expected, left_point_expected], np.int32)
        
        # Calculate the "True" diamond based on the center of mass of the detected grid
        roi_mask = np.zeros(screenshot_gray.shape, dtype=np.uint8)
        cv2.fillPoly(roi_mask, [expected_diamond_pts], 255)
        grid_points_in_roi = cv2.bitwise_and(filtered_mask, filtered_mask, mask=roi_mask)
        M = cv2.moments(grid_points_in_roi)
        
        true_diamond_pts = expected_diamond_pts # Fallback
        if M["m00"] != 0:
            true_center_x, true_center_y = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            expected_center_x, expected_center_y = int(np.mean([p[0] for p in expected_diamond_pts])), int(np.mean([p[1] for p in expected_diamond_pts]))
            offset_x, offset_y = true_center_x - expected_center_x, true_center_y - expected_center_y
            true_diamond_pts = expected_diamond_pts + [offset_x, offset_y]

        # The final, stable diamond is the average of the expected and true diamonds
        median_diamond_pts = np.int32((expected_diamond_pts + true_diamond_pts) / 2)

        # --- Step 2: Process the "Black Noise" within the Median Diamond ---
        print("-> Isolating and processing the base layout (black noise)...")

        # Create a mask for the median diamond area
        diamond_area_mask = np.zeros(screenshot_gray.shape, dtype=np.uint8)
        cv2.fillPoly(diamond_area_mask, [median_diamond_pts], 255)
        
        # Isolate the original filtered grid, but only within the diamond
        grid_inside_diamond = cv2.bitwise_and(filtered_mask, diamond_area_mask)
        
        # Invert the image so the base layout (gaps) becomes white
        base_noise_mask = cv2.bitwise_not(grid_inside_diamond)
        # We must also mask this inverted image to the diamond area to remove the outer white areas
        base_noise_mask = cv2.bitwise_and(base_noise_mask, diamond_area_mask)

        # --- Step 3: Expand and Smooth the Base Shape ---
        print("-> Expanding and smoothing base into concave shape...")
        
        # Dilate the base noise to connect nearby components
        dilate_kernel = np.ones((25, 25), np.uint8)
        dilated_base = cv2.dilate(base_noise_mask, dilate_kernel, iterations=1)
        
        # Apply a heavy Gaussian blur to create smooth, concave curves
        blurred_base = cv2.GaussianBlur(dilated_base, (51, 51), 0)
        
        # Threshold the blurred image to get a solid shape
        _, solid_base_mask = cv2.threshold(blurred_base, 127, 255, cv2.THRESH_BINARY)

        # --- Step 4: Reconstruct the Final Image ---
        print("-> Reconstructing final mask...")
        
        # Start with a solid white version of the median diamond
        final_mask = np.zeros_like(screenshot_gray)
        cv2.fillPoly(final_mask, [median_diamond_pts], 255)
        
        # Punch out the smoothed base shape by drawing it in black
        final_mask[solid_base_mask == 255] = 0
        
        # --- Step 4.5: Add Outer Border to Median Diamond ---
        print("-> Adding 10% white border to median diamond...")
        # Calculate the center of the median diamond
        center_x = np.mean(median_diamond_pts[:, 0])
        center_y = np.mean(median_diamond_pts[:, 1])
        center = np.array([center_x, center_y])
        
        # Create a scaled-down inner diamond (90% of original size for a 10% border)
        inner_diamond_pts = (0.90 * (median_diamond_pts - center) + center).astype(np.int32)
        
        # Create a mask for the border
        border_mask = np.zeros_like(screenshot_gray)
        # Draw the full median diamond in white
        cv2.fillPoly(border_mask, [median_diamond_pts], 255)
        # Punch out the center with the smaller diamond, leaving a border
        cv2.fillPoly(border_mask, [inner_diamond_pts], 0)
        
        # Add the white border to the final mask
        final_mask = cv2.bitwise_or(final_mask, border_mask)

        # --- Step 5: Final Cleanup ---
        LowerCutoffToBlack_y_line = 1210
        print("-> Applying final cleanup and blackout zones...")
        cv2.rectangle(final_mask,
                      (0, LowerCutoffToBlack_y_line),
                      (width, height),
                      (0, 0, 0),
                      thickness=cv2.FILLED)

        # Save the final, clean image
        cv2.imwrite(output_filename, final_mask)
        
        print(f"✅ Successfully created and saved calibration image as '{output_filename}'")
        print(f"Full path: {os.path.abspath(output_filename)}")

    except Exception as e:
        print(f"❌ ERROR: Failed to capture or process the screen: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    OUTPUT_FILENAME = "CoC_Diamond_Calibration.png"
    
    print("Starting in 3 seconds... Switch to your game window.")
    time.sleep(3)
    
    position_camera()
    
    generate_calibration_image(output_filename=OUTPUT_FILENAME)
