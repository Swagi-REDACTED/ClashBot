# main_bot_windows.py
#
# DISCLAIMER: This script is for educational purposes only.
# Using a bot to automate gameplay is against Supercell's Terms of Service
# and can result in a permanent ban of your game account.
# Use this code at your own risk and only for learning about computer vision and automation.

import customtkinter as ctk
import cv2
import numpy as np
import time
import os
import random
import pyautogui
from PIL import Image, ImageGrab
import threading
import keyboard
import tkinter as tk
import queue
import subprocess
import sys
import traceback
import CoCscreenshot 

# --- FIX FOR PYINSTALLER: Helper function to handle resource paths ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- Defer heavy imports until after loading screen ---
easyocr = None
torch = None

# --- Main Configuration ---
DEBUG_MODE = True
LOOP_DELAY_SECONDS = 3
CONFIDENCE_THRESHOLD = 0.8
STOP_KEY = 'q'
FORCE_QUIT_KEY = 'alt+q'
CALIBRATION_IMAGE_PATH = "CoC_Diamond_Calibration.png"

# --- Template Path Configuration ---
TEMPLATE_PATHS = {
    'gold': resource_path('templates/gold_collector.png'),
    'elixir': resource_path('templates/elixir_collector.png'),
    'attack_button': resource_path('templates/attack_button.png'),
    'find_match_button': resource_path('templates/findamatch_button.png'),
    'surrender_button': resource_path('templates/surrender_button.png'),
    'end_button': resource_path('templates/end_button.png'),
    'return_home_button': resource_path('templates/returnhome_button.png'),
    'upgrade_button': resource_path('templates/upgrade_button.png'),
    'confirm_button': resource_path('templates/confirm_button.png'),
    'exit_button': resource_path('templates/exit_button.png'),
    'no_builders': resource_path('templates/no_builders.png'),
    'gem': resource_path('templates/gem.png'),
    'next': resource_path('templates/next_opponent.png'),
    'victory': resource_path('templates/victory_button.png'),
    'okay': resource_path('templates/okay_button.png'),
}

# --- Resource & Attack Configuration ---
RESOURCE_BAR_CONFIG = {
    'gold': {'roi': (2115, 45, 300, 25)},
    'elixir': {'roi': (2115, 157, 300, 25)}
}
# Adjusted ROIs to be wider to capture larger numbers
ENEMY_RESOURCE_CONFIG = {
    'gold':   {'roi': (106, 160, 300, 55)},
    'elixir': {'roi': (105, 225, 300, 55)}
}
ATTACK_CONFIG = {
    'troop_slots': [
        (470, 1305), (635, 1305), (800, 1305), (965, 1305), (1130, 1305), (1295, 1305),
        (1460, 1305), (1790, 1305), (1955, 1305), (2120, 1305), (2285, 1305) # Adjusted for 11 slots
    ],
}

# --- Shared State for GUI and Bot Logic ---
shared_state = {
    'is_bot_loop_enabled': False,
    'skip_upgrades': False,
    'action_queue': queue.Queue(),
    'app_running': True,
    'status_message': "Idle. Toggle loop or select an action.",
    'min_gold': 100000, 
    'min_elixir': 100000,
    'troops_to_use': 11,
    'successful_attacks': 0,
    'loading_complete': threading.Event(),
    'loading_progress': 0,
    'stop_action_flag': threading.Event(),
    # New Attack State
    'attack_style': 'Dragon Spam', # Default attack style
    'safe_attacking_enabled': False,
    'attack_count_for_safe_mode': 0,
    'next_safe_attack_in': random.randint(3, 7)
}
ocr_reader = None

# --- Bot Logic Functions ---

def capture_screen_windows(filename="screen.png"):
    """Captures the screen and returns it as a OpenCV image."""
    try:
        screenshot_pil = ImageGrab.grab()
        screenshot_np = np.array(screenshot_pil)
        screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
        if DEBUG_MODE and filename:
            cv2.imwrite(filename, screenshot_cv)
        return filename, screenshot_cv
    except Exception as e:
        print(f"[-] ERROR: Failed to capture screen: {e}")
        return None, None

def simulate_click_windows(x, y, duration=0.01, clicks=1, interval=0.01):
    """Simulates a mouse click at the given coordinates."""
    if not shared_state['app_running']: return
    pyautogui.moveTo(x, y, duration=duration)
    pyautogui.click(clicks=clicks, interval=interval)

def find_button_location(screenshot_cv, template_name, custom_threshold=None):
    """Finds the location of a template image on the screen."""
    template_path = TEMPLATE_PATHS.get(template_name)
    if not os.path.exists(template_path): 
        print(f"[!] Template not found: {template_path}")
        return None
    template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template_img is None: return None
    screenshot_gray = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(screenshot_gray, template_img, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    threshold = custom_threshold if custom_threshold is not None else CONFIDENCE_THRESHOLD
    if max_val >= threshold:
        h, w = template_img.shape
        return (max_loc[0], max_loc[1], w, h)
    return None

def find_and_click_button(screenshot_cv, template_name, custom_threshold=None):
    """Finds and clicks a button based on a template image."""
    location = find_button_location(screenshot_cv, template_name, custom_threshold)
    if location:
        x, y, w, h = location
        center_x = x + w // 2
        center_y = y + h // 2
        shared_state['status_message'] = f"Clicking '{template_name}' button."
        print(f"[+] Found '{template_name}' button. Clicking it.")
        simulate_click_windows(center_x, center_y, clicks=1)
        return True
    return False

def initialize_heavy_resources():
    """Initializes EasyOCR."""
    global ocr_reader, torch, easyocr
    try:
        shared_state['loading_progress'] = 10
        print("-> Loading heavy libraries (EasyOCR)...")
        import easyocr
        import torch
        shared_state['loading_progress'] = 50
        print("--> Initializing EasyOCR reader...")
        try:
            gpu_available = torch.cuda.is_available()
            print(f"---> Attempting to initialize OCR with GPU (CUDA): {gpu_available}")
            ocr_reader = easyocr.Reader(['en'], gpu=gpu_available)
            shared_state['loading_progress'] = 90
            print(f"[+] EasyOCR reader initialized successfully on {'GPU' if gpu_available else 'CPU'}!")
        except Exception as e:
            print(f"[!] OCR initialization with GPU failed: {e}. Defaulting to CPU.")
            ocr_reader = easyocr.Reader(['en'], gpu=False)
            shared_state['loading_progress'] = 90
            print("[+] EasyOCR reader initialized successfully on CPU.")
    except Exception as e:
        print(f"[-] CRITICAL ERROR: Could not initialize heavy resources: {e}")
        traceback.print_exc()
        shared_state['app_running'] = False
    finally:
        shared_state['loading_progress'] = 100
        shared_state['loading_complete'].set()

def read_resource_value(screenshot_cv, resource_type):
    """
    Reads the resource value from the screen using OCR with robust preprocessing.
    """
    if ocr_reader is None: return 0
    try:
        config = ENEMY_RESOURCE_CONFIG[resource_type]
        x, y, w, h = config['roi']
        roi_color = screenshot_cv[y:y+h, x:x+w]

        # 1. Convert to grayscale
        gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
        
        # 2. Apply a high-contrast binary threshold to isolate the bright text.
        # This is more robust than specific color masking.
        # We set any pixel brighter than 170 to white, and the rest to black.
        # This works because both gold and elixir text are very bright.
        _, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

        # 3. Dilation to make text bolder and fill in gaps from thresholding
        kernel = np.ones((2, 2), np.uint8)
        processed_roi = cv2.dilate(thresh, kernel, iterations=1)
        
        if DEBUG_MODE:
            cv2.imwrite(f"debug_ocr_processed_{resource_type}.png", processed_roi)

        # Use EasyOCR on the preprocessed image
        result = ocr_reader.readtext(processed_roi, detail=0, paragraph=False, allowlist='0123456789')
        
        if not result:
            print(f"[!] OCR found no numbers for {resource_type}.")
            return 0
        
        # Join all found parts and filter for digits
        numeric_string = "".join(filter(str.isdigit, "".join(result)))
        
        if not numeric_string:
            print(f"[!] OCR result for {resource_type} contained no digits: {result}")
            return 0
            
        return int(numeric_string)
    except Exception as e:
        print(f"[-] Error during OCR for {resource_type}: {e}")
        traceback.print_exc()
        return 0

def find_multiple_locations(screenshot_cv, template_name, custom_threshold=None):
    template_path = TEMPLATE_PATHS.get(template_name)
    if not template_path or not os.path.exists(template_path): return []
    template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template_img is None: return []
    screenshot_gray = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(screenshot_gray, template_img, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= (custom_threshold or CONFIDENCE_THRESHOLD))
    h, w = template_img.shape
    rects = [[int(pt[0]), int(pt[1]), w, h] for pt in zip(*loc[::-1])]
    return cv2.groupRectangles(rects, 1, 0.5)[0] if rects else []

def find_and_click_collectors(screenshot_cv, template_name):
    print(f"-> Searching for {template_name} collectors...")
    locations = find_multiple_locations(screenshot_cv, template_name, custom_threshold=0.7)
    if len(locations) > 0:
        print(f"[+] Found {len(locations)} {template_name} collector(s).")
        for (x, y, w, h) in locations:
            simulate_click_windows(x + w // 2, y + h // 2, clicks=1)
            time.sleep(0.3)
        shared_state['status_message'] = f"Collected from {len(locations)} {template_name}(s)."
        return True
    return False

def run_calibration_script():
    """Positions the camera and then calls the generate_calibration_image function."""
    print("-> Starting calibration sequence...")
    try:
        # Add the delay and camera positioning call that was in the original script's main block.
        print("--> SWITCH TO YOUR GAME WINDOW. Calibrating in 3 seconds...")
        time.sleep(3)
        
        # Explicitly call the camera positioning function first.
        CoCscreenshot.position_camera()
        
        # Now generate the image.
        CoCscreenshot.generate_calibration_image(output_filename=CALIBRATION_IMAGE_PATH)
        
        print("[+] Calibration script finished successfully.")
        return True
    except Exception as e:
        print(f"[-] ERROR: An unexpected error occurred during calibration: {e}")
        traceback.print_exc()
        return False

def get_deployable_coordinates():
    """
    Runs the calibration script and then loads the resulting image to get deployable coordinates.
    """
    print("--- Getting deployable zones from calibration image ---")
    
    if not run_calibration_script():
        return None

    try:
        deployable_mask = cv2.imread(CALIBRATION_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
        if deployable_mask is None:
            print(f"[-] ERROR: Failed to load calibration image at '{CALIBRATION_IMAGE_PATH}'.")
            return None
    except Exception as e:
        print(f"[-] ERROR loading calibration image: {e}")
        return None

    y_coords, x_coords = np.where(deployable_mask > 0)
    
    if len(y_coords) == 0:
        print(f"[!] No deployable area found in '{CALIBRATION_IMAGE_PATH}'.")
        return None
    
    print(f"[+] Found {len(y_coords)} safe pixels to deploy on from calibration image.")
    return list(zip(x_coords, y_coords))

def is_troop_available(screenshot_cv, troop_pos):
    """
    Checks if a troop, spell, or hero is available for deployment.
    - Heroes are checked by looking for a green health bar.
    - Troops/Spells are checked by color saturation (not grayed out).
    """
    try:
        x, y = troop_pos
        # Define a 40x40 ROI around the troop icon center
        roi_bgr = screenshot_cv[y-20:y+20, x-20:x+20]
        if roi_bgr.shape[0] < 40 or roi_bgr.shape[1] < 40: return False

        # 1. Hero Check: Look for the green health bar at the top of the icon
        health_bar_roi = roi_bgr[0:5, 5:35] # A small strip at the top of the ROI
        hsv_health_bar = cv2.cvtColor(health_bar_roi, cv2.COLOR_BGR2HSV)
        # HSV range for bright green
        lower_green = np.array([35, 150, 150])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv_health_bar, lower_green, upper_green)
        if np.sum(mask) > 250: # If enough green pixels are found
            if DEBUG_MODE: print(f"-> Troop at {troop_pos} is an ACTIVE HERO.")
            return True

        # 2. Troop/Spell Check: Check for color saturation
        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mean_saturation = np.mean(roi_hsv[:, :, 1])
        
        # A higher saturation threshold indicates a colored (available) troop/spell
        if mean_saturation > 50:
            if DEBUG_MODE: print(f"-> Troop at {troop_pos} is AVAILABLE (Saturation: {mean_saturation:.1f}).")
            return True
        else:
            if DEBUG_MODE: print(f"-> Troop at {troop_pos} is EXPENDED (Saturation: {mean_saturation:.1f}).")
            return False
            
    except (IndexError, ValueError) as e:
        print(f"[-] Error checking troop availability at {troop_pos}: {e}")
        return False # Assume unavailable if there's an error

def handle_post_battle_popups():
    """Checks for and clicks the 'Okay' button for 6 seconds after a battle."""
    print("-> Checking for post-battle pop-ups (e.g., Star Bonus)...")
    start_time = time.time()
    while time.time() - start_time < 6:
        if shared_state['stop_action_flag'].is_set(): return
        _, screen_cv = capture_screen_windows(filename=None)
        if screen_cv is not None:
            if find_and_click_button(screen_cv, 'okay'):
                print("[+] Found and clicked 'Okay' button.")
                time.sleep(1) # Wait a moment for the dialog to disappear
                return
        time.sleep(1)
    print("-> No 'Okay' button found after 6 seconds. Continuing...")

def perform_full_attack(deployable_coords):
    """
    Main attack function that uses the selected attack style from the GUI.
    """
    print("\n--- INITIATING ATTACK SEQUENCE ---")
    
    # Determine attack style for this run
    attack_style = shared_state['attack_style']
    if shared_state['safe_attacking_enabled']:
        shared_state['attack_count_for_safe_mode'] += 1
        if shared_state['attack_count_for_safe_mode'] >= shared_state['next_safe_attack_in']:
            print("[!] SAFE ATTACKING triggered. Overriding style to 'Slow and Steady'.")
            attack_style = 'Slow and Steady'
            shared_state['attack_count_for_safe_mode'] = 0
            shared_state['next_safe_attack_in'] = random.randint(3, 7)

    print(f"-> Using Attack Style: {attack_style}")
    shared_state['status_message'] = f"Attacking: {attack_style}"

    num_troops_to_use = shared_state.get('troops_to_use', len(ATTACK_CONFIG['troop_slots']))
    active_troop_slots = ATTACK_CONFIG['troop_slots'][:num_troops_to_use]

    # --- Deployment Logic based on Style ---
    deployment_functions = {
        'Dragon Spam': deploy_dragon_spam,
        'Slow and Steady': deploy_slow_and_steady,
        "Spam 'em Down": deploy_spam_em_down
    }
    
    deployment_function = deployment_functions.get(attack_style, deploy_slow_and_steady)
    deployment_function(active_troop_slots, deployable_coords)

    # --- Post-Deployment Monitoring ---
    if shared_state['stop_action_flag'].is_set():
        print("-> Attack halted prematurely by user.")
        return

    print("\n-> Deployment phase finished. Monitoring for battle end.")
    shared_state['status_message'] = "Deployment finished. Waiting..."
    end_time = time.time() + 180
    attack_outcome_determined = False
    while time.time() < end_time and shared_state['app_running'] and not shared_state['stop_action_flag'].is_set():
        _, screen_cv = capture_screen_windows(filename=None)
        if screen_cv is None:
            time.sleep(2)
            continue
        if find_button_location(screen_cv, 'return_home_button'):
            if find_button_location(screen_cv, 'victory', custom_threshold=0.65):
                print("[+] Battle won! Logging successful attack.")
                shared_state['successful_attacks'] += 1
            else:
                 print("[-] 'Return Home' button found without victory text. Not logging as a win.")
            find_and_click_button(screen_cv, 'return_home_button')
            attack_outcome_determined = True
            break
        print("-> Waiting for battle to end...")
        time.sleep(3)
    
    if not attack_outcome_determined and not shared_state['stop_action_flag'].is_set():
        print("[!] Battle timeout reached. Attempting to return home.")
        _, screen_cv = capture_screen_windows()
        if screen_cv is not None:
            if find_button_location(screen_cv, 'return_home_button') and find_button_location(screen_cv, 'victory'):
                 print("[+] Found 'Return Home' with victory text after timeout. Logging as a win.")
                 shared_state['successful_attacks'] += 1
            find_and_click_button(screen_cv, 'return_home_button')
            
    if not shared_state['stop_action_flag'].is_set():
        handle_post_battle_popups()

    print("\n[+] Attack sequence complete.")

# --- NEW: Attack Style Helper Functions ---

def deploy_dragon_spam(active_troop_slots, deployable_coords):
    print("--> Executing Dragon Spam logic...")
    # Deploy all troops EXCEPT the first slot (single clicks)
    for i, troop_pos in reversed(list(enumerate(active_troop_slots))):
        if shared_state['stop_action_flag'].is_set(): return
        if i == 0: continue # Skip the first slot (dragons)
        
        _, screen_cv = capture_screen_windows(None)
        if is_troop_available(screen_cv, troop_pos):
            print(f"--> Spawning support troops (Slot #{i+1})")
            simulate_click_windows(troop_pos[0], troop_pos[1], clicks=1)
            time.sleep(0.1)
            # Spam until it's grayed out
            while is_troop_available(capture_screen_windows(None)[1], troop_pos):
                if shared_state['stop_action_flag'].is_set():
                    print("-> Stop signal received during deployment. Halting.")
                    return
                if not deployable_coords: break
                deploy_pos = random.choice(deployable_coords)
                # Corrected: Use random.uniform for randomized single clicks
                simulate_click_windows(deploy_pos[0], deploy_pos[1], clicks=1, duration=random.uniform(0.01, 0.02))
                time.sleep(random.uniform(0.008, 0.015))
    
    # Now, spam the first troop slot (dragons) with double clicks
    if shared_state['stop_action_flag'].is_set(): return
    print("--> Spawning main force (Slot #1) with double clicks...")
    dragon_pos = active_troop_slots[0]
    _, screen_cv = capture_screen_windows(None)
    if is_troop_available(screen_cv, dragon_pos):
        simulate_click_windows(dragon_pos[0], dragon_pos[1], clicks=1)
        time.sleep(0.1)
        while is_troop_available(capture_screen_windows(None)[1], dragon_pos):
            if shared_state['stop_action_flag'].is_set():
                print("-> Stop signal received during deployment. Halting.")
                return
            if not deployable_coords: break
            deploy_pos = random.choice(deployable_coords)
            # Corrected: Use random.uniform for randomized interval
            simulate_click_windows(deploy_pos[0], deploy_pos[1], clicks=random.randint(2, 3), interval=random.uniform(0.008, 0.012))
            time.sleep(random.uniform(0.015, 0.025))

def deploy_slow_and_steady(active_troop_slots, deployable_coords):
    print("--> Executing Slow and Steady logic...")
    all_troops_expended = False
    while not all_troops_expended and not shared_state['stop_action_flag'].is_set():
        all_troops_expended = True
        for i, troop_pos in enumerate(active_troop_slots):
            if shared_state['stop_action_flag'].is_set(): return
            _, screen_cv = capture_screen_windows(None)
            if is_troop_available(screen_cv, troop_pos):
                all_troops_expended = False
                print(f"--> Deploying 10 units from Slot #{i+1} (Slowly)...")
                simulate_click_windows(troop_pos[0], troop_pos[1], clicks=1)
                time.sleep(0.2)
                for _ in range(10):
                    if shared_state['stop_action_flag'].is_set():
                        print("-> Stop signal received during deployment. Halting.")
                        return
                    if not deployable_coords: break
                    if not is_troop_available(capture_screen_windows(None)[1], troop_pos): break
                    deploy_pos = random.choice(deployable_coords)
                    # Corrected: Use random.randint and random.uniform for randomized clicks and duration
                    simulate_click_windows(deploy_pos[0], deploy_pos[1], clicks=random.randint(1, 3), duration=random.uniform(0.01, 0.03))
                    time.sleep(random.uniform(0.15, 0.20))
                break # Move to the next troop in the next main loop iteration
        if all_troops_expended:
            print("--> All troops appear to be expended.")

def deploy_spam_em_down(active_troop_slots, deployable_coords):
    print("--> Executing Spam 'em Down logic...")
    all_troops_expended = False
    while not all_troops_expended and not shared_state['stop_action_flag'].is_set():
        all_troops_expended = True
        for i, troop_pos in enumerate(active_troop_slots):
            if shared_state['stop_action_flag'].is_set(): return
            _, screen_cv = capture_screen_windows(None)
            if is_troop_available(screen_cv, troop_pos):
                all_troops_expended = False
                print(f"--> Deploying 10 units from Slot #{i+1} (Fast Spam)...")
                simulate_click_windows(troop_pos[0], troop_pos[1], clicks=1)
                time.sleep(0.1)
                for _ in range(5): # 5 clicks, 2 troops each
                    if shared_state['stop_action_flag'].is_set():
                        print("-> Stop signal received during deployment. Halting.")
                        return
                    if not deployable_coords: break
                    if not is_troop_available(capture_screen_windows(None)[1], troop_pos): break
                    deploy_pos = random.choice(deployable_coords)
                    # Corrected: Use random.uniform for randomized interval
                    simulate_click_windows(deploy_pos[0], deploy_pos[1], clicks=random.randint(2, 3), interval=random.uniform(0.008, 0.012))
                    time.sleep(random.uniform(0.015, 0.025))
                break
        if all_troops_expended:
            print("--> All troops appear to be expended.")

def get_resource_percentage(screenshot_cv, resource_type):
    try:
        config = RESOURCE_BAR_CONFIG[resource_type]
        x, y, w, h = config['roi']
        bar_roi = screenshot_cv[y:y+h, x:x+w]
        hsv_bar = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2HSV)
        if resource_type == 'gold':
            lower_color, upper_color = np.array([15, 100, 100]), np.array([35, 255, 255])
        elif resource_type == 'elixir':
            lower_color, upper_color = np.array([130, 50, 50]), np.array([170, 255, 255])
        else:
            return 0
        mask = cv2.inRange(hsv_bar, lower_color, upper_color)
        kernel = np.ones((3,3), np.uint8)
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return 0
        largest_contour = max(contours, key=cv2.contourArea)
        _x, _y, contour_w, _h = cv2.boundingRect(largest_contour)
        return (contour_w / w) * 100
    except Exception as e:
        print(f"[-] Error getting resource percentage for {resource_type}: {e}")
        return 0

def generate_randomized_search_points(center_x, center_y, max_dist, step_size):
    points = []
    for r in range(step_size, max_dist, step_size):
        for i in range(12):
            angle = 2 * np.pi * i / 12
            x = int(center_x + r * np.cos(angle))
            y = int(center_y + r * np.sin(angle))
            points.append((x, y))
    random.shuffle(points)
    return points

def perform_upgrade_sequence():
    print("\n--- INITIATING UPGRADE SEQUENCE ---")
    shared_state['status_message'] = "Searching for upgrades..."
    _, screen_cv = capture_screen_windows()
    if screen_cv is not None:
        builder_hut_loc = find_button_location(screen_cv, 'no_builders')
        if builder_hut_loc:
            print("-> Found 'No Builders' icon. Checking number...")
            x, y, w, h = builder_hut_loc
            roi = screen_cv[y-5:y+h+5, x-5:x+w+5]
            if ocr_reader:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                result = ocr_reader.readtext(roi_thresh, detail=0, paragraph=True, allowlist='0')
                if '0' in "".join(result):
                    print("[!] No builders available (0 found). Skipping upgrade sequence.")
                    shared_state['status_message'] = "No builders available."
                    return True 
    screen_w, screen_h = pyautogui.size()
    pyautogui.moveTo(screen_w // 2, screen_h // 2)
    pyautogui.scroll(-5000); time.sleep(1)
    search_points = generate_randomized_search_points(1232, 745, 912, step_size=75)
    upgraded_something_in_sequence = False
    for point_x, point_y in search_points:
        if not shared_state['app_running'] or shared_state['stop_action_flag'].is_set(): break
        simulate_click_windows(point_x, point_y, clicks=1); time.sleep(0.5)
        _, screen_cv = capture_screen_windows()
        if screen_cv is None: continue
        if find_and_click_button(screen_cv, 'upgrade_button'):
            time.sleep(1.5)
            _, confirm_screen_cv = capture_screen_windows()
            if confirm_screen_cv is None: continue
            if not find_and_click_button(confirm_screen_cv, 'confirm_button'):
                find_and_click_button(confirm_screen_cv, 'exit_button')
                continue
            time.sleep(1)
            _, gem_check_cv = capture_screen_windows()
            if gem_check_cv is None: continue
            if find_button_location(gem_check_cv, 'gem'):
                print("[-] Gem cost detected. Stopping upgrade search.")
                shared_state['status_message'] = "Gem cost. Stopping upgrades."
                find_and_click_button(gem_check_cv, 'exit_button'); time.sleep(1)
                break
            else:
                print("[+] Confirmed upgrade without gems. One upgrade started.")
                shared_state['status_message'] = "Upgrade successful! Stopping search."
                upgraded_something_in_sequence = True
                time.sleep(1)
                _, after_upgrade_screen = capture_screen_windows()
                if after_upgrade_screen is not None:
                     if find_and_click_button(after_upgrade_screen, 'exit_button'):
                         print("-> Closed menu after successful upgrade.")
                break 
    if not upgraded_something_in_sequence:
        print("--- Upgrade sequence finished. No upgrades found or started. ---")
    else:
        print("--- Upgrade sequence finished. ---")
    return True

def run_failsafe_recovery():
    print("[-] ULTIMATE FAILSAFE TRIGGERED [-]")
    shared_state['status_message'] = "Failsafe: Attempting to recover..."
    for i in range(5):
        if shared_state['stop_action_flag'].is_set(): break
        _, screen_cv = capture_screen_windows()
        if screen_cv is not None:
            if find_and_click_button(screen_cv, 'exit_button', custom_threshold=0.8):
                time.sleep(1.5)
            else:
                print("[+] Failsafe: No exit button found, assuming recovery is complete.")
                shared_state['status_message'] = "Failsafe recovery complete."
                return
    print("[!] Failsafe: Reached max exit clicks. Recovery may not be complete.")

def run_collect_sequence():
    if not shared_state['loading_complete'].is_set(): 
        shared_state['status_message'] = "Bot is still loading..."
        return True
    shared_state['status_message'] = "Collecting resources..."
    print("\n--- Running Collect Sequence ---")
    screen_w, screen_h = pyautogui.size()
    pyautogui.moveTo(screen_w // 2, screen_h // 2)
    pyautogui.scroll(-5000)
    time.sleep(1)
    _, screenshot_cv = capture_screen_windows()
    if screenshot_cv is None: 
        return True
    find_and_click_collectors(screenshot_cv, 'gold')
    time.sleep(0.5)
    find_and_click_collectors(screenshot_cv, 'elixir')
    print("--- Collect Sequence Finished ---")
    return True

def run_upgrade_check_sequence():
    if not shared_state['loading_complete'].is_set(): 
        shared_state['status_message'] = "Bot is still loading..."
        return True
    shared_state['status_message'] = "Checking resources for upgrade..."
    print("\n--- Running Upgrade Check Sequence ---")
    _, screenshot_cv = capture_screen_windows()
    if screenshot_cv is None: return True
    gold_level = get_resource_percentage(screenshot_cv, 'gold')
    elixir_level = get_resource_percentage(screenshot_cv, 'elixir')
    print(f"-> Resource Levels: Gold {gold_level:.1f}%, Elixir {elixir_level:.1f}%")
    if gold_level > 95 or elixir_level > 95:
        return perform_upgrade_sequence()
    else:
        shared_state['status_message'] = "Resources not high enough to upgrade."
        print("-> Resources not high enough to warrant an upgrade search.")
    print("--- Upgrade Check Finished ---")
    return True

def run_attack_sequence():
    if not shared_state['loading_complete'].is_set(): 
        shared_state['status_message'] = "Bot is still loading..."
        return False
    shared_state['status_message'] = "Starting attack sequence..."
    print("\n--- Running Attack Sequence ---")
    _, screenshot_cv = capture_screen_windows()
    if screenshot_cv is None: return False
    if not find_and_click_button(screenshot_cv, 'attack_button', custom_threshold=0.7):
        print("-> Could not find attack button on home screen.")
        shared_state['status_message'] = "Could not find attack button."
        run_failsafe_recovery()
        return False
    time.sleep(1.5)
    if shared_state['stop_action_flag'].is_set(): return True
    _, after_attack_click_cv = capture_screen_windows()
    if not find_and_click_button(after_attack_click_cv, 'find_match_button'):
        print("-> Could not find 'Find a Match' button.")
        run_failsafe_recovery()
        return False
    shared_state['status_message'] = "Searching for opponent..."
    print("\n--- Searching for a worthy opponent... ---")
    start_time = time.time()
    max_search_time_seconds = 300
    while time.time() - start_time < max_search_time_seconds:
        if not shared_state['app_running'] or shared_state['stop_action_flag'].is_set(): return True
        battle_screen_cv = None
        for _ in range(15):
            if shared_state['stop_action_flag'].is_set(): return True
            _, temp_screen = capture_screen_windows(filename=None)
            if temp_screen is not None and find_button_location(temp_screen, 'end_button'):
                battle_screen_cv = temp_screen
                break
            time.sleep(1)
        if battle_screen_cv is None:
            print("[!] Timed out waiting for battle screen to load.")
            run_failsafe_recovery()
            return False
        print("-> Opponent found. Reading resource values.")
        found_gold = read_resource_value(battle_screen_cv, 'gold')
        found_elixir = read_resource_value(battle_screen_cv, 'elixir')
        required_gold = shared_state['min_gold']
        required_elixir = shared_state['min_elixir']
        print(f"-> Loot available: {found_gold:,} Gold, {found_elixir:,} Elixir")
        print(f"-> Loot required: {required_gold:,} Gold, {required_elixir:,} Elixir")
        if found_gold >= required_gold and found_elixir >= required_elixir:
            print("[+] Worthy opponent found! Analyzing for attack.")
            shared_state['status_message'] = "Opponent found! Analyzing..."
            deployable_coords = get_deployable_coordinates()
            if deployable_coords:
                perform_full_attack(deployable_coords)
            else:
                print("[!] No deployable zones found after analysis. Surrendering.")
                shared_state['status_message'] = "No deploy zones. Surrendering."
                find_and_click_button(battle_screen_cv, 'end_button')
                time.sleep(1.5)
                _, surrender_screen = capture_screen_windows()
                if surrender_screen is not None:
                    find_and_click_button(surrender_screen, 'end_button')
            return True
        else:
            print("-> Loot too low. Clicking 'Next'...")
            shared_state['status_message'] = "Loot too low, skipping..."
            if not find_and_click_button(battle_screen_cv, 'next', custom_threshold=0.85):
                print("[!] Could not find the 'Next' button. Aborting.")
                run_failsafe_recovery()
                return False
            time.sleep(2)
    print("[!] Max search time reached. Aborting attack sequence.")
    shared_state['status_message'] = "Search timed out."
    run_failsafe_recovery()
    return False

# --- MODERN GUI Class (using customtkinter) ---
class LoadingWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Loading...")
        self.root.overrideredirect(True)
        width, height = 400, 100
        screen_width, screen_height = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        x, y = (screen_width / 2) - (width / 2), (screen_height / 2) - (height / 2)
        self.root.geometry(f'{width}x{height}+{int(x)}+{int(y)}')
        self.label = ctk.CTkLabel(self.root, text="Initializing...", font=("Segoe UI", 12))
        self.label.pack(pady=(20, 10))
        self.progress_bar = ctk.CTkProgressBar(self.root, width=300)
        self.progress_bar.set(0)
        self.progress_bar.pack()
    def update_status_and_progress(self, text, progress):
        self.label.configure(text=text)
        self.progress_bar.set(progress / 100)
        self.root.update()
    def finish(self):
        self.root.after(1500, self.root.destroy)

class BotGUI(threading.Thread):
    def __init__(self, state):
        super().__init__()
        self.daemon = True
        self.state = state
        self.root = None
        self.attack_counter_label = None
        self.gold_slider_var = None
        self.elixir_slider_var = None
        self.troops_slider_var = None
        self.skip_upgrades_var = None
        self.attack_style_var = None
        self.safe_attacking_var = None

    def toggle_loop(self):
        if not self.state['loading_complete'].is_set(): return
        self.state['is_bot_loop_enabled'] = not self.state['is_bot_loop_enabled']
        status = "ON" if self.state['is_bot_loop_enabled'] else "OFF"
        self.state['status_message'] = f"Loop is {status}."
        if self.state['is_bot_loop_enabled']:
            self.state['stop_action_flag'].clear()
    def trigger_action(self, action_name):
        if not self.state['loading_complete'].is_set(): return
        self.state['stop_action_flag'].clear()
        self.state['action_queue'].put(action_name)
    def stop_bot_processes(self):
        print("--- STOP CURRENT PROCESS triggered ---")
        self.state['is_bot_loop_enabled'] = False
        if not self.state['action_queue'].empty():
            with self.state['action_queue'].mutex:
                self.state['action_queue'].queue.clear()
        self.state['stop_action_flag'].set()
        def reset_state_after_stop():
            global current_sequence_index
            current_sequence_index = 0
            self.state['status_message'] = "PROCESS STOPPED. Ready for new command."
            if self.attack_counter_label:
                self.update_attack_counter_display()
        threading.Thread(target=reset_state_after_stop, daemon=True).start()
    def close_app(self):
        print("--- FORCE QUIT triggered ---")
        self.state['app_running'] = False
        self.state['stop_action_flag'].set()
        self.state['is_bot_loop_enabled'] = False
        os._exit(0)
    def update_attack_counter_display(self):
        if self.attack_counter_label and self.attack_counter_label.winfo_exists():
            self.attack_counter_label.configure(text=f"Successful Attacks: {self.state['successful_attacks']}")
    def update_status_label(self, label):
        try:
            if self.root and self.root.winfo_exists():
                if self.state['loading_complete'].is_set():
                    label.configure(text=f"Status: {self.state['status_message']}")
                else:
                    label.configure(text="Status: Initializing, please wait...")
                self.update_attack_counter_display()
                self.root.after(500, self.update_status_label, label)
        except (tk.TclError, RuntimeError): pass
    def on_slider_change(self, value, resource_type, label_widget):
        val = int(float(value))
        self.state[f'{resource_type}'] = val
        try:
            if label_widget.winfo_exists():
                if resource_type == 'troops_to_use':
                    label_widget.configure(text=f"Troops to Use: {val}")
                else:
                    label_widget.delete(0, "end")
                    label_widget.insert(0, f"{val:,}")
        except tk.TclError: pass
    def update_from_entry(self, event, resource_type):
        widget = event.widget
        new_value_str = widget.get().replace(',', '')
        try:
            new_value = int(new_value_str)
            if new_value < 0: new_value = 0
            if new_value > 1000000: new_value = 1000000
            slider_var = self.gold_slider_var if resource_type == 'min_gold' else self.elixir_slider_var
            self.state[resource_type] = new_value
            slider_var.set(new_value)
            widget.delete(0, "end")
            widget.insert(0, f"{new_value:,}")
        except (ValueError, tk.TclError):
            current_value = self.state[resource_type]
            widget.delete(0, "end")
            widget.insert(0, f"{current_value:,}")
    def run(self):
        self.state['loading_complete'].wait() 
        if not self.state['app_running']: return 
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.root = ctk.CTk()
        self.root.title("CoC Bot Control Panel")
        self.root.geometry("420x680") # Increased height for new options
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)
        self.root.resizable(False, False)
        
        main_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        main_frame.pack(padx=15, pady=15, fill="both", expand=True)
        
        header_label = ctk.CTkLabel(main_frame, text="Clash of Clans Bot", font=("Segoe UI", 20, "bold"))
        header_label.pack(pady=(0, 10))
        
        status_label = ctk.CTkLabel(main_frame, text="Status: Ready.", wraplength=370, font=("Segoe UI", 12))
        status_label.pack(pady=(0, 15), fill="x")
        self.update_status_label(status_label)
        
        # --- Main Controls ---
        controls_frame = ctk.CTkFrame(main_frame)
        controls_frame.pack(fill="x", pady=5)
        loop_toggle = ctk.CTkCheckBox(controls_frame, text="Toggle Bot Loop", command=self.toggle_loop)
        loop_toggle.pack(pady=10, anchor="w", padx=15)
        self.skip_upgrades_var = tk.BooleanVar(value=self.state['skip_upgrades'])
        skip_upgrades_toggle = ctk.CTkCheckBox(controls_frame, text="Skip Upgrades in Loop", variable=self.skip_upgrades_var,
                                               command=lambda: self.state.update({'skip_upgrades': self.skip_upgrades_var.get()}))
        skip_upgrades_toggle.pack(pady=(0, 10), anchor="w", padx=15)

        # --- Attack Style Controls ---
        attack_style_frame = ctk.CTkFrame(main_frame)
        attack_style_frame.pack(pady=10, fill="x")
        ctk.CTkLabel(attack_style_frame, text="Attack Strategy", font=("Segoe UI", 12, "bold")).pack(pady=(5,10))
        self.attack_style_var = ctk.StringVar(value=self.state['attack_style'])
        attack_styles = ["Dragon Spam", "Slow and Steady", "Spam 'em Down"]
        for style in attack_styles:
            ctk.CTkRadioButton(attack_style_frame, text=style, variable=self.attack_style_var, value=style,
                               command=lambda s=style: self.state.update({'attack_style': s})).pack(anchor="w", padx=15, pady=2)
        
        self.safe_attacking_var = tk.BooleanVar(value=self.state['safe_attacking_enabled'])
        safe_attack_toggle = ctk.CTkCheckBox(attack_style_frame, text="Enable Safe Attacking", variable=self.safe_attacking_var,
                                               command=lambda: self.state.update({'safe_attacking_enabled': self.safe_attacking_var.get()}))
        safe_attack_toggle.pack(pady=(10, 10), anchor="w", padx=15)

        # --- Loot & Troop Sliders ---
        sliders_frame = ctk.CTkFrame(main_frame)
        sliders_frame.pack(pady=10, fill="x")
        sliders_frame.grid_columnconfigure(1, weight=1)
        loot_label = ctk.CTkLabel(sliders_frame, text="Loot & Troop Settings", font=("Segoe UI", 12, "bold"))
        loot_label.grid(row=0, column=0, columnspan=3, pady=(5, 10), sticky="n")
        self.gold_slider_var = tk.IntVar(value=self.state['min_gold'])
        ctk.CTkLabel(sliders_frame, text="Min Gold:").grid(row=1, column=0, sticky='w', padx=(10, 5))
        gold_slider = ctk.CTkSlider(sliders_frame, from_=0, to=1000000, variable=self.gold_slider_var)
        gold_slider.grid(row=1, column=1, sticky='ew', padx=5)
        gold_value_entry = ctk.CTkEntry(sliders_frame, width=80)
        gold_value_entry.grid(row=1, column=2, sticky='e', padx=(5, 10))
        gold_slider.configure(command=lambda v: self.on_slider_change(v, 'min_gold', gold_value_entry))
        gold_value_entry.bind("<Return>", lambda e: self.update_from_entry(e, 'min_gold'))
        gold_value_entry.bind("<FocusOut>", lambda e: self.update_from_entry(e, 'min_gold'))
        self.on_slider_change(self.state['min_gold'], 'min_gold', gold_value_entry)
        self.elixir_slider_var = tk.IntVar(value=self.state['min_elixir'])
        ctk.CTkLabel(sliders_frame, text="Min Elixir:").grid(row=2, column=0, sticky='w', padx=(10, 5), pady=(10,0))
        elixir_slider = ctk.CTkSlider(sliders_frame, from_=0, to=1000000, variable=self.elixir_slider_var)
        elixir_slider.grid(row=2, column=1, sticky='ew', padx=5, pady=(10,0))
        elixir_value_entry = ctk.CTkEntry(sliders_frame, width=80)
        elixir_value_entry.grid(row=2, column=2, sticky='e', padx=(5, 10), pady=(10,0))
        elixir_slider.configure(command=lambda v: self.on_slider_change(v, 'min_elixir', elixir_value_entry))
        elixir_value_entry.bind("<Return>", lambda e: self.update_from_entry(e, 'min_elixir'))
        elixir_value_entry.bind("<FocusOut>", lambda e: self.update_from_entry(e, 'min_elixir'))
        self.on_slider_change(self.state['min_elixir'], 'min_elixir', elixir_value_entry)
        self.troops_slider_var = tk.IntVar(value=self.state['troops_to_use'])
        troops_label = ctk.CTkLabel(sliders_frame, text=f"Troops to Use: {self.state['troops_to_use']}")
        troops_label.grid(row=3, column=0, sticky='w', padx=(10, 5), pady=(15,10))
        troops_slider = ctk.CTkSlider(sliders_frame, from_=1, to=11, number_of_steps=10, variable=self.troops_slider_var)
        troops_slider.grid(row=3, column=1, columnspan=2, sticky='ew', padx=(5,10), pady=(15,10))
        troops_slider.configure(command=lambda v: self.on_slider_change(v, 'troops_to_use', troops_label))
        
        # --- Action Buttons & Counters ---
        self.attack_counter_label = ctk.CTkLabel(main_frame, text="Successful Attacks: 0", font=("Segoe UI", 14, "bold"))
        self.attack_counter_label.pack(pady=(10, 5), fill="x")
        action_frame = ctk.CTkFrame(main_frame)
        action_frame.pack(pady=5, fill="x")
        action_frame.grid_columnconfigure((0, 1, 2), weight=1)
        ctk.CTkButton(action_frame, text="Collect", command=lambda: self.trigger_action('collect')).grid(row=0, column=0, padx=5, pady=10, sticky="ew")
        ctk.CTkButton(action_frame, text="Upgrade", command=lambda: self.trigger_action('upgrade')).grid(row=0, column=1, padx=5, pady=10, sticky="ew")
        ctk.CTkButton(action_frame, text="Attack", command=lambda: self.trigger_action('attack')).grid(row=0, column=2, padx=5, pady=10, sticky="ew")
        
        stop_button = ctk.CTkButton(main_frame, text=f"STOP CURRENT PROCESS ({STOP_KEY.upper()})", 
                                    command=self.stop_bot_processes, 
                                    fg_color="#D32F2F", hover_color="#B71C1C")
        stop_button.pack(pady=(15,0), fill="x")
        self.root.mainloop()

# --- Main Bot Logic Loop ---
current_sequence_index = 0

def main_bot_logic(state):
    state['loading_complete'].wait()
    if not state['app_running']: return

    global current_sequence_index
    sequence_order = ['collect', 'upgrade', 'attack']

    while state['app_running']:
        try:
            if state['stop_action_flag'].is_set():
                time.sleep(0.1)
                continue

            action_to_run = None
            is_manual_action = False
            
            try:
                action_to_run = state['action_queue'].get_nowait()
                is_manual_action = True
                print(f"Running manual action: {action_to_run}")
                if action_to_run in sequence_order:
                    current_sequence_index = (sequence_order.index(action_to_run) + 1) % len(sequence_order)
            except queue.Empty:
                if state['is_bot_loop_enabled']:
                    action_to_run = sequence_order[current_sequence_index]
                else:
                    time.sleep(1)
                    continue

            if state.get('is_bot_loop_enabled') and state.get('skip_upgrades') and action_to_run == 'upgrade':
                print("-> 'Skip Upgrades' is enabled. Skipping upgrade sequence.")
                state['status_message'] = "Skipping upgrade sequence."
                current_sequence_index = (current_sequence_index + 1) % len(sequence_order)
                time.sleep(LOOP_DELAY_SECONDS)
                continue

            if action_to_run:
                success = False
                action_map = {
                    'collect': run_collect_sequence,
                    'upgrade': run_upgrade_check_sequence,
                    'attack': run_attack_sequence
                }
                if action_to_run in action_map:
                    success = action_map[action_to_run]()
                
                if not is_manual_action and not success and not state['stop_action_flag'].is_set():
                    print(f"[!] Sequence '{action_to_run}' failed. Attempting recovery before retry.")
                    state['status_message'] = f"'{action_to_run}' failed. Recovering..."
                    run_failsafe_recovery()
                elif is_manual_action and not success:
                    print(f"[!] Manual action '{action_to_run}' failed.")
                    state['status_message'] = f"Manual '{action_to_run}' failed."
                else:
                    if not is_manual_action and not state['stop_action_flag'].is_set():
                        current_sequence_index = (current_sequence_index + 1) % len(sequence_order)
                    elif not state['is_bot_loop_enabled']:
                        state['status_message'] = "Manual action complete. Idling."

            time.sleep(LOOP_DELAY_SECONDS)

        except Exception as e:
            print(f"[-] FATAL ERROR in main loop: {e}")
            traceback.print_exc()
            state['is_bot_loop_enabled'] = False
            state['status_message'] = "FATAL ERROR. Loop stopped."
            time.sleep(5)
            
    print("Bot logic loop has stopped.")

if __name__ == "__main__":
    try:
        from multiprocessing import freeze_support
        freeze_support()
    except:
        pass

    print("--- Clash of Clans Bot ---")
    
    ctk.set_appearance_mode("dark")
    
    loading_root = ctk.CTk()
    loading_ui = LoadingWindow(loading_root)
    loading_root.update()

    def loading_task():
        loading_ui.update_status_and_progress("Initializing OCR...", 10)
        initialize_heavy_resources()
        if shared_state['app_running']:
            loading_ui.update_status_and_progress("Done!", 100)
            time.sleep(1.5)
        loading_root.quit()

    loading_thread = threading.Thread(target=loading_task, daemon=True)
    loading_thread.start()
    
    loading_root.mainloop()
    loading_root.destroy()

    if shared_state['app_running']:
        gui = BotGUI(shared_state)
        
        keyboard.add_hotkey(STOP_KEY, gui.stop_bot_processes)
        keyboard.add_hotkey(FORCE_QUIT_KEY, gui.close_app)

        print(f"\nStarting bot... Press '{STOP_KEY.upper()}' to stop process, '{FORCE_QUIT_KEY.upper()}' to quit.")
        
        bot_thread = threading.Thread(target=main_bot_logic, args=(shared_state,), daemon=True)
        bot_thread.start()
        gui.run()
        
    print("\nBot has been stopped")
