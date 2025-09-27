#!/usr/bin/env python3
'''
# Team ID:          3300
# Theme:            Krishi Drone
# Author List:      Sarvesh Mishra, Abhinava Kalita, Sahil Lakhmani, Ratna Vernan
# Filename:         KD_3300_task1a.py
# Functions:        DronePlantDetector.__init__, DronePlantDetector.order_points, DronePlantDetector.get_contours,
#                   DronePlantDetector.detect_aruco_markers, DronePlantDetector.detect_yellow_plants,
#                   DronePlantDetector.save_results, main
# Global variables: ARUCO_TYPE, FG_HSV_MIN, FG_HSV_MAX
'''

import argparse
import sys
import cv2
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter

# ------------------ CONFIG (Global variables) ------------------ #
ARUCO_TYPE = cv2.aruco.DICT_4X4_100  # ArUCo dictionary type
FG_HSV_MIN = [0, 200, 50]            # Lower HSV for yellow masking
FG_HSV_MAX = [29, 255, 141]          # Upper HSV for yellow masking


class DronePlantDetector:
    '''
    Class to detect ArUCo markers, warp the image to a canonical frame,
    detect yellow (infected) plants within rectangular ROIs and save results.
    '''

    def __init__(self,
                 aruco_type: int = ARUCO_TYPE,
                 fg_hsv_min: List[int] = FG_HSV_MIN,
                 fg_hsv_max: List[int] = FG_HSV_MAX,
                 warp_size: int = 600):
        '''
        Purpose:
        ---
        Initialize the detector with ArUCo dictionary, HSV thresholds and output warp size.

        Input Arguments:
        ---
        `aruco_type` :  [ int ]
            OpenCV predefined aruco dictionary (e.g., cv2.aruco.DICT_4X4_100)

        `fg_hsv_min` :  [ list ]
            Lower HSV threshold for yellow masking [H, S, V]

        `fg_hsv_max` :  [ list ]
            Upper HSV threshold for yellow masking [H, S, V]

        `warp_size` :  [ int ]
            Size (pixels) of square warped output image

        Returns:
        ---
        None

        Example call:
        ---
        det = DronePlantDetector(aruco_type=cv2.aruco.DICT_4X4_100)
        '''
        # aruco dictionary id used for detection
        self.aruco_type = aruco_type

        # HSV thresholds
        # fg_hsv_min: Lower HSV boundary for masking yellow plants
        self.fg_hsv_min = np.array(fg_hsv_min, dtype=np.uint8)
        # fg_hsv_max: Upper HSV boundary for masking yellow plants
        self.fg_hsv_max = np.array(fg_hsv_max, dtype=np.uint8)

        # Size of warped output (width = height = warp_size)
        self.warp_size = warp_size

        # Prepare aruco dictionary object
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.aruco_type)

        # Detector parameters: prefer create() for compatibility
        try:
            # Some OpenCV builds provide DetectorParameters_create()
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        except AttributeError:
            # Fallback to constructor if create() is not available
            self.aruco_params = cv2.aruco.DetectorParameters()

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        '''
        Purpose:
        ---
        Sort 4 points into the order: top-left, top-right, bottom-right, bottom-left.

        Input Arguments:
        ---
        `pts` :  [ np.ndarray ]
            Array of shape (4,2) containing four 2D points (x,y).

        Returns:
        ---
        `rect` :  [ np.ndarray ]
            Array shape (4,2) of points ordered as [tl, tr, br, bl].

        Example call:
        ---
        ordered = detector.order_points(np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], dtype=np.float32))
        '''
        # Create output container
        rect = np.zeros((4, 2), dtype="float32")

        # Sum and diff are used to determine corners
        # s = x + y helps find tl (min) and br (max)
        s = pts.sum(axis=1)
        # diff = y - x; flatten to 1d for argmin/argmax
        diff = np.diff(pts, axis=1).flatten()

        rect[0] = pts[np.argmin(s)]   # top-left
        rect[2] = pts[np.argmax(s)]   # bottom-right
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left

        return rect

    def get_contours(self, img_canny: np.ndarray) -> List[List[int]]:
        '''
        Purpose:
        ---
        Find large rectangular contours (ROIs) in a Canny image and return bounding boxes.

        Input Arguments:
        ---
        `img_canny` :  [ np.ndarray ]
            Binary (edge) image from Canny edge detector.

        Returns:
        ---
        `rois` :  [ list of list ]
            List of ROIs in [x, y, w, h] format.

        Example call:
        ---
        rois = detector.get_contours(canny_image)
        '''
        # Find all external contours
        contours, _ = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        rois: List[List[int]] = []
        # Iterate through contours and filter by area and polygonal approximation
        for cnt in contours:
            contour_area = cv2.contourArea(cnt)
            # Filter out small shapes (noise)
            if contour_area > 10000:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
                # Only rectangles (4 corners) are considered as ROI candidates
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    # Ensure it's rectangular, not near-square
                    if abs(w - h) > 5:
                        rois.append([x, y, w, h])
        return rois

    def detect_aruco_markers(self, image: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        '''
        Purpose:
        ---
        Detect ArUCo markers in the input image. If exactly 4 markers are detected,
        compute a perspective transform to warp the image into a canonical square
        (self.warp_size x self.warp_size) using the top-left corner point of each marker.

        Input Arguments:
        ---
        `image` :  [ np.ndarray ]
            BGR input image in which ArUCo markers need to be detected.

        Returns:
        ---
        `output_image` :  [ np.ndarray ]
            Warped BGR image (self.warp_size x self.warp_size) if 4 markers detected,
            otherwise the original input image.

        `detected_marker_ids` :  [ list of int ]
            List of detected ArUCo marker IDs (may be empty if no markers detected).

        Example call:
        ---
        warped_image, ids = detector.detect_aruco_markers(image)
        '''
        # Detect markersa
        corners_list, ids_array, rejected = cv2.aruco.detectMarkers(image, self.aruco_dict, parameters=self.aruco_params)

        detected_marker_ids: List[int] = []
        if ids_array is not None and len(corners_list) > 0:
            # flatten ids to a normal list
            detected_marker_ids = ids_array.flatten().tolist()

        # If we don't have at least 4 detected marker corners, skip warping
        if corners_list is None or len(corners_list) < 4:
            # Debug information for user
            print(f"[detect_aruco_markers] Found {0 if corners_list is None else len(corners_list)} markers. Skipping warp.")
            return image, detected_marker_ids

        # Extract the top-left corner point from each detected marker (c[0][0]) -> (x,y)
        try:
            points = np.array([c[0][0] for c in corners_list], dtype=np.float32)
        except Exception as e:
            print("[detect_aruco_markers] Error extracting marker corner points:", e)
            return image, detected_marker_ids

        # If more than 4 markers exist, choose four representative outer points
        if points.shape[0] > 4:
            s = points.sum(axis=1)
            d = np.diff(points, axis=1).flatten()
            idxs = set([int(np.argmin(s)), int(np.argmax(s)), int(np.argmin(d)), int(np.argmax(d))])
            if len(idxs) < 4:
                # fallback: pick the first 4 entries
                idxs = set(range(4))
            points = points[list(idxs)]

        if points.shape[0] != 4:
            # Safety check
            print(f"[detect_aruco_markers] After selection, number of points != 4 (got {points.shape[0]}). Skipping warp.")
            return image, detected_marker_ids

        # Order points to [tl, tr, br, bl]
        src_pts = self.order_points(points)

        # Destination coordinates for perspective (canonical square)
        dst_pts = np.float32([[0, 0],
                              [self.warp_size - 1, 0],
                              [self.warp_size - 1, self.warp_size - 1],
                              [0, self.warp_size - 1]])
        # Compute transform and warp
        try:
            transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped_image = cv2.warpPerspective(image, transform_matrix, (self.warp_size, self.warp_size))
            return warped_image, detected_marker_ids
        except cv2.error as e:
            # If OpenCV fails to compute transform, return original image
            print("[detect_aruco_markers] OpenCV error during perspective transform:", e)
            return image, detected_marker_ids

    def detect_yellow_plants(self, image: np.ndarray) -> Tuple[Dict[str, List[str]], np.ndarray]:
        '''
        Purpose:
        ---
        Detect yellow (infected) plants in an image. The function first detects 4 rectangular ROIs
        using edge/contour detection and then analyzes each ROI split into 3 vertical sections,
        assigning plant IDs A-F and grouping detections into 'Block 1' or 'Block 2'
        depending on vertical position.

        Input Arguments:
        ---
        `image` :  [ np.ndarray ]
            BGR image (ideally warped to canonical frame) in which yellow plants need to be detected.

        Returns:
        ---
        `infected_plants` :  [ dict ]
            A dictionary of block-wise infected plant IDs. Example:
            {'Block 1': ['A','B'], 'Block 2': ['C']}

        `output_image` :  [ np.ndarray ]
            The input image (unchanged) returned for convenience.

        Example call:
        ---
        infected, out_img = detector.detect_yellow_plants(warped_image)
        '''
        # Attempt to detect exactly 4 ROIs; try different blur sizes if needed
        roi_candidates: List[List[int]] = []
        blur_kernel_size = 3  # must be odd
        while len(roi_candidates) != 4 and blur_kernel_size <= 11:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Apply Gaussian blur to reduce noise before edges
            img_blur = cv2.GaussianBlur(img_gray, (blur_kernel_size, blur_kernel_size), 1)
            img_canny = cv2.Canny(img_blur, 100, 100)
            roi_candidates = self.get_contours(img_canny)
            blur_kernel_size += 2  # try next odd kernel size

        infected_plants: Dict[str, List[str]] = {}

        if len(roi_candidates) != 4:
            print(f"[detect_yellow_plants] Expected 4 rectangles but found {len(roi_candidates)}. Will process available ROIs.")

        # Sort ROIs by y (top->bottom) then x (left->right) for consistent ordering
        roi_candidates = sorted(roi_candidates, key=lambda r: (r[1], r[0]))

        # We map index -> plant IDs in each rectangle. This mapping preserves your original mapping.
        # For indices: 0..3, alternate mapping between ['D','E','F'] and ['A','B','C'] per original code logic.
        for rect_index, (x, y, w, h) in enumerate(roi_candidates):
            # ids_for_rect: three plant IDs inside this rectangle (left-to-right)
            ids_for_rect = [['D', 'E', 'F'], ['A', 'B', 'C']][(rect_index) % 2]

            # Crop ROI region for color processing
            roi_img = image[y:y + h, x:x + w]
            if roi_img.size == 0:
                continue

            # Convert to HSV for color thresholding
            roi_hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

            # Mask out yellow pixels using defined HSV range
            mask = cv2.inRange(roi_hsv, self.fg_hsv_min, self.fg_hsv_max)

            # Split ROI width into 3 vertical slices (columns)
            slice_width = max(1, w // 3)
            for plant_index in range(3):
                start_col = plant_index * slice_width
                # last slice takes remaining columns to cover full width
                end_col = w if plant_index == 2 else (start_col + slice_width)
                # Safety clamp
                start_col = max(0, min(start_col, mask.shape[1] - 1))
                end_col = max(0, min(end_col, mask.shape[1]))

                if start_col >= end_col:
                    continue

                # Compute mean of mask in the slice; >0 indicates presence of yellow-ish pixels
                mean_mask_value = float(np.mean(mask[:, start_col:end_col]))

                # Threshold the mean to decide if a plant in this slice is infected
                if mean_mask_value > 20.0:
                    block_name = 'Block 1' if y > (self.warp_size / 2) else 'Block 2'
                    infected_plants.setdefault(block_name, []).append(ids_for_rect[plant_index])

        return infected_plants, image

    def save_results(self, detected_marker_ids: List[int], infected_plants: Dict[str, List[str]], out_filename: str = "detection.txt") -> None:
        '''
        Purpose:
        ---
        Save the detection results (ArUCo IDs and infected plants per block) to a text file.
        Output format matches grading requirements exactly:

        Detected marker IDs: [ID1, ID2, ID3, ID4]
        Infected plant in Block 1: P1<PlantID>
        Infected plant in Block 2: P2<PlantID>

        Input Arguments:
        ---
        `detected_marker_ids` :  [ list ]
            List of detected ArUCo marker IDs.

        `infected_plants` :  [ dict ]
            Dictionary of block-wise infected plant IDs (lists of IDs like ['A','B']).

        `out_filename` :  [ str ]
            Path/filename for the results text file.

        Returns:
        ---
        None

        Example call:
        ---
        detector.save_results([1,2,3,4], {"Block 1": ["A","A"], "Block 2": ["B"]}, "detection.txt")
        '''
        # Sort marker ids for deterministic output (but keep as list format)
        try:
            marker_list = list(detected_marker_ids)
        except Exception:
            marker_list = []

        # Helper to pick most common plant ID in a block; returns '' if none
        def most_common_plant_id(ids_list: List[str]) -> str:
            if not ids_list:
                return ""
            cnt = Counter(ids_list)
            most_common = cnt.most_common(1)[0][0]
            return most_common

        # Determine most infected plant per block (single plant ID or empty)
        block1_id = most_common_plant_id(infected_plants.get('Block 1', []))
        block2_id = most_common_plant_id(infected_plants.get('Block 2', []))

        # Format plant strings: if no detection, keep blank after colon
        plant1_str = f"P1{block1_id}" if block1_id else ""
        plant2_str = f"P2{block2_id}" if block2_id else ""

        # Write to file in required exact format (three lines)
        try:
            with open(out_filename, "w") as f:
                # Detected marker IDs: [50, 60, 55, 65]
                f.write(f"Detected marker IDs: {marker_list}\n")
                # Infected plant in Block 1: P1C
                f.write(f"Infected plant in Block 1: {plant1_str}\n")
                # Infected plant in Block 2: P2B
                f.write(f"Infected plant in Block 2: {plant2_str}\n")
            print(f"[save_results] Results saved to {out_filename}")
        except Exception as e:
            print("[save_results] Error writing results file:", e)


def main():
    '''
    Purpose:
    ---
    Entry point for script. Parses arguments, runs detection pipeline, and writes results.

    Input Arguments:
    ---
    None

    Returns:
    ---
    None

    Example call:
    ---
    python3 KD_3300_task1a_oop_submission.py -i input_image.jpg
    '''
    parser = argparse.ArgumentParser(description="Detect ArUCo markers and yellow plants in an image (Krishi Drone).")
    parser.add_argument("-i", "--image", required=True, help="path to input image containing ArUCo tags and plant regions")
    args = parser.parse_args()

    # Read image from disk
    image_path = args.image
    input_image = cv2.imread(image_path)
    if input_image is None:
        print("[main] Could not read image:", image_path)
        sys.exit(1)

    # Instantiate detector (uses global HSV/ARUCO defaults; can override here if needed)
    detector = DronePlantDetector()

    # Step 1: detect ArUCo and warp if possible
    warped_image, marker_ids = detector.detect_aruco_markers(input_image)

    # Step 2: detect yellow plants in warped image
    infected_dict, annotated_image = detector.detect_yellow_plants(warped_image)

    # Step 3: save results to a text file in the required submission format
    detector.save_results(marker_ids, infected_dict, out_filename="detection.txt")

if __name__ == "__main__":
    main()