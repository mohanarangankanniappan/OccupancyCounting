"""
==============================================================
File: entry_exit_tracker.py
Description:
    This script performs person entry/exit counting inside a 
    user-defined polygon area in a video using YOLOv5 object 
    detection and simple tracking. It allows interactive 
    drawing of a polygon ("fence"), detects people frame by 
    frame, tracks them with centroid matching, and updates 
    entry/exit counts.
    
    Output:
        - Annotated video saved to disk
        - Console logs showing entry/exit/occupant counts

Dependencies:
    - OpenCV (cv2)
    - PyTorch
    - YOLOv5 (via torch.hub)

Author:
    Mohanarangan Kanniappan
Date:
    10/05/2025
==============================================================
"""

import warnings
import cv2
import torch
import numpy as np

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------
# Load YOLOv5 pretrained model
# -------------------------
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
print("YOLOv5 model loaded.")

# -------------------------
# Setup polygon drawing variables
# -------------------------
drawing_points = []  # list to store user-drawn polygon points
drawing_done = False  # flag to confirm when drawing is complete

def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback function for OpenCV window.

    Purpose:
        Captures points (x, y) on left mouse button click and 
        stores them to define a polygon.

    Args:
        event (int): OpenCV mouse event type.
        x (int): X-coordinate of mouse event.
        y (int): Y-coordinate of mouse event.
        flags (int): OpenCV event flags (unused).
        param: Additional parameters (unused).
    """
    global drawing_points, drawing_done
    if event == cv2.EVENT_LBUTTONDOWN and not drawing_done:
        drawing_points.append((x, y))

# -------------------------
# Video input and initial frame read
# -------------------------
video_path = 'VIRAT_S_010001_02_000195_000498.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error opening video file: {video_path}")
    exit()

ret, first_frame = cap.read()
if not ret:
    print("Failed to read video.")
    exit()

# -------------------------
# Open window for user to draw polygon area (fence)
# -------------------------
cv2.namedWindow("Draw Polygon (click points, press 'q' to confirm)")
cv2.setMouseCallback("Draw Polygon (click points, press 'q' to confirm)", mouse_callback)

while True:
    temp_frame = first_frame.copy()
    for point in drawing_points:
        cv2.circle(temp_frame, point, 5, (0, 0, 255), -1)
    if len(drawing_points) > 1:
        cv2.polylines(temp_frame, [np.array(drawing_points)], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.imshow("Draw Polygon (click points, press 'q' to confirm)", temp_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        drawing_done = True
        break
    elif key == 27:  # ESC key
        print("Exiting without starting.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

fence_points = np.array(drawing_points, np.int32)
cv2.destroyWindow("Draw Polygon (click points, press 'q' to confirm)")

# -------------------------
# Setup video writer for output
# -------------------------
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_video_path = "./data/final_entry_exit_confirmed.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 30, (frame_width, frame_height))

# Reset video to start
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# -------------------------
# Initialize tracking and counting variables
# -------------------------
tracks = {}  # dictionary of active tracked objects
entry_count = 0
exit_count = 0
occupants = 0
distance_threshold = 100  # max distance to match detection to existing track
max_disappeared = 100  # max frames before track is removed
frame_count = 0
next_track_id = 1  # unique ID for new tracks

print("Starting processing...")

def is_point_inside_polygon(point, polygon):
    """
    Check if a point is inside a polygon.

    Args:
        point (tuple): (x, y) coordinates.
        polygon (list or np.ndarray): List of (x, y) points forming the polygon.

    Returns:
        bool: True if point is inside polygon, False otherwise.
    """
    return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0

# -------------------------
# Main video processing loop
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    frame_draw = frame.copy()
    cv2.polylines(frame_draw, [fence_points], isClosed=True, color=(255, 0, 0), thickness=2)

    # Run YOLOv5 detection
    results = model(frame)
    detections = []
    for det in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 0:  # Class 0 = person
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            detections.append((cx, cy, x1, y1, x2, y2))

    # Update disappeared tracks
    for tid in list(tracks.keys()):
        tracks[tid]['disappeared'] += 1
        if tracks[tid]['disappeared'] > max_disappeared:
            del tracks[tid]

    detected_tids = []
    for cx, cy, x1, y1, x2, y2 in detections:
        matched_id = None
        min_dist = float('inf')

        # Match detections to existing tracks
        for tid, track in tracks.items():
            tx, ty = track['centroid']
            dist = np.linalg.norm([cx - tx, cy - ty])
            if dist < distance_threshold and dist < min_dist:
                min_dist = dist
                matched_id = tid

        if matched_id:
            # Update existing track
            track = tracks[matched_id]
            track['centroid'] = (cx, cy)
            prev_inside = track['inside']
            track['inside'] = is_point_inside_polygon((cx, cy), fence_points)
            track['disappeared'] = 0
            detected_tids.append(matched_id)

            if not prev_inside and track['inside'] and not track['entry_recorded']:
                entry_count += 1
                occupants += 1
                track['entry_recorded'] = True
                track['blink_counter'] = 10
                print(f"[ENTRY MADE] Frame: {frame_count} Track {matched_id} → Entry Count: {entry_count}, Occupants: {occupants}")
            elif prev_inside and not track['inside'] and not track['exit_recorded']:
                exit_count += 1
                occupants = max(occupants - 1, 0)
                track['exit_recorded'] = True
                track['blink_counter'] = 10
                print(f"[EXIT MADE] Frame: {frame_count} Track {matched_id} → Exit Count: {exit_count}, Occupants: {occupants}")
        else:
            # Create new track
            tid = next_track_id
            next_track_id += 1
            tracks[tid] = {
                'centroid': (cx, cy),
                'inside': is_point_inside_polygon((cx, cy), fence_points),
                'disappeared': 0,
                'entry_recorded': False,
                'exit_recorded': False,
                'tid': tid,
                'blink_counter': 0
            }
            detected_tids.append(tid)
            print(f"[NEW TRACK]  Frame: {frame_count} Track {tid} → Initially {'INSIDE' if tracks[tid]['inside'] else 'OUTSIDE'}")

    # Draw detections and track info
    for tid, track in tracks.items():
        x, y = int(track['centroid'][0]), int(track['centroid'][1])
        cv2.circle(frame_draw, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(frame_draw, f"ID {track['tid']}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        x1, y1, x2, y2 = 0, 0, 0, 0
        for det_cx, det_cy, det_x1, det_y1, det_x2, det_y2 in detections:
            if np.linalg.norm(np.array([x, y]) - np.array([det_cx, det_cy])) < distance_threshold:
                x1, y1, x2, y2 = det_x1, det_y1, det_x2, det_y2
                break

        # Handle blink effect on new entry/exit
        if track['blink_counter'] > 0:
            color = (0, 255, 255)  # yellow blink
            track['blink_counter'] -= 1
        else:
            color = (0, 0, 255)  # red box

        cv2.rectangle(frame_draw, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    # Draw stats on frame
    font_scale = 0.6
    thickness = 1
    cv2.putText(frame_draw, f"Entry: {entry_count}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    cv2.putText(frame_draw, f"Exit: {exit_count}", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    cv2.putText(frame_draw, f"Occupants: {occupants}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thickness, cv2.LINE_AA)
    cv2.putText(frame_draw, f"Frame: {frame_count}", (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thickness, cv2.LINE_AA)

    # Display and optionally exit on keypress
    cv2.imshow("Entry/Exit Tracker", frame_draw)
    key = cv2.waitKey(0)
    if key == ord('q') or key == 27:
        print("Exiting by user request.")
        break

    out.write(frame_draw)

# -------------------------
# Release resources
# -------------------------
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processed video saved to: {output_video_path}")
