import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import subprocess
from math import ceil
import os

video_path = "/Volumes/repos/sheet-music-extractor/videos/Teddy Swims - Some Things I'll Never Know.mp4"
video_path = "/Volumes/repos/sheet-music-extractor/videos/Coldplay - Yellow - Piano Tutorial with Sheet Music.mp4"
video_path = "/Volumes/repos/sheet-music-extractor/videos/Twenty One Pilots - The Run And Go.mp4"
video_path = "/Volumes/repos/sheet-music-extractor/videos/Coldplay - A Sky Full of Stars - Accurate Piano Tutorial with Sheet Music.mp4"

show = True

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))

kernel_size = min(int(round(width*0.005)), 3)
th1 = int(round(width*0.25))
mid = int(round(width*0.5))
th2 = int(round(width*0.85))
bar_position_saturation_threshold = 10
pentagram_detection_bar_width = 20
min_corner_brightness = 230

print("Frames per second:", fps)
print(f"Resolution: {width} x {height}")
frame_skip = 1;int(round(fps))/6


bar_positions_y = []
bar_positions_x = []
state_1_x = []
state_2_x = []
state_3_x = []
state_1_y = []
state_2_y = []
state_3_y = []

bar_position = None
blue_bar_min_x = 99999
blue_bar_max_x = -99999
transition_delay_frames = 5
bars = []
frame_idx = 0
boundary_y = None
state = 0
lines_data = []



def get_bar_position(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Sum the saturation along vertical axis to find column with highest intensity
    kernel = np.ones(kernel_size) / kernel_size
    smoothed_sum = np.convolve(np.mean(hsv[:, :, 1], axis=0), kernel, mode='same')

    if np.max(smoothed_sum) > bar_position_saturation_threshold:
        x_pos = np.argmax(smoothed_sum)
    else:
        x_pos = None
    return x_pos

def detect_pentagram_boundary(frame):
    """
    Detects the pentagram boundary in the given frame.
    Returns the y-coordinate of the boundary or None if not found.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Use only the left pixels of the image to find the pentagram boundary
    gray = gray[:,:pentagram_detection_bar_width]
    # Calculate the average brightness of each row
    vertical_brightness = np.mean(gray, axis=1)
    # Find the index of the row with the maximum brightness
    average_corner_brightness = np.mean(vertical_brightness[:pentagram_detection_bar_width])
    
    diffs = np.diff(vertical_brightness)

    k = pentagram_detection_bar_width
    # Location of sharpest change, ignoring first/last k pixels
    brightness_drop_y_coordinate = np.argmax(np.abs(diffs[k:-k])) + k  
    
    return average_corner_brightness, brightness_drop_y_coordinate

while True:

    ret, frame = cap.read()
    frame_idx += 1

    if not (ret):
        break


    if state == 0:
        average_corner_brightness, brightness_drop_y_coordinate = detect_pentagram_boundary(frame)
        if average_corner_brightness > min_corner_brightness:
            boundary_y = brightness_drop_y_coordinate
            state = 1
            print(f"Found pentagram boundary {boundary_y}")
    elif state in [1, 2, 3]:

        ################################## Detect pentagram boundary #################################
        # Convert to grayscale
        # Use only the left pixels of the image to find the pentagram boundary

        average_corner_brightness, brightness_drop_y_coordinate = detect_pentagram_boundary(frame)

        current_boundary_y = brightness_drop_y_coordinate

        if average_corner_brightness < min_corner_brightness:
            boundary_y = None
            state = 0
            print(f"Lost pentagram boundary {average_corner_brightness} {min_corner_brightness}, resetting state")
            continue
        elif abs(current_boundary_y - boundary_y) > 10:
            boundary_y = current_boundary_y
            print(f"Boundary changed significantly, resetting state from {boundary_y} to {current_boundary_y}")

        ################################# Detect bar position #################################
        staff = frame[:boundary_y,:]
        bar_position = get_bar_position(staff)

        if bar_position is not None:
            bar_positions_y.append(bar_position)
            bar_positions_x.append(frame_idx)

            if state == 1:
                if bar_position > th1:
                    print(f"{frame_idx} state 1 to 2")
                    print(boundary_y, mid)
                    cut_right = staff[:,mid:].copy()
                    state = 2
                    state_2_y.append(bar_position)
                    state_2_x.append(frame_idx)
            if state == 2:
                if bar_position > th2:
                    print(f"{frame_idx} state 2 to 3")
                    cut_left = staff[:,:mid].copy()

                    merged = np.hstack((cut_left, cut_right))
                    merged = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)
                    
                    bars.append(merged)
                    #cv2.imwrite(f"{len(bars)}.png", merged)

                    state = 3
                    state_3_y.append(bar_position)
                    state_3_x.append(frame_idx)
            if state == 3:
                if bar_position < th1:
                    staff = frame[:boundary_y,:]
                    #lines_data = extract_vertical_lines(staff)
                    print(f"{frame_idx} state 3 to 1")
                    state = 1
                    state_1_y.append(bar_position)
                    state_1_x.append(frame_idx)
    if show:
        cv2.line(frame, (th1, 0), (th1, boundary_y), (0, 0, 0), 1)
        cv2.line(frame, (th2, 0), (th2, boundary_y), (0, 0, 0), 1)
        if bar_position:
            cv2.line(frame, (bar_position, 0), (bar_position, boundary_y), (0, 0, 255), 2)
        if boundary_y:
            cv2.line(frame, (0, boundary_y), (width, boundary_y), (0, 0, 255), 2)
        # if len(lines_data):
        #     for line in lines_data:
        #         x1, y1, x2, y2 = line[0]
        #         # Draw on black background
        #         cv2.line(frame, (x1, y1), (x2, y2), (250,0,0), 2)
        cv2.imshow('Video', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


plt.plot(bar_positions_x, bar_positions_y)
plt.plot(state_1_x, state_1_y, 'x')
plt.plot(state_2_x, state_2_y, 'x')
plt.plot(state_3_x, state_3_y, 'x')
plt.plot()
plt.axhline(th1)
plt.axhline(th2)
plt.show()




# A4 size in pixels at 300 DPI
A4_WIDTH_PX = 2480
A4_HEIGHT_PX = 3508
TOP_BOTTOM_MARGIN_PX = 200

# Resize all bars to A4 width, keeping aspect ratio
resized_bars = []
for bar in bars:
    h, w = bar.shape
    scale = A4_WIDTH_PX / w
    new_h = int(h * scale)
    resized_bar = cv2.resize(bar, (A4_WIDTH_PX, new_h), interpolation=cv2.INTER_LINEAR)
    resized_bars.append(resized_bar)

# Group bars into pages, fitting as many as possible per A4 page
pages = []
current_page = []
current_height = TOP_BOTTOM_MARGIN_PX

# Add a title bar with the video_path text to each page
title_bar_height = TOP_BOTTOM_MARGIN_PX
font_scale = 2
font_thickness = 3
font = cv2.FONT_HERSHEY_PLAIN

# Create a white title bar
title_bar = np.ones((title_bar_height, A4_WIDTH_PX), dtype=np.uint8) * 255
# Put the video_path text on the title bar
title = os.path.splitext(video_path.split('/')[-1])[0]
text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
text_x = (A4_WIDTH_PX - text_size[0]) // 2
text_y = (title_bar_height + text_size[1]) // 2
cv2.putText(title_bar, title, (text_x, text_y), font, font_scale, (0,), font_thickness, cv2.LINE_AA)
current_page.append(title_bar)


for bar in resized_bars:
    if current_height + bar.shape[0] > A4_HEIGHT_PX-TOP_BOTTOM_MARGIN_PX and current_page:
        # Start new page
        pages.append(np.vstack(current_page))
        current_page = [np.ones((TOP_BOTTOM_MARGIN_PX, A4_WIDTH_PX), dtype=np.uint8) * 255]
        current_height = TOP_BOTTOM_MARGIN_PX
    current_page.append(bar)
    current_height += bar.shape[0]
    print(current_height)


if len(current_page) > 1:
    pages.append(np.vstack(current_page))

# Save each page as PDF
images = []
for page in pages:
    # Pad page to A4 height if needed
    if page.shape[0] < A4_HEIGHT_PX:
        pad_height = A4_HEIGHT_PX - page.shape[0]
        page = np.pad(page, ((0, pad_height), (0, 0)), mode='constant', constant_values=255)
    pil_img = Image.fromarray(page)
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    images.append(pil_img)

images[0].save(f"./sheets/{title}.pdf", save_all=True, append_images=images[1:])


print(title)

subprocess.Popen([f'open "./sheets/{title}.pdf"'],shell=True)
