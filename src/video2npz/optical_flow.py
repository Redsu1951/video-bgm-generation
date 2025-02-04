# encoding=utf-8
import cv2
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Function to create directories if they don't exist
def makedirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

# Set directories for video, flow data, and figure output
video_dir = '../../videos/'
flow_dir = 'flow/'
fig_dir = 'fig/'
makedirs([video_dir, flow_dir, fig_dir, 'optical_flow/'])

TIME_PER_BAR = 2  # Time per segment (in seconds) for calculating flow magnitude

# Optical Flow calculation function using different methods
def dense_optical_flow(method, video_path, params=[], to_gray=False):
    assert os.path.exists(video_path), f"Video path {video_path} does not exist"
    
    # Open the video with OpenCV
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Get the first frame
    ret, old_frame = cap.read()
    if not ret:
        print("Error: unable to read the video frame.")
        return

    # Convert to grayscale if required
    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    flow_magnitude_list = []

    # Iterate through the video frames
    while True:
        ret, new_frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to grayscale if required
        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        flow = method(old_frame, new_frame, None, *params)
        flow_magnitude = np.mean(np.abs(flow))
        flow_magnitude_list.append(flow_magnitude)

        # Update the previous frame
        old_frame = new_frame

    cap.release()

    frame_per_bar = TIME_PER_BAR * fps
    flow_magnitude_per_bar = []
    temp = np.zeros(len(flow_magnitude_list))

    # Calculate the flow magnitude for each segment (based on the frame rate)
    for i in np.arange(0, len(flow_magnitude_list), frame_per_bar):
        mean_flow = np.mean(flow_magnitude_list[int(i): min(int(i + frame_per_bar), len(flow_magnitude_list))])
        flow_magnitude_per_bar.append(mean_flow)
        temp[int(i): min(int(i + frame_per_bar), len(flow_magnitude_list))] = mean_flow

    # Save flow magnitudes to a file
    np.savez(os.path.join(flow_dir, os.path.basename(video_path).split('.')[0] + '.npz'),
             flow=np.asarray(flow_magnitude_list))

    # Create a plot for the flow magnitude
    x = np.arange(0, len(flow_magnitude_list))
    plt.figure(figsize=(10, 4))
    plt.plot(x, temp, 'b.')
    plt.title('Optical Flow Magnitude')
    plt.savefig(os.path.join(fig_dir, os.path.basename(video_path).split('.')[0] + '.jpg'))

    return flow_magnitude_per_bar, flow_magnitude_list

# Main function to execute the optical flow calculation
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["farneback", "lucaskanade_dense", "rlof"], default="farneback")
    parser.add_argument("--video", default="../../videos/final_640.mp4")
    args = parser.parse_args()

    video_path = args.video
    print("Video path:", video_path)

    flow = []

    if args.method == 'lucaskanade_dense':
        method = cv2.optflow.calcOpticalFlowSparseToDense
        optical_flow, flow_magnitude_list = dense_optical_flow(method, video_path, to_gray=True)
    elif args.method == 'farneback':
        method = cv2.calcOpticalFlowFarneback
        params = [0.5, 3, 15, 3, 5, 1.2, 0]  # Default Farneback algorithm parameters
        optical_flow, flow_magnitude_list = dense_optical_flow(method, video_path, params, to_gray=True)
    elif args.method == "rlof":
        method = cv2.optflow.calcOpticalFlowDenseRLOF
        optical_flow, flow_magnitude_list = dense_optical_flow(method, video_path)

    flow += optical_flow

    flow = np.asarray(flow)
    
    # Print the flow percentiles for analysis
    for percentile in range(10, 101, 10):
        print('Percentile %d: %.4f' % (percentile, np.percentile(flow, percentile)))

    # Save the flow results
    np.savez('optical_flow/flow.npz', flow=flow)
