import cv2
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import skvideo.io
from tqdm import tqdm


def makedirs(dirs: list):
	for dir in dirs:
		if not os.path.exists(dir):
			os.makedirs(dir)

video_dir = '../../videos/'
flow_dir = 'flow/'
fig_dir = 'fig/'
makedirs([video_dir, flow_dir, fig_dir])

TIME_PER_BAR = 2  # 暂定2s一小节

# 文字参数
ORG = (50, 50)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
COLOR = (0, 0, 255)
THICKNESS = 2


def dense_optical_flow(method, video_path, params=[], to_gray=False):
#	print(video_path)
	assert os.path.exists(video_path)
	metadata = skvideo.io.ffprobe(video_path)
	print(metadata)
	frame, time = metadata['video']['@avg_frame_rate'].split('/')
	fps = round(float(frame) / float(time))
	if os.path.exists(os.path.join(flow_dir, video_path.split('/')[-1].split('.')[0] + '.npz')):
		flow_magnitude_list = list(np.load(os.path.join(flow_dir, video_path.split('/')[-1].split('.')[0] + '.npz'))['flow'])
	else:
		# Read the video and first frame
		video = skvideo.io.vread(video_path)[:]
		n_frames = len(video)  # 总帧数
		old_frame = video[0]

		# crate HSV & make Value a constant
		hsv = np.zeros_like(old_frame)
		hsv[..., 1] = 255

		# Preprocessing for exact method
		if to_gray:
			old_frame = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)

		flow_magnitude_list = []
		# optical_flow = np.zeros_like(video)
		for i in tqdm(range(1, n_frames)):
			# Read the next frame
			new_frame = video[i]
			# frame_copy = cv2.cvtColor(new_frame, cv2.COLOR_RGB2BGR)

			# Preprocessing for exact method
			if to_gray:
				new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)

			# Calculate Optical Flow
			flow = method(old_frame, new_frame, None, *params)
			flow_magnitude = np.mean(np.abs(flow))
			flow_magnitude_list.append(flow_magnitude)

			# # Encoding: convert the algorithm's output into Polar coordinates
			# mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
			# # Use Hue and Value to encode the Optical Flow
			# hsv[..., 0] = ang * 180 / np.pi / 2
			# hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

			# # Convert HSV image into BGR for demo
			# bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
			# cv2.putText(frame_copy, '%.2f' % flow_magnitude, ORG, FONT, FONT_SCALE, COLOR, THICKNESS)
			# cv2.putText(bgr, '%.2f' % flow_magnitude, ORG, FONT, FONT_SCALE, COLOR, THICKNESS)
			# cv2.imshow("frame", frame_copy)
			# cv2.imshow("optical flow", bgr)
			# optical_flow[i-1] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
			# k = cv2.waitKey(25) & 0xFF
			# if k == 27:
			# 	break

			# Update the previous frame
			old_frame = new_frame
	
	# cv2.destroyAllWindows()

	frame_per_bar = TIME_PER_BAR * fps
	flow_magnitude_per_bar = []
	temp = np.zeros((len(flow_magnitude_list)))
	for i in range(0, len(flow_magnitude_list), frame_per_bar):
		mean_flow = np.mean(flow_magnitude_list[i : min(i+frame_per_bar, len(flow_magnitude_list))])
		flow_magnitude_per_bar.append(mean_flow)
		temp[i : min(i+frame_per_bar, len(flow_magnitude_list))] = mean_flow

	np.savez(os.path.join(flow_dir, video_path.split('/')[-1].split('.')[0] + '.npz'), flow=np.asarray(flow_magnitude_list))

	# 绘制flow强度折线图
	x = np.arange(0, len(flow_magnitude_list))
	plt.figure(figsize=(10, 4))
	# plt.plot(x, flow_magnitude_list, 'r.')
	plt.plot(x, temp, 'b.')
	plt.title('Optical Flow Magnitude')
	# plt.xlabel('Frame')
	# plt.ylabel('Optical Flow Magnitude')
	# plt.legend()
	# plt.show()
	plt.savefig(os.path.join(fig_dir, video_path.split('/')[-1].split('.')[0] + '.jpg'))

	# return optical_flow, flow_magnitude_list
	return flow_magnitude_per_bar, flow_magnitude_list


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--method", choices=["farneback", "lucaskanade_dense", "rlof"], default="farneback")
	parser.add_argument("--video", default="../../videos/pku.mp4")
	args = parser.parse_args()

	flow = []
	if True:
#		video_path = 'vlog_360p.mp4'
	# for i, video_path in enumerate(os.listdir(video_dir)[:]):
	# 	# try:
	# 	print('Processing %d/%d: %s' % (i, len(os.listdir(video_dir)[:]), os.path.join(video_dir, video_path)))
#		if '.mp4' in video_path:
		if True:
			video_path = args.video
			print("video_path", video_path)
			if args.method == 'lucaskanade_dense':
				method = cv2.optflow.calcOpticalFlowSparseToDense
				optical_flow, flow_magnitude_list = dense_optical_flow(method, video_path, to_gray=True)
			elif args.method == 'farneback':
				method = cv2.calcOpticalFlowFarneback
				params = [0.5, 3, 15, 3, 5, 1.2, 0]  # default Farneback's algorithm parameters
				optical_flow, flow_magnitude_list = dense_optical_flow(method, video_path, params, to_gray=True)
			elif args.method == "rlof":
				method = cv2.optflow.calcOpticalFlowDenseRLOF
				optical_flow, flow_magnitude_list = dense_optical_flow(method, video_path)

			flow += optical_flow
		# except Exception as ex:
			# print(ex)
	
	flow = np.asarray(flow)
	for percentile in range(10, 101, 10):
		print('percentile %d: %.4f' % (percentile, np.percentile(flow, percentile)))
	np.savez('optical_flow/flow.npz', flow = flow)


	# 保存Optical Flow
	# skvideo.io.vwrite('optical_flow_%s.mp4' % args.method, optical_flow)
