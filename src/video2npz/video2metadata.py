#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import matplotlib
matplotlib.use('Agg')
import visbeat3 as vb
import os
import os.path as osp
import cv2
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def makedirs(d):
    if not osp.exists(d):
        os.makedirs(d)


def frange(start, stop, step=1.0):
    while start < stop:
        yield start
        start += step


def get_flow_directory():
    """Returns the correct path to the 'flow' directory relative to the current script."""
    script_dir = osp.dirname(os.path.realpath(__file__))
    flow_dir = osp.join(script_dir, 'flow')
    if osp.exists(flow_dir):
        return flow_dir
    else:
        raise FileNotFoundError("Flow directory not found. Please ensure it exists in the correct location.")


def process_all_videos(args):
    out_json = {}
    for i, video_name in enumerate(os.listdir(args.video_dir)):
        if '.mp4' not in video_name:
            continue
        print('%d/%d: %s' % (i, len(os.listdir(args.video_dir)), video_name))
        metadata = process_video(video_name, args)
        out_json[video_name] = metadata

    json_str = json.dumps(out_json, indent=4)
    with open(osp.join(args.video_dir, 'metadata.json'), 'w') as f:
        f.write(json_str)


def process_video(video_path, args):
    figsize = (32, 4)
    dpi = 200
    xrange = (0, 95)
    x_major_locator = MultipleLocator(2)

    vb.Video.getVisualTempo = vb.Video_CV.getVisualTempo

    video = os.path.basename(video_path)
    vlog = vb.PullVideo(name=video, source_location=osp.join(video_path), max_height=360)
    vbeats = vlog.getVisualBeatSequences(search_window=None)[0]

    tempo, beats = vlog.getVisualTempo()
    print("Tempo is", tempo)
    vbeats_list = []
    for vbeat in vbeats:
        i_beat = np.round(vbeat.start.item() / 60 * tempo.item() * 4)  # Extract scalar with tempo.item()

        vbeat_dict = {
            'start_time': vbeat.start,
            'bar'       : int(i_beat // 16),
            'tick'      : int(i_beat % 16),
            'weight'    : vbeat.weight
        }
        if vbeat_dict['tick'] % args.resolution == 0:  # only select vbeat that lands on the xth tick
            vbeats_list.append(vbeat_dict)
    print('%d / %d vbeats selected' % (len(vbeats_list), len(vbeats)))

    flow_dir = get_flow_directory()
    npz_path = osp.join(flow_dir, video.replace('.mp4', '.npz'))
    if not osp.exists(npz_path):
        raise FileNotFoundError(f"Flow file for {video} not found at {npz_path}")

    npz = np.load(npz_path, allow_pickle=True)
    print(npz.keys())
    flow_magnitude_list = npz['flow']
    fps = round(vlog.n_frames() / float(vlog.getDuration()))
    fpb = int(round(fps * 4 * 60 / tempo.item()))  # Extract scalar with tempo.item()

    fmpb = []  # flow magnitude per bar
    temp = np.zeros((len(flow_magnitude_list)))
    for i in range(0, len(flow_magnitude_list), fpb):
        mean_flow = np.mean(flow_magnitude_list[i: min(i + fpb, len(flow_magnitude_list))])
        fmpb.append(float(mean_flow))
        temp[i: min(i + fpb, len(flow_magnitude_list))] = mean_flow

    # Visualization and other code remains unchanged...
    return {
        'duration': vlog.getDuration(),
        'tempo': tempo.item(),
        'vbeats': vbeats_list,
        'flow_magnitude_per_bar': fmpb,
        'beats_array': beats.tolist() if isinstance(beats, np.ndarray) else beats,
    }


if __name__ == '__main__':
    # vb.SetAssetsDir('.' + os.sep + 'VisBeatAssets' + os.sep)

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='../../videos/final_640.mp4')
    parser.add_argument('--visualize', action='store_true', default=True)
    parser.add_argument('--resolution', type=int, default=1)
    args = parser.parse_args()

    metadata = process_video(args.video, args)
    with open("metadata.json", "w") as f:
        json.dump(metadata, f)
    print("saved to metadata.json")
