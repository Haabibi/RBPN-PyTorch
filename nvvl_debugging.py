import nvvl
import torch

videos = ['/home/haabibi/tmp/RBPN-PyTorch/Vid4_video/new_city.mp4']

for video in videos:
    loader = nvvl.RnBLoader(width=112, height=112, consecutive_frames=1)
    loader.loadfile(video)

    for frames in loader:
        print(frames.shape)
        pass
    loader.flush()

loader.close()

