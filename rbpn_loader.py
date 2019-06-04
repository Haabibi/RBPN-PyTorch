"""Loader implementation for loading videos to the RBPN model. 

"""
from torch.multiprocessing import Queue 

def loader(filename, frame_queue):
  import torch
  import nvvl
  from rbpn_sampler import RBPNSampler
  # Use our own CUDA stream to avoid synchronizing with other processes
  with torch.cuda.device(0):
    device = torch.device('cuda:0')
    stream = torch.cuda.Stream(device=0)
    with torch.cuda.stream(stream):
      with torch.no_grad():
        loader = nvvl.RnBLoader(width=112, height=112, 
                                consecutive_frames=1, device_id=0)
        # first "warm up" the loader with a few videos
        samples = [
          '/cmsdata/ssd0/cmslab/Kinetics-400/sparta/laughing/0gR5FP7HpZ4_000024_000034.mp4'
        ]
        for sample in samples:
          loader.loadfile(sample)
        for frames in loader:
          pass
        loader.flush()

        
        loader.loadfile(filename)
        for frames in loader:
          pass
        loader.flush()

        #frames = frames.float()
        # (num_clips, consecutive_frames, 3, width, height)
        # --> (num_clips, 3, consecutive_frames, width, height)
        #frames = frames.permute(0, 2, 1, 3, 4)
        
        #target, input, neighbor = 

        #frame_queue.put_nowait(frames)
    
        loader.close()
if __name__ == '__main__':
  queue = Queue()
  loader( '/cmsdata/ssd0/cmslab/Kinetics-400/sparta/laughing/0gR5FP7HpZ4_000024_000034.mp4', queue)
