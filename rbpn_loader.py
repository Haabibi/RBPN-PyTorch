"""Loader implementation for loading videos to the RBPN model. 

"""

def RBPN_loader(filename, frame_queue, clip_length=7):
  import torch
  import nvvl
  from dataset import get_flow
  
  # Use our own CUDA stream to avoid synchronizing with other processes
  with torch.cuda.device(0):
    device = torch.device('cuda:0')
    stream = torch.cuda.Stream(device=0)
    with torch.cuda.stream(stream):
      with torch.no_grad():
        loader = nvvl.RnBLoader(width=480, height=270, 
                                consecutive_frames=1, device_id=0)
        
        # first "warm up" the loader with a few videos
        samples = [
          '/cmsdata/ssd0/cmslab/Kinetics-400/sparta/laughing/0gR5FP7HpZ4_000024_000034.mp4'
          #'/home/haabibi/tmp/RBPN-PyTorch/SPMCS/SPMCS_mp4_vid/AMVTG_004_truth.mp4'
        ]
        
        for sample in samples:
          loader.loadfile(sample)
        for frames in loader:
          pass
        loader.flush()

        
        loader.loadfile(filename)
        for frames in loader:
          print("RBPN_LOADER: ", type(frames), frames.size())
          pass
        loader.flush()

        frames = frames.float()
        # (num_clips, consecutive_frames, 3, height, width)
        # --> (num_clips, 3, consecutive_frames, height, width)
        frames = frames.permute(0, 2, 1, 3, 4)
        # --> (num_clips, 3, height, width)
        frames = frames.squeeze()
        
        # SAMPLER LOGIC COMES HERE
        tt = int(clip_length/2)
        seq = [x for x in range(-tt, tt+1) if x!=0] 
        frame_idx_list = range(len(frames))
        
        for i in frame_idx_list:
          input = frames[i]
          neighbor = [] 
          print("FRAME IDX LIST: ", i)
          for s in seq:
            if i+s in frame_idx_list:
              neighbor.append(frames[i+s])
              print("IS i+s in the list? YES", i+s)
            else: 
              neighbor.append(frames[i])
              print("IS i+s in the list? no", i+s)
          print("THIS IS NEIGHBOR: ", len(neighbor)) 
          #flow = [get_flow(input, j) for j in neighbor]  

          #frame_queue.put_nowait([input, neighbor, flow])
          frame_queue.put_nowait([input, neighbor])
    
        loader.close()
        loader.flush()
