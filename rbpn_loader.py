"""Loader implementation for loading videos to the RBPN model. 

"""

def get_flow(im1, im2, flownet):
  # im1 shape: (128, 192, 3)
  # [[im1, im2]]: (1, 2, 128, 192, 3) >transpose> (1, 3, 2, 128, 192)
  # (num_clips, consecutive_frames, 3, height, width)
  # --> (num_clips, 3, consecutive_frames, height, width)
  import torch
  from torch.autograd import Variable
  flownet.eval()
  ims = torch.cat((im1, im2), 0)
  ims = ims.reshape(1, 3, 2, 128, 192)
  ims_v = Variable(ims.cuda(), requires_grad=False)
  #pred_flow = flownet(ims_v).squeeze()
  pred_flow = flownet(ims_v)
  return pred_flow


def loader(filename, frame_queue, flownet, nFrames, sta_bar, fin_bar):
  import torch
  import nvvl
  from torch.multiprocessing import Queue 
  from flownet2.models import FlowNet2
  import time
  sta_bar.wait()
  # Use our own CUDA stream to avoid synchronizing with other processes
  with torch.cuda.device(0):
    #device = torch.device('cuda:0')
    stream = torch.cuda.Stream(device=0)
    with torch.cuda.stream(stream):
      with torch.no_grad():
        loader = nvvl.RnBLoader(width=192, height=128, 
                                consecutive_frames=1,  device_id=0)
        # first "warm up" the loader with a few videos
        '''
        samples = [
          '/cmsdata/ssd0/cmslab/Kinetics-400/sparta/laughing/0gR5FP7HpZ4_000024_000034.mp4'
        ]
        for sample in samples:
          loader.loadfile(sample)
        for frames in loader:
          print("THIS IS SAMPLE FRAMES: ", frames.shape)
          pass
        loader.flush()
        '''
        
        loader.loadfile(filename)
        for frames in loader:
          pass
        #loader.flush()
        
        # NUM_FRAMES, 1, 3, 192, 128
        frames = frames.float()
        original_frames = frames
        normalized_frames = frames / 255.

        #sampler logic
        tt = int(nFrames / 2) 
        seq = [x for x in range(-tt, tt+1) if x!=0]
        NUM_FRAMES = len(frames)

        clip_collections = []
        for target in range(NUM_FRAMES):
          _tmp = []
          for i in seq:
            if target+i<0:
              # for the first tt frames, which do not have neighboring frames that come earlier,
              # append itself 
              _tmp.append(target)
            elif target+i >= NUM_FRAMES: 
              # for the last tt frames, which do not have neighboring frames that come later,
              # append itself
              _tmp.append(target)
            else:
              _tmp.append(target+i)
          clip_collections.append(_tmp)
        
        counter = 0
        for clip in clip_collections: 
          starting_time = time.time()
          original_input = original_frames[counter]
          normalized_input = normalized_frames[counter]
          original_neighbors, normalized_neighbors = [], []
          for i in range(len(clip)):
            original_neighbors.append(original_frames[clip[i]])
            normalized_neighbors.append(normalized_frames[clip[i]])
          # FlowNet2 requires tensors ranging from [0, 255] 
          flow = [get_flow(original_input, j, flownet) for j in original_neighbors]  
          # RBPN requires tensors ranging from [0, 1]
          frame_queue.put_nowait([normalized_input, normalized_neighbors, flow, starting_time])         
          counter += 1  
    
        if counter == NUM_FRAMES: 
            frame_queue.put_nowait([[], [], [], [] ])
            loader.close() 
    fin_bar.wait()
    frame_queue.cancel_join_thread()

