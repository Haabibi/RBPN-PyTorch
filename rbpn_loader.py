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
  sta_bar.wait()
  # Use our own CUDA stream to avoid synchronizing with other processes
  with torch.cuda.device(0):
    #device = torch.device('cuda:0')
    stream = torch.cuda.Stream(device=0)
    with torch.cuda.stream(stream):
      with torch.no_grad():
        loader = nvvl.RnBLoader(width=192, height=128, 
                                consecutive_frames=1, device_id=0)
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
        loader.flush()
        
        #300, 1, 3, 112, 112
        frames = frames.float()
        #ONLY SAMPLE FIRST FIVE FRAMES
        #frames = frames[:10]
        original_frames = frames
        normalized_frames = frames / 255.
        print("THIS IS FRAMES: ", frames.shape, frames[0], torch.max(frames[0]), torch.min(frames[0]), torch.mean(frames[0]))

        #sampler logic
        tt = int(nFrames / 2) 
        seq = [x for x in range(-tt, tt+1) if x!=0]
        num_frames = len(frames)

        clip_collections = []
        for target in range(num_frames):
          _tmp = []
          for i in seq:
            if target+i<0:
              _tmp.append(target)
            elif target+i >= num_frames:
              _tmp.append(target)
            else:
              _tmp.append(target+i)
          clip_collections.append(_tmp)
        
        print(len(frames), "THIS IS CLIP COLLECTIONS: ", clip_collections)
        #clip_collections = [range(i-3, i+4) for i in range(num_neighbors, num_frames-4)]
        counter = 1
        for clip in clip_collections: 
          mid_idx = clip[int(len(clip)/2)]
          original_input = original_frames[mid_idx]
          normalized_input = normalized_frames[mid_idx]
          original_neighbors = []
          normalized_neighbors = []
          for i in range(len(clip)):
            original_neighbors.append(original_frames[clip[i]])
            normalized_neighbors.append(normalized_frames[clip[i]])
          
          
          #neighbors.extend([frames[clip[i]] for i in clip if i != mid_idx])
          neighbors_tensor = torch.stack(normalized_neighbors, dim=1)
          print("[NVVL {}] FlowNet INPUT of Neighbors: ".format(counter), neighbors_tensor.is_cuda, neighbors_tensor.shape, neighbors_tensor.type(), torch.mean(neighbors_tensor), torch.max(neighbors_tensor), torch.min(neighbors_tensor))

          flow = [get_flow(original_input, j, flownet) for j in original_neighbors]  

          frame_queue.put_nowait([normalized_input, normalized_neighbors, flow])         
          flow_tensor = torch.stack(flow, dim=1)
          print("[NVVL {}] FlowNet OUTPUT: ".format(counter), flow_tensor.is_cuda, flow_tensor.shape, flow_tensor.type(), torch.mean(flow_tensor), torch.max(flow_tensor), torch.min(flow_tensor))
          counter += 1  
    
        loader.close()
    fin_bar.wait()

'''
if __name__ == '__main__':
  import argparse
  import torch
  parser = argparse.ArgumentParser()
  parser.add_argument('--rgb_max', type=float, default=255.0)
  parser.add_argument('--fp16', action='store_true')
  opt = parser.parse_args()
  #rgb_max = 255.0
  #fp16=False
  
  queue = Queue()
  print('===>Building FlowNet model ')
  path = '/home/haabibi/official-flownet2-pytorch/ckpt/FlowNet2_checkpoint.pth.tar'
  flownet2 = FlowNet2(opt)
  pretrained_dict = torch.load(path)['state_dict']
  model_dict = flownet2.state_dict()
  pretrained_dict = {k:v for k,
                     v in pretrained_dict.items() if k in model_dict}
  model_dict.update(pretrained_dict)
  flownet2.load_state_dict(model_dict)
  flownet2.cuda()
  loader( '/cmsdata/ssd0/cmslab/Kinetics-400/sparta/laughing/0gR5FP7HpZ4_000024_000034.mp4', queue, flownet2, 7)
  while True:
    input, neighbors, flow = queue.get()
    print(input.shape, len(neighbors), neighbors[0].shape, len(flow), flow[0].shape)

'''
