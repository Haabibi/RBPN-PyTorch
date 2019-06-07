"""Loader implementation for loading videos to the RBPN model. 

"""
from torch.multiprocessing import Queue 
from flownet2.models import FlowNet2
from torch.autograd import Variable

def get_flow(im1, im2, flownet, opt):
 #im1 shape: (128, 192, 3)
 #[[im1, im2]]: (1, 2, 128, 192, 3) >transpose> (1, 3, 2, 128, 192)
 # (num_clips, consecutive_frames, 3, height, width)
 # --> (num_clips, 3, consecutive_frames, height, width)
  ims = torch.cat((im1, im2), 0)
  ims = ims.reshape(1, 3, 2, 128, 192)
  ims_v = Variable(ims.cuda(), requires_grad=False)
  
  pred_flow = flownet(ims_v).squeeze()
  return pred_flow


def loader(filename, frame_queue, flownet, nFrames):
  import torch
  import nvvl
  # Use our own CUDA stream to avoid synchronizing with other processes
  with torch.cuda.device(0):
    device = torch.device('cuda:0')
    stream = torch.cuda.Stream(device=0)
    with torch.cuda.stream(stream):
      with torch.no_grad():
        loader = nvvl.RnBLoader(width=192, height=128, 
                                consecutive_frames=1, device_id=0)
        # first "warm up" the loader with a few videos
        samples = [
          '/cmsdata/ssd0/cmslab/Kinetics-400/sparta/laughing/0gR5FP7HpZ4_000024_000034.mp4'
        ]
        for sample in samples:
          loader.loadfile(sample)
        for frames in loader:
          pass
        #loader.flush()

        
        loader.loadfile(filename)
        for frames in loader:
          pass
        #loader.flush()
        
        #300, 1, 3, 112, 112
        frames = frames.float()
        #sampler logic
        num_neighbors = int(nFrames / 2) 
        num_frames = len(frames)
        clip_collections = [range(i-3, i+4) for i in range(num_neighbors, num_frames-4)]
        print("THIS IS CLIP_COLLECTIONS: ", clip_collections)
        for clip in clip_collections: 
          mid_idx = clip[int(len(clip)/2)]
          input = frames[mid_idx]
          neighbors = []
          for i in range(len(clip)):
            if i != 3: 
              neighbors.append(frames[clip[i]])
          #neighbors.extend([frames[clip[i]] for i in clip if i != mid_idx])
          flow = [get_flow(input, j, flownet, opt) for j in neighbors]  
          frame_queue.put_nowait([input, neighbors, flow])         

    
        #loader.close()
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
