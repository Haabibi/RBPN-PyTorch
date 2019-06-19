if __name__ == '__main__':
  from torch.multiprocessing import set_start_method
  set_start_method('spawn', force=True)
    
  import argparse
  import os
  import torch
  import torch.nn as nn
  import torch.optim as optim
  from torch.autograd import Variable
  from rbpn import Net as RBPN
  from functools import reduce
  import numpy as np

  from imageio import imsave
  import scipy.io as sio
  import time
  import cv2
  import math
  import pdb

  from rbpn_loader import loader
  from rbpn_eval import eval, save_vid
  import nvvl
  from flownet2.models import FlowNet2 

  from torch.multiprocessing import Process, Queue, Barrier

  # Training settings
  parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
  parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
  parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
  parser.add_argument('--gpu_mode', type=bool, default=True)
  parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
  parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
  parser.add_argument('--data_dir', type=str, default='./Vid4')
  parser.add_argument('--file_list', type=str, default='foliage.txt')
  parser.add_argument('--vid_dir', type=str, default='./Vid4_video/new_city.mp4')
  parser.add_argument('--other_dataset', type=bool, default=True, help="use other dataset than vimeo-90k")
  parser.add_argument('--future_frame', type=bool, default=True, help="use future frame")
  parser.add_argument('--nFrames', type=int, default=7)
  parser.add_argument('--model_type', type=str, default='RBPN')
  parser.add_argument('--residual', type=bool, default=False)
  parser.add_argument('--output', default='Results/', help='Location to save checkpoint models')
  parser.add_argument('--model', default='weights/RBPN_4x.pth', help='sr pretrained base model')
   

  ## FlowNet specific parser arguments ##
  parser.add_argument('--rgb_max', type=float, default=255.0)
  parser.add_argument('--fp16', action='store_true')
  opt = parser.parse_args()

  gpus_list=range(opt.gpus)
  print(opt)

  cuda = opt.gpu_mode
  if cuda and not torch.cuda.is_available():
      raise Exception("No GPU found, please run without --cuda")

  torch.manual_seed(opt.seed)
  if cuda:
      torch.cuda.manual_seed(opt.seed)

  bar_total = 4 # 1 main process + 1 eval + 1 flownet + 1 save_vid 
  
  sta_bar = Barrier(bar_total)
  fin_bar = Barrier(bar_total)

  frame_queue = Queue()
  vid_frame_queue = Queue()

  print('===> Building model ', opt.model_type)
  if opt.model_type == 'RBPN':
      model = RBPN(num_channels=3, base_filter=256,  feat = 64, num_stages=3, n_resblock=5, nFrames=opt.nFrames, scale_factor=opt.upscale_factor)

  if cuda:
      model = torch.nn.DataParallel(model, device_ids=gpus_list)

  model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
  print('Pre-trained SR model is loaded.')

  print('===> Building FlowNet model ')
  path = '/home/haabibi/official-flownet2-pytorch/ckpt/FlowNet2_checkpoint.pth.tar'
  flownet2 = FlowNet2(opt)
  pretrained_dict = torch.load(path)['state_dict']
  model_dict = flownet2.state_dict()
  pretrained_dict = {k:v for k,
                     v in pretrained_dict.items() if k in model_dict}
  model_dict.update(pretrained_dict)

  flownet2.load_state_dict(model_dict)
  print('Pre-trained FlowNet model is loaded.') 

  if cuda:
      model = model.cuda(gpus_list[0])
      flownet2 = flownet2.cuda(gpus_list[0])
          
  eval_p = Process(target=eval, args=(model, flownet2, frame_queue, vid_frame_queue, opt, sta_bar, fin_bar))    
  loader_p = Process(target=loader, args=(opt.vid_dir, frame_queue, flownet2, 7, sta_bar, fin_bar))
  save_vid_p = Process(target=save_vid, args=(vid_frame_queue, sta_bar, fin_bar))

  for p in [eval_p, loader_p, save_vid_p]:
      p.start()
      
  sta_bar.wait()
  
  fin_bar.wait()
  for p in [eval_p, loader_p]:
      p.join()
  print("ALL PROCESSES ENDING PROPERLY!")
