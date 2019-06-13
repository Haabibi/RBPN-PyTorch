def eval(model, flownet2, frame_queue, opt):
    import torch
    import time 

    from torch.autograd import Variable
    model.eval()
    count = 1
    while True: 
        batch = frame_queue.get()

        input, neighbors, flow = batch[0], batch[1], batch[2]

        with torch.no_grad():
            
            input = Variable(input).cuda()
            neighbors = [Variable(j).cuda() for j in neighbors]
            flow = [Variable(j).cuda().float() for j in flow]
            
            neighbors_ = torch.stack(neighbors, dim=1) 

            flow_ = torch.stack(flow, dim=1)
            print("[NVVL {}] RBPN Input of target frame: ".format(count), input.shape, input.type(), torch.mean(input), torch.max(input), torch.min(input)) 
            print("[NVVL {}] RBPN Input of neighbors: ".format(count), neighbors_.shape, neighbors_.type(), torch.mean(neighbors_), torch.max(neighbors_), torch.min(neighbors_)) 
            print("[NVVL {}] RBPN Input of flows: ".format(count), flow_.shape, flow_.type(), torch.mean(flow_), torch.max(flow_), torch.min(flow_)) 
            print("[[[[[[[[L18]]]]]]]]", neighbors[0].shape, flow[0].shape)
            t0 = time.time()
            prediction = model(input, neighbors, flow)
            t1 = time.time()         
            print("[NVVL] RBPN Output: ", prediction.shape, prediction.type(), torch.mean(prediction))
            save_img(prediction.cpu().data, str(count), True, opt)
            print("===> Processing: %s || Timer: %.4f sec." % (str(count), (t1-t0)))

        count+=1
    
def save_img(img, img_name, pred_flag, opt):
    import os 
    import torch
    import cv2
    import torchvision.transforms.functional as F

    #print(img.shape, img)
    # img.shape ==> [1, 3, 512, 768]
    # original cv2 version 
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
    #save_img = img.squeeze().numpy().transpose(1,2,0)
    # save_img.shape ==> (512, 768, 3)
    #print(save_img.shape, save_img)

    # new way of saving in PIL image
    #frame = img.to(torch.uint8)
    #save_img = frame.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
    #save_img = F.to_pil_image(frame.squeeze().cpu())
    
    # save img
    save_dir=os.path.join(opt.output, opt.data_dir, os.path.splitext(opt.file_list)[0]+'_'+str(opt.upscale_factor)+'x')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    if pred_flag:
        save_fn = save_dir +'/'+ img_name+'_'+opt.model_type+'F'+str(opt.nFrames)+'.png'
    else:
        save_fn = save_dir +'/'+ img_name+'.png'
    
    #save_img.save(save_fn, "PNG")
    #print("[L28] EVAL.PY: ", save_img.shape, save_img.dtype, type(save_img)) 
    cv2.imwrite(save_fn, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])


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
  import nvvl
  from flownet2.models import FlowNet2 

  from torch.multiprocessing import Process, Queue

  # Training settings
  parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
  parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
  parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
  parser.add_argument('--gpu_mode', type=bool, default=True)
  parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
  parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
  parser.add_argument('--data_dir', type=str, default='./Vid4')
  parser.add_argument('--file_list', type=str, default='foliage.txt')
  parser.add_argument('--vid_dir', type=str, default='./Vid4/video/foliage.avi')
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

  frame_queue = Queue()

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
          
  eval_p = Process(target=eval, args=(model, flownet2, frame_queue, opt))    
  loader_p = Process(target=loader, args= ( '/cmsdata/ssd0/cmslab/Kinetics-400/sparta/laughing/0gR5FP7HpZ4_000024_000034.mp4', frame_queue, flownet2, 7))

  for p in [eval_p, loader_p]:
      p.start()
      

  for p in [eval_p, loader_p]:
      p.join()

