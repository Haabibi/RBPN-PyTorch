from __future__ import print_function
import argparse

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from rbpn import Net as RBPN
from data import get_test_set
from functools import reduce
import numpy as np
from imageio import imsave
import scipy.io as sio
import time
import cv2
import math
import pdb
from flownet2.models import FlowNet2
from multiprocessing import Queue
from PIL import Image

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--chop_forward', type=bool, default=False)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='./Vid4')
parser.add_argument('--file_list', type=str, default='new_city.txt')
parser.add_argument('--other_dataset', type=bool, default=True, help="use other dataset than vimeo-90k")
parser.add_argument('--future_frame', type=bool, default=True, help="use future frame")
parser.add_argument('--nFrames', type=int, default=7)
parser.add_argument('--model_type', type=str, default='RBPN')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--output', default='Results/', help='Location to save checkpoint models')
parser.add_argument('--model', default='weights/RBPN_4x.pth', help='sr pretrained base model')
parser.add_argument('--rgb_max', type=float, default=255.)
parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode.')
opt = parser.parse_args()

gpus_list=range(opt.gpus)
print(opt)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

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

print('===> Loading datasets')
time_queue = Queue()
inf_queue = Queue()
test_set = get_test_set(opt.data_dir, opt.nFrames, opt.upscale_factor, opt.file_list, opt.other_dataset, opt.future_frame, flownet2, time_queue, opt)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model ', opt.model_type)
if opt.model_type == 'RBPN':
    model = RBPN(num_channels=3, base_filter=256,  feat = 64, num_stages=3, n_resblock=5, nFrames=opt.nFrames, scale_factor=opt.upscale_factor)

if cuda:
    model = torch.nn.DataParallel(model, device_ids=gpus_list)

model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
print('Pre-trained SR model is loaded.')


if cuda:
    model = model.cuda(gpus_list[0])

def eval(inf_queue):
    model.eval()
    count=1
    avg_psnr_predicted = 0.0
    for batch in testing_data_loader:
        t_real_zero = time.time()
        input, target, neigbor, flow, bicubic = batch[0], batch[1], batch[2], batch[3], batch[4]
        print("[L95] INPUT: ", input[0]) 
        with torch.no_grad():
            input = Variable(input).cuda(gpus_list[0])
            bicubic = Variable(bicubic).cuda(gpus_list[0])
            neigbor = [Variable(j).cuda(gpus_list[0]) for j in neigbor]
            flow = [Variable(j).cuda(gpus_list[0]).float() for j in flow]
            print("[L101] INPUT: ", input[0]) 

        t0 = time.time()
        neighbors_ = torch.stack(neigbor, dim=1)
        flow_ = torch.stack(flow, dim=1)
        print("[NVVL {}] RBPN Input of target frame: ".format(count), input.shape, input.type(), torch.mean(input), torch.max(input), torch.min(input))
        print("[NVVL {}] RBPN Input of neighbors: ".format(count), neighbors_.shape, neighbors_.type(), torch.mean(neighbors_), torch.max(neighbors_), torch.mean(neighbors_))
        print("[NVVL {}] RBPN Input of flows: ".format(count), flow_.shape, flow_.type(), torch.mean(flow_), torch.max(flow_), torch.min(flow_))
        if opt.chop_forward:
            with torch.no_grad():
                prediction = chop_forward(input, neigbor, flow, model, opt.upscale_factor)
        else:
            with torch.no_grad():
                
                print("[[[[LINE113]]]]", neigbor[0].shape, flow[0].shape)
                prediction = model(input, neigbor, flow) 
                print("[NVVL] RBPN Output: ", prediction.shape, prediction.type(), torch.mean(prediction), torch.max(prediction), torch.min(prediction))
        
        if opt.residual:
            prediction = prediction + bicubic
            
        t1 = time.time()
        print("===> Processing: %s || Timer: %.4f sec. " % (str(count), (t1 - t0)))
        inf_queue.put(t1-t0) 
        save_img(prediction.cpu().data, str(count), True)
        save_img(target, str(count), False)
        
        prediction = prediction.cpu()
        prediction = prediction.data[0].numpy().astype(np.float32)
        prediction = prediction*255.
        
        target = target.squeeze().numpy().astype(np.float32)
        target = target*255.
        #print("THIS IS PREDICTION & TARGET: ", prediction.shape, target.shape)
        psnr_predicted = PSNR(prediction,target, shave_border=opt.upscale_factor)
        avg_psnr_predicted += psnr_predicted
        count+=1
        #print("inside loop: AVG_PSNR: {}, COUNT: {}".format(avg_psnr_predicted, count)) 

    print("AVG_PSNR: {}, COUNT: {}".format(avg_psnr_predicted, count)) 
    print("PSNR_predicted=", avg_psnr_predicted/count)

def save_img(img, img_name, pred_flag):
    #print(type(img), img.shape, img)
    # img.shape ==> [1, 3, 512, 768]
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
    #save_img = img.squeeze().numpy().transpose(1,2,0)
    # save_img.shape ==> (512, 768, 3)
    #print(save_img.shape, save_img)

    # save img
    save_dir=os.path.join(opt.output, opt.data_dir, os.path.splitext(opt.file_list)[0]+'_'+str(opt.upscale_factor)+'x')
    #print("L118 saving directory: ", save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    if pred_flag:
        save_fn = save_dir +'/'+ img_name+'_'+opt.model_type+'F'+str(opt.nFrames)+'.png'
    else:
        save_fn = save_dir +'/'+ img_name+'.png'
    
    # original way of saving img using cv2 
    cv2.imwrite(save_fn, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    # new way of saving img using PIL
    #save_img = Image.fromarray(save_img)
    #save_img.save(save_fn)

def PSNR(pred, gt, shave_border=0):
    #height, width = pred.shape[:2]
    height, width = pred.shape[1:]
    #pred = pred[1+shave_border:height - shave_border, 1+shave_border:width - shave_border, :]
    #gt = gt[1+shave_border:height - shave_border, 1+shave_border:width - shave_border, :]
    pred = pred[:, 1+shave_border:height - shave_border, 1+shave_border:width - shave_border]
    gt = gt[:, 1+shave_border:height - shave_border, 1+shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        print("L135 at eval.py HERE! rmse==0!")
        return 100
    return 20 * math.log10(255.0 / rmse)
    
def chop_forward(x, neigbor, flow, model, scale, shave=8, min_size=2000, nGPUs=opt.gpus):
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        [x[:, :, 0:h_size, 0:w_size], [j[:, :, 0:h_size, 0:w_size] for j in neigbor], [j[:, :, 0:h_size, 0:w_size] for j in flow]],
        [x[:, :, 0:h_size, (w - w_size):w], [j[:, :, 0:h_size, (w - w_size):w] for j in neigbor], [j[:, :, 0:h_size, (w - w_size):w] for j in flow]],
        [x[:, :, (h - h_size):h, 0:w_size], [j[:, :, (h - h_size):h, 0:w_size] for j in neigbor], [j[:, :, (h - h_size):h, 0:w_size] for j in flow]],
        [x[:, :, (h - h_size):h, (w - w_size):w], [j[:, :, (h - h_size):h, (w - w_size):w] for j in neigbor], [j[:, :, (h - h_size):h, (w - w_size):w] for j in flow]]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, nGPUs):
            with torch.no_grad():
                input_batch = inputlist[i]#torch.cat(inputlist[i:(i + nGPUs)], dim=0)
                output_batch = model(input_batch[0], input_batch[1], input_batch[2])
            outputlist.extend(output_batch.chunk(nGPUs, dim=0))
    else:
        outputlist = [
            chop_forward(patch[0], patch[1], patch[2], model, scale, shave, min_size, nGPUs) \
            for patch in inputlist]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    with torch.no_grad():
        output = Variable(x.data.new(b, c, h, w))
    output[:, :, 0:h_half, 0:w_half] \
        = outputlist[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

##Eval Start!!!!
eval(inf_queue)
accumulated = 0 
of_queue_len = time_queue.qsize()
for _ in range(of_queue_len):
  accumulated += time_queue.get() 
print(of_queue_len, "AVG TIME EXTRACTING OF: ", accumulated / of_queue_len)


accumulated_inf = 0
inf_queue_len = inf_queue.qsize()
for _ in range(inf_queue_len):
  item = inf_queue.get() 
  accumulated_inf += item 

print(inf_queue_len, "AVG TIME INFERENCING MODEL: ", accumulated_inf / inf_queue_len)

