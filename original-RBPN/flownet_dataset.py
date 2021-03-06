import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
from skimage import img_as_float
from skimage.transform import resize
from random import randrange
import os.path
from flownet2.models import FlowNet2
from torch.autograd import Variable
import time

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath, nFrames, scale, other_dataset):
    seq = [i for i in range(1, nFrames)]
    #random.shuffle(seq) #if random sequence
    if other_dataset:
        target = modcrop(Image.open(filepath).convert('RGB'),scale)
        input=target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        
        char_len = len(filepath)
        neigbor=[]

        for i in seq:
            index = int(filepath[char_len-7:char_len-4])-i
            file_name=filepath[0:char_len-7]+'{0:03d}'.format(index)+'.png'
            if os.path.exists(file_name):
                temp = modcrop(Image.open(filepath[0:char_len-7]+'{0:03d}'.format(index)+'.png').convert('RGB'),scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
                neigbor.append(temp)
            else:
                print('neigbor frame is not exist')
                temp = input
                neigbor.append(temp)
    else:
        target = modcrop(Image.open(join(filepath,'im'+str(nFrames)+'.png')).convert('RGB'), scale)
        input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        neigbor = [modcrop(Image.open(filepath+'/im'+str(j)+'.png').convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC) for j in reversed(seq)]
    
    return target, input, neigbor

def load_img_future(filepath, nFrames, scale, other_dataset):
    tt = int(nFrames/2)
    if other_dataset:
        #target = modcrop(Image.open(filepath).convert('RGB'),scale)
        target = Image.open(filepath).convert('RGB')
        target = target.resize((768, 512))
        input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        #input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)))
        #input = target.resize(( 192, 128), Image.BICUBIC)
        char_len = len(filepath)
        neigbor=[]
        if nFrames%2 == 0:
            seq = [x for x in range(-tt,tt) if x!=0] # or seq = [x for x in range(-tt+1,tt+1) if x!=0]
        else:
            seq = [x for x in range(-tt,tt+1) if x!=0]
        #random.shuffle(seq) #if random sequence
        filename_list = [] 
        for i in seq:
            index1 = int(filepath[char_len-7:char_len-4])+i
            file_name1=filepath[0:char_len-7]+'{0:03d}'.format(index1)+'.png'
            if os.path.exists(file_name1):
                temp = modcrop(Image.open(file_name1).convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
                #temp = modcrop(Image.open(file_name1).convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)))
                print("LEMME SEET HIS: ", temp)
                #temp = modcrop(Image.open(file_name1).convert('RGB'), scale).resize((192, 128), Image.BICUBIC)
                neigbor.append(temp)
                filename_list.append(file_name1)
            else:
                print('neigbor frame- is not exist')
                temp=input
                neigbor.append(temp)
                filename_list.append(filepath)
        print("FILE NAME LIST : ", filename_list)
        filename_list = []
    else:
        target = modcrop(Image.open(join(filepath,'im4.png')).convert('RGB'),scale)
        input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        neigbor = []
        seq = [x for x in range(4-tt,5+tt) if x!=4]
        #random.shuffle(seq) #if random sequence
        for j in seq:
            neigbor.append(modcrop(Image.open(filepath+'/im'+str(j)+'.png').convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC))
    return target, input, neigbor

def get_flow(im1, im2, net, args, time_queue):
    #im1 = resize(np.array(im1), (128, 192), anti_aliasing=False)
    #im2 = resize(np.array(im2), (128, 192), anti_aliasing=False)
    im1 = np.array(im1)
    im2 = np.array(im2)
    ims = np.array([[im1, im2]]).transpose((0, 4, 1, 2, 3)).astype(np.float32)
    #DATASET L92 IM1.shape: (128, 192, 3) Concat shape: (1, 2, 128, 192, 3) <class 'numpy.ndarray'>
    #DATASET L94 TRanspose shape:  (1, 3, 2, 128, 192)
    ims = torch.from_numpy(ims)
    #torch.cuda.init()
    ims_v = Variable(ims.cuda(), requires_grad=False)
    
    t1= time.time()
    pred_flow = net(ims_v).squeeze()
    t2 = time.time()
    time_queue.put(t2-t1)
    return pred_flow

def rescale_flow(x,max_range,min_range):
    max_val = np.max(x)
    min_val = np.min(x)
    return (max_range-min_range)/(max_val-min_val)*(x-max_val)+max_range

def modcrop(img, modulo):
    (ih, iw) = img.size
    ih = ih - (ih%modulo);
    iw = iw - (iw%modulo);
    img = img.crop((0, 0, ih, iw))
    return img

def get_patch(img_in, img_tar, img_nn, patch_size, scale, nFrames, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale #if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy,ix,iy + ip, ix + ip))#[:, iy:iy + ip, ix:ix + ip]
    img_tar = img_tar.crop((ty,tx,ty + tp, tx + tp))#[:, ty:ty + tp, tx:tx + tp]
    img_nn = [j.crop((iy,ix,iy + ip, ix + ip)) for j in img_nn] #[:, iy:iy + ip, ix:ix + ip]
                
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, img_nn, info_patch

def augment(img_in, img_tar, img_nn, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        img_nn = [ImageOps.flip(j) for j in img_nn]
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            img_nn = [ImageOps.mirror(j) for j in img_nn]
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            img_nn = [j.rotate(180) for j in img_nn]
            info_aug['trans'] = True

    return img_in, img_tar, img_nn, info_aug
    
def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir,nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size, future_frame, transform=None):
        super(DatasetFromFolder, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir,file_list))]
        self.image_filenames = [join(image_dir,x) for x in alist]
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.other_dataset = other_dataset
        self.patch_size = patch_size
        self.future_frame = future_frame

    def __getitem__(self, index):
        if self.future_frame:
            target, input, neigbor = load_img_future(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset)
        else:
            target, input, neigbor = load_img(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset)

        if self.patch_size != 0:
            input, target, neigbor, _ = get_patch(input,target,neigbor,self.patch_size, self.upscale_factor, self.nFrames)
        
        if self.data_augmentation:
            input, target, neigbor, _ = augment(input, target, neigbor)
            
        flow = [get_flow(input,j) for j in neigbor]
            
        bicubic = rescale_img(input, self.upscale_factor)
        
        if self.transform:
            target = self.transform(target)
            input = self.transform(input)*255
            bicubic = self.transform(bicubic)
            neigbor = [self.transform(j)*255 for j in neigbor]
            flow = [torch.from_numpy(j.transpose(2,0,1))*255 for j in flow]

        return input, target, neigbor, flow, bicubic

    def __len__(self):
        return len(self.image_filenames)

class DatasetFromFolderTest(data.Dataset):
    def __init__(self, image_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame, net, time_queue, args, transform=None):
        super(DatasetFromFolderTest, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir,file_list))]
        self.image_filenames = [join(image_dir,x) for x in alist][:15]
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.other_dataset = other_dataset
        self.future_frame = future_frame
        self.args = args
        self.flownet = net
        self.time_queue = time_queue 

    def __getitem__(self, index):
        if self.future_frame:
            target, input, neigbor = load_img_future(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset)
        else:
            target, input, neigbor = load_img(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset)
        
        flow = []
        time_flow = []
        flow_t_origin = time.time()
        for j in neigbor:
          flow_t = time.time()
          flow.append(get_flow(input, j, self.flownet, self.args, self.time_queue))
          time_flow.append(time.time() - flow_t)
        print("TIME TO EXTRACT FLOWS: ", time.time() - flow_t_origin, time_flow)  
        
        #neigbor_tensor = torch.stack(neigbor, dim=1)
        #print("[NVVL] FlowNet INPUT of neighbors: ", neigbor_tensor.is_cuda, neigbor_tensor.shape, neigbor_tensor.type(), torch.mean(neigbor_tensor), torch.max(neigbor_tensor))

        flow_tensor = torch.stack(flow, dim=1)
        print("[NVVL] FlowNet OUTPUT: ", flow_tensor.is_cuda, flow_tensor.shape, flow_tensor.type(), torch.mean(flow_tensor), torch.max(flow_tensor), torch.min(flow_tensor))

        bicubic = rescale_img(input, self.upscale_factor)
        
        print("INPUT LINE243: ", type(input), input)
        if self.transform:
            target = self.transform(target)
            input = self.transform(input)
            bicubic = self.transform(bicubic)
            neigbor = [self.transform(j) for j in neigbor]
            #flow = [torch.from_numpy(j.transpose(2,0,1)) for j in flow]
            print("||||||NEIGBOR INPUT||||||", neigbor[0].shape, type(neigbor[0]), torch.mean(neigbor[0]), torch.max(neigbor[0]), torch.min(neigbor[0]))
            #print("NEIGHBOR", neigbor[0].shape, type(neigbor[0]), neigbor[0])
            #print("FLOW", flow[0].shape, type(flow[0]), flow[0])
            
        return input, target, neigbor, flow, bicubic
      
    def __len__(self):
        return len(self.image_filenames)
