from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
import torch

from torchvision.transforms import Compose, ToTensor

from flownet_dataset import DatasetFromFolderTest, DatasetFromFolder
#from dataset import DatasetFromFolderTest, DatasetFromFolder

def transform():
    return Compose([
        ToTensor(),
    ])

def get_training_set(data_dir, nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size, future_frame):
    return DatasetFromFolder(data_dir,nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size,future_frame,
                             transform=transform())


def get_eval_set(data_dir, nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size, future_frame):
    return DatasetFromFolder(data_dir,nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size,future_frame,
                             transform=transform())

def get_test_set(data_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame, net, time_queue, args):
    return DatasetFromFolderTest(data_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame, net, time_queue, args, transform=transform())

