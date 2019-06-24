# Recurrent Back-Projection Network for Video Super-Resolution (CVPR2019)

The original RBPN implementation was forked from [alterzero/RBPN-PyTorch](https://github.com/alterzero/RBPN-PyTorch).
The codes in this repository uses [NVIDIA/nvll](https://github.com/NVIDIA/nvvl), a library that loads video frames straight on GPUs. Whereas the original implementation of RBPN applies super-resolution(SR) techniques on already extracted video frames, this repository aims to apply the SR technique on raw video files so that low-resolution(LR) videos do not have to go through preprocessing step (extracting frames from videos and saving them on disk). 

Since we want to expedite the computation by doing all the computations on GPU, we won't be using pyflow, which does all the computations on CPU, when extracting optical flows between an input RGB frame and neighboring frames. 
We will instead be using [NVIDIA/FlowNet2.0](https://github.com/NVIDIA/flownet2-pytorch), a pytorch implementation of [FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks](https://arxiv.org/abs/1612.01925). 

This implementation of RBPN spawns three processes excluding the main process. 
The first process extracts optical flows using FlowNet2 and sends the optical flows, neighboring RGB frames and an input RGB frame to a queue. The second process runs RBPN with the item in the queue and sends the SR frames to another queue. The last process receives the SR frames and encode them into a video using opencv and save the video in disk.



## Dependencies
* Python 3.5
* PyTorch >= 1.0.0
* NVVL1.1 -> https://github.com/Haabibi/nvvl
  ```Shell
  # get forked version of NVVL1.1 source
  cd nvvl/pytorch1.0/ 
  python setup.py install
  ```
* FlowNet2.0 -> code adapted from https://github.com/NVIDIA/flownet2-pytorch
  ```Shell
  cd flownet2
  bash install.sh
  ```

## Pretrained Model and Testset
https://drive.google.com/drive/folders/1sI41DH5TUNBKkxRJ-_w5rUf90rN97UFn?usp=sharing

## HOW TO

### Training

    ```python
    main.py
    ```

### Testing

    ```python
    nvvl_eval.py
    ```

![RBPN](https://alterzero.github.io/projects/RBPN.png)

## Dataset
* Vid4 Video: The Vid4 dataset the original RBPN uses contains extracted frames in sequence. Since this repository aims to receive a video as an input, video was encoded from the original frames using ffmpeg.
If you want to use any dataset other than Vid4 videos, you can follow the instructions below to get videos by encoding frames.


### Encoding extracted frames to h264 codec format video 
NVVL1.1 only reads h264 codec format video, and to make LR video to SR, you need to have a video ready.
Adapted from [hamelot.io](http://hamelot.io/visualization/using--to-convert-a-set-of-images-into-a-video/).
```Shell
ffmpeg -r 25 -s 768x512 -i %03d.png -vcodec libx264 -crf 25 test.mp4
```
* `-r` is the framerate (fps)
* `-crf` is the quality, lower means better quality, 15-25 is usally good 
* -s is the resolution
the file will be output to: `test.mp4`

### Converting any video file codec format to h264 
 ```Shell
 INPUT.avi -vcodec libx264 -crf 25 OUTPUT.mp4
 ```
## The paper on Image Super-Resolution
### [Deep Back-Projection Networks for Super-Resolution (CVPR2018)](https://github.com/alterzero/DBPN-Pytorch)
#### Winner (1st) of [NTIRE2018](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w13/Timofte_NTIRE_2018_Challenge_CVPR_2018_paper.pdf) Competition (Track: x8 Bicubic Downsampling)
#### Winner of [PIRM2018](https://arxiv.org/pdf/1809.07517.pdf) (1st on Region 2, 3rd on Region 1, and 5th on Region 3)
#### Project page: https://alterzero.github.io/projects/DBPN.html


## Citations
If you find the original work of Super Resolution useful, please consider citing it.
```
@inproceedings{RBPN2019,
  title={Recurrent Back-Projection Network for Video Super-Resolution},
  author={Haris, Muhammad and Shakhnarovich, Greg and Ukita, Norimichi},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```
