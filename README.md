# Recurrent Back-Projection Network for Video Super-Resolution (CVPR2019)

Project page: https://alterzero.github.io/projects/RBPN.html

The original RBPN implementation was forked from [alterzero/RBPN-PyTorch](https://github.com/alterzero/RBPN-PyTorch).
The codes in this repository uses [NVIDIA/nvll](https://github.com/NVIDIA/nvvl), a library that loads video frames straight on GPUs. Whereas the original implementation of RBPN applies super-resolution(SR) techniques on already extracted video frames, this repository aims to apply the SR technique on raw video files so that low-resolution(LR) videos do not have to go through preprocessing step (frame extraction and saving them on disk). 

## Dependencies
* Python 3.5
* PyTorch >= 1.0.0
* Pyflow -> https://github.com/pathak22/pyflow
  ```Shell
  cd pyflow/
  python setup.py build_ext -i
  cp pyflow*.so ..
  ```
* NVVL1.1 -> https://github.com/Haabibi/nvvl
  ```Shell
  cd nvvl/pytorch1.0/ 
  python setup.py install
  ```

## Dataset
* [Vimeo-90k Dataset](http://toflow.csail.mit.edu)

## Pretrained Model and Testset
https://drive.google.com/drive/folders/1sI41DH5TUNBKkxRJ-_w5rUf90rN97UFn?usp=sharing

## HOW TO

#Training

    ```python
    main.py
    ```

#Testing

    ```python
    eval.py
    ```

![RBPN](https://alterzero.github.io/projects/RBPN.png)

#Encoding extracted frames to h264 codec format video 
NVVL1.1 only reads h264 codec format video, and to make LR video to SR, you need to have a video ready.
Adapted from [hamelot.io](http://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/).
```Shell
ffmpeg -r 25 -s 768x512 -i %03d.png -vcodec libx264 -crf 25 test.mp4
```
* `-r` is the framerate (fps)
* `-crf` is the quality, lower means better quality, 15-25 is usally good 
* -s is the resolution
the file will be output to: `test.mp4`

#Converting any video file codec format to h264 
 ```Shell
 INPUT.avi -vcodec libx264 -crf 25 OUTPUT.mp4
 ```
## The paper on Image Super-Resolution
### [Deep Back-Projection Networks for Super-Resolution (CVPR2018)](https://github.com/alterzero/DBPN-Pytorch)
#### Winner (1st) of [NTIRE2018](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w13/Timofte_NTIRE_2018_Challenge_CVPR_2018_paper.pdf) Competition (Track: x8 Bicubic Downsampling)
#### Winner of [PIRM2018](https://arxiv.org/pdf/1809.07517.pdf) (1st on Region 2, 3rd on Region 1, and 5th on Region 3)
#### Project page: https://alterzero.github.io/projects/DBPN.html


## Citations
If you find this work useful, please consider citing it.
```
@inproceedings{RBPN2019,
  title={Recurrent Back-Projection Network for Video Super-Resolution},
  author={Haris, Muhammad and Shakhnarovich, Greg and Ukita, Norimichi},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```
