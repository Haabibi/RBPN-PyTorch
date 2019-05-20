import cv2
import numpy as np
import os
from os.path import isfile, join 

def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = sorted([f for f in os.listdir(pathIn) if isfile(join(pathIn, f))])
    for a_file in files:
        img = cv2.imread(os.path.join(pathIn, a_file))
        height, width, layers = img.shape
        size = (width,height)
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
    # writing to a image array
        out.write(frame_array[i])
    out.release()


def main():
    pathIn= './Vid4/walk'
    pathOut = './Vid4/video/walk.avi'
    fps = 25.0
    convert_frames_to_video(pathIn, pathOut, fps)

if __name__=="__main__":
    main()
         
