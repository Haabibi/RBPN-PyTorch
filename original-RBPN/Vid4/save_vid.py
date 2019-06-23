import cv2
import os 


fourcc = cv2.VideoWriter_fourcc(*'DIVX')
frame_array = []
for frame in os.listdir('./resized_new_city'):
    path = './resized_new_city/'+frame
    img = cv2.imread(path)
    height, width, layers = img.shape
    size = (width, height)
    frame_array.append(img)

out = cv2.VideoWriter('./test.avi', fourcc, 25.0, size)
    

for i in range(len(frame_array)):
    out.write(frame_array[i])

    #video.write(img)



out.release()
