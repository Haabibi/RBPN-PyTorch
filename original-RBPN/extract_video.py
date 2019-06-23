import cv2

cap = cv2.VideoCapture('/cmsdata/ssd0/cmslab/Kinetics-400/sparta/laughing/0gR5FP7HpZ4_000024_000034.mp4')
i=0
while (cap.isOpened()):
  ret, frame = cap.read()
  if ret == False:
    print("THIS IS WHERE IT BROKE: ", i)
    break
  #cv2.imwrite('./frames/kinetics_lauging_{:03}'.format(i) + '.jpg', frame)
  i+=1

cap.release()
cv2.destroyAllWindows()
