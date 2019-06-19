def eval(model, flownet2, frame_queue, vid_frame_queue, opt, sta_bar, fin_bar):
    import torch
    import cv2
    import time 
    from torch.autograd import Variable
    sta_bar.wait()
    model.eval()
    count = 1
    result_list = []
    while True: 
        batch = frame_queue.get()

        input, neighbors, flow, t0 = batch[0], batch[1], batch[2], batch[3]
        if type(input)==list:
            vid_frame_queue.put([])
            break
        else:
            with torch.no_grad():
                input = Variable(input)
                neighbors = [Variable(j) for j in neighbors]
                flow = [Variable(j).float() for j in flow]
                prediction = model(input, neighbors, flow)
                t1 = time.time()         
                
                frame_np = prediction.cpu().data.squeeze().clamp(0,1).numpy().transpose(1,2,0)
                
                # sending frames to queue to save as a video
                vid_frame_queue.put(frame_np)

                # saving img on disk just to check
                save_img(frame_np, str(count), True, opt)
                print("===> Processing: %s || Timer: %.4f sec." % (str(count), (t1-t0)))
                count += 1

    fin_bar.wait()
    frame_queue.cancel_join_thread()
    vid_frame_queue.cancel_join_thread()

def save_vid(vid_frame_queue, sta_bar, fin_bar):
    import cv2
    sta_bar.wait()
    fourcc =  cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('./sample.mp4', fourcc, 25, (512, 768))
    frame_list = []
    
    while True:
        if vid_frame_queue.qsize() > 0:
            frame_np = vid_frame_queue.get()
            if frame_np == []:
                break
            frame_list.append(frame_np) 
    
    for frame in frame_list:
        frame = cv2.cvtColor(frame*255, cv2.COLOR_BGR2RGB) 
        frame = frame.astype('uint8')
        video.write(frame)
    video.release()
    
    fin_bar.wait()
    vid_frame_queue.cancel_join_thread()

def save_img(img, img_name, pred_flag, opt):
    import os 
    import torch
    import cv2
    import torchvision.transforms.functional as F

    # img.shape ==> [1, 3, 512, 768]
    #save_img = img.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
    # save_img.shape ==> (512, 768, 3)

    # save img
    save_dir=os.path.join(opt.output, opt.data_dir, os.path.splitext(opt.file_list)[0]+'_'+str(opt.upscale_factor)+'x')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    if pred_flag:
        save_fn = save_dir +'/'+ img_name+'_'+opt.model_type+'F'+str(opt.nFrames)+'.png'
    else:
        save_fn = save_dir +'/'+ img_name+'.png'
    
    cv2.imwrite(save_fn, cv2.cvtColor(img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
