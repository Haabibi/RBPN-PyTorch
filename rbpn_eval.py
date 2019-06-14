def eval(model, flownet2, frame_queue, opt, sta_bar, fin_bar):
    import torch
    import time 
    from torch.autograd import Variable
    sta_bar.wait()
    model.eval()
    count = 1
    while True: 
        batch = frame_queue.get()

        input, neighbors, flow = batch[0], batch[1], batch[2]

        with torch.no_grad():
            input = Variable(input)
            neighbors = [Variable(j) for j in neighbors]
            flow = [Variable(j).float() for j in flow]
            

            '''
            neighbors_ = torch.stack(neighbors, dim=1) 
            flow_ = torch.stack(flow, dim=1)
            print("[NVVL {}] RBPN Input of target frame: ".format(count), input.shape, input.type(), torch.mean(input), torch.max(input), torch.min(input)) 
            print("[NVVL {}] RBPN Input of neighbors: ".format(count), neighbors_.shape, neighbors_.type(), torch.mean(neighbors_), torch.max(neighbors_), torch.min(neighbors_)) 
            print("[NVVL {}] RBPN Input of flows: ".format(count), flow_.shape, flow_.type(), torch.mean(flow_), torch.max(flow_), torch.min(flow_)) 
            print("[[[[[[[[L18]]]]]]]]", neighbors[0].shape, flow[0].shape)
            '''
            t0 = time.time()
            prediction = model(input, neighbors, flow)
            t1 = time.time()         
            print("[NVVL] RBPN Output: ", prediction.shape, prediction.type(), torch.mean(prediction))
            save_img(prediction.cpu().data, str(count), True, opt)
            print("===> Processing: %s || Timer: %.4f sec." % (str(count), (t1-t0)))

        count+=1
    fin_bar.wait()

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

