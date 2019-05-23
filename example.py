from rbpn_loader import RBPN_loader
#from r2p1d_loader import loader
from torch.multiprocessing import Queue

frame_queue = Queue()
RBPN_loader( '/cmsdata/ssd0/cmslab/Kinetics-400/sparta/laughing/0gR5FP7HpZ4_000024_000034.mp4', frame_queue, clip_length=7)
#RBPN_loader('/home/haabibi/tmp/RBPN-PyTorch/SPMCS/SPMCS_mp4_vid/gree3_001_truth.mp4', frame_queue)

print("LEN OF FRAME QUEUE: ", frame_queue.qsize())
