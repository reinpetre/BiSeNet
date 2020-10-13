
import sys
sys.path.insert(0, '.')
import argparse
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
import time
import lib.transform_cv2 as T
from lib.models import model_factory
from configs import cfg_factory

torch.set_grad_enabled(False)
np.random.seed(123)


# args
parse = argparse.ArgumentParser()
parse.add_argument('--model', dest='model', type=str, default='bisenetv2',)
parse.add_argument('--weight-path', type=str, default='./res/model_final.pth',)
parse.add_argument('--img-path', dest='img_path', type=str, default='./rs00000.jpg',)
args = parse.parse_args()
cfg = cfg_factory[args.model]


# palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
palette = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)

# define model
net = model_factory[cfg.model_type](19)
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
net.eval()
net.cuda()

# prepare data
to_tensor = T.ToTensor(
    mean=(0.3257, 0.3690, 0.3223), # city, rgb
    std=(0.2112, 0.2148, 0.2115),
)


im = cv2.imread(args.img_path)[:, :, ::-1]
im = cv2.resize(im, (1920, 1024), interpolation = cv2.INTER_AREA)
im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()

#cap = cv2.VideoCapture(args.img_path)
#frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#frame_width = int(cap.get(3))
#frame_weight = int(cap.get(4))

# out = cv2.VideoWriter("./output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

#for i in range(frame_num):
#    ret, image = cap.read() 
#    if ret is False:
#        break
#    start_time = time.time()
#    image = cv2.resize(image, (1920, 1024), interpolation=cv2.INTER_AREA)
    
#    image = to_tensor(dict(im=image, lb=None))['im'].unsqueeze(0).cuda()
    
#    pred = net(image)[0].argmax(dim=1).squeeze().detach().cpu().numpy()
       
#    pred = palette[pred]
#    end = time.time() - start_time
#    print("FPS is:" + str(1/end))
        
    # out.write(pred)
#    cv2.imwrite("./output/res_" + str(i) + ".jpg", pred)

#cap.release()
# out.release()

# inference
start_time = time.time()
out = net(im)[0].argmax(dim=1).squeeze().detach().cpu().numpy()
pred = palette[out]
end = time.time() - start_time
print("FPS is:" + str(1/end))
cv2.imwrite('./res.jpg', pred)
