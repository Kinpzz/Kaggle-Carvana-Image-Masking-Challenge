from model.u_net import get_unet_128, get_unet_256, get_unet_512, get_unet_1024, get_unet_1280

input_size = 1024

max_epochs = 100
batch_size = 5
test_batch_size = 12

orig_width = 1918
orig_height = 1280

threshold = 0.5

model_factory = get_unet_1024
model_factory_1280 = get_unet_1280
import numpy as np

def getPoint(percent=1, phase=0):
    if percent < 0.8:
        print "[x] error percentage..."
    if phase == 0:
        print "begin phase left"
    elif phase == 1:
        print "begin phase right"
    else:
        print "[x] error phase..."
    width = np.round(orig_width * percent)
    height = np.round(orig_height * percent)
    rmin = ((height - 1024) / 2).astype(int)
    rmax = (1024 + rmin).astype(int)
    cmin = (phase * (width - 1024)).astype(int)
    cmax = (1024 + cmin).astype(int)

    return height.astype(int), width.astype(int), rmin, rmax, cmin, cmax
    
