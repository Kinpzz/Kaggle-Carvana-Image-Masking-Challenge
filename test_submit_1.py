import cv2
import numpy as np
import pandas as pd
import threading
import Queue
import tensorflow as tf
from tensorflow.python.client import device_lib
from tqdm import tqdm

import params
import os

input_size = params.input_size
batch_size = params.test_batch_size
orig_width = params.orig_width
orig_height = params.orig_height
threshold = params.threshold
model_factory = params.model_factory

# phase and percentage config
PHASE = 2
#PERCENT = 1.1
#new_height, new_width, rmin, rmax, cmin, cmax = params.getPoint(PERCENT, PHASE)
if PHASE == 1:
    rmin = 0
    rmax = 640
    cmin = 0
    cmax = 1024
elif PHASE == 2:
    rmin = 0
    rmax = 640
    cmin = 894
    cmax = 1918
elif PHASE == 3:
    rmin = 640
    rmax = 1280
    cmin = 0
    cmax = 1024
elif PHASE == 4:
    rmin = 640
    rmax = 1280
    cmin = 894
    cmax = 1918

# gpu setting
gpu_id = '2'
os.environ['CUDA_VISIBLE_DEVICES']=str(gpu_id)
gpus = [x.name for x in device_lib.list_local_devices() if x.name[:4] == '/gpu']

# path setting
DATA_PATH = '/media/Disk/yanpengxiang/dataset/carvana/'
SAVE_PATH = '/media/Disk/yanpengxiang/Unet/output/1024_4/'
if os.path.isdir(SAVE_PATH) == False:
    os.makedirs(SAVE_PATH)


df_test = pd.read_csv('/media/Disk/yanpengxiang/dataset/sample_submission.csv')
ids_test = np.array(df_test['img'].map(lambda s: s.split('.')[0]).values)

## read txt ## 
# test_list = open('list/test_hard.txt')
# df_test = test_list.readlines()
# ids_test = []
# for i in xrange(len(df_test)):
#    ids_test.append(df_test[i][:15])
# ids_test = np.array(ids_test)

names = []
rles = []
q_size = 10

for id in ids_test:
    names.append('{}.jpg'.format(id))


# https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle

def create_model(gpu):
    with tf.device(gpu):
        model = model_factory()
    model.load_weights(filepath='weights/best_weights_up.hdf5')
    return model

def data_loader(q, ):

    for start in tqdm(range(0, len(ids_test), batch_size)):
        x_batch = []
        end = min(start + batch_size, len(ids_test))
        ids_test_batch = ids_test[start:end]
        for id in ids_test_batch:
            img = cv2.imread((DATA_PATH+'test/{}.jpg').format(id))
            if input_size is not None:
                #if new_width != orig_width:
                #   img = cv2.resize(img, (new_width, new_height))
                img = img[rmin:rmax, cmin:cmax]
                img = cv2.resize(img, (1024, 1024))
            x_batch.append(img)
        x_batch = np.array(x_batch, np.float32) / 255
        q.put((ids_test_batch, x_batch))
    for g in gpus:
        q.put((None, None))

def predictor(q, gpu):
    PHASE_CHAR = str(PHASE)
    config = tf.ConfigProto()
    config.allow_soft_placement=True
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    with sess.as_default():
        model = create_model(gpu)
        while True:
            ids, x_batch = q.get()
            if ids is None:
                break
            preds = model.predict_on_batch(x_batch)
            preds = np.squeeze(preds, axis=3)
            for i,pred in enumerate(preds):
                if input_size is not None:
                    prob = pred
                    #mask = np.zeros_like(prob)
                    #mask[prob>threshold] = 255
                    mask = prob * 255
                    mask = cv2.resize(mask, (1024,640))
                    id = ids[i]
                    cv2.imwrite(SAVE_PATH + id + '_mask_' + PHASE_CHAR + '.png', mask)

print('Predicting on {} samples with batch_size = {}...'.format(len(ids_test), batch_size))
q = Queue.Queue(maxsize=q_size)

threads = []
threads.append(threading.Thread(target=data_loader, name='DataLoader', args=(q,)))
threads[0].start()
for gpu in gpus:
     print("Starting predictor at device " + gpu)
     t = threading.Thread(target=predictor, name='Predictor', args=(q, gpu))
     threads.append(t)
     t.start()

# Wait for all threads to finish
for t in threads:
    t.join()

#print("Generating submission file...")
#df = pd.DataFrame(rles, columns=['img', 'rle_mask'])
#df['img'] += '.jpg'
#df.to_csv('submit/submission.csv.gz', index=False, compression='gzip')
