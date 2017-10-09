import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import params

input_size = params.input_size
batch_size = params.test_batch_size
orig_width = params.orig_width
orig_height = params.orig_height
threshold = params.threshold
model = params.model_factory()

#df_test = pd.read_csv('input/sample_submission.csv')
#ids_test = df_test['img'].map(lambda s: s.split('.')[0])

DATA_PATH='/media/Disk/yanpengxiang/dataset/carvana/'
test_list = open('list/test_hard.txt')
df_test = test_list.readlines()
ids_test = []
for i in xrange(len(df_test)):
    ids_test.append(df_test[i][:15])
ids_test = np.array(ids_test)

names = []
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


rles = []

model.load_weights(filepath='weights/save/acc09968.hdf5')

print('Predicting on {} samples with batch_size = {}...'.format(len(ids_test), batch_size))
for start in tqdm(range(0, len(ids_test), batch_size)):
    ids_test_batch_name = []
    x_batch = []
    for i in range(2):
        end = min(start + (i+1)*batch_size/2, len(ids_test))
        ids_test_batch = ids_test[start + i*batch_size/2:end]
        for id in xrange(len(ids_test_batch)):
            img = cv2.imread((DATA_PATH+'test/{}.jpg').format(ids_test_batch[id]))
            if input_size is not None:
                #img = cv2.resize(img, (input_size, input_size))
                if i == 0:
                    img = img[128:1152,:1024]
                    ids_test_batch_name.append(ids_test_batch[id] + '_01')
                elif i == 1:
                    img = img[128:1152,894:]
                    ids_test_batch_name.append(ids_test_batch[id] + '_02')
            x_batch.append(img)
    print np.array(x_batch).shape
    preds = model.predict_on_batch(x_batch)
    preds = np.squeeze(preds, axis=3)
    for i,pred in enumerate(preds):
        prob = pred
        mask = np.zeros_like(prob)
        mask[prob > threshold] = 255
        id = ids[i]
        cv2.imwrite('/media/Disk/yanpengxiang/Unet/test_hard/' + id + '_mask.png', mask)
#print("Generating submission file...")
#df = pd.DataFrame({'img': names, 'rle_mask': rles})
#df.to_csv('submit/submission.csv.gz', index=False, compression='gzip')
