import os
import time
import cv2
from collections import defaultdict

from segmentKM import segment, save

if __name__ == '__main__':

    import glob
    import matplotlib.pyplot as plt

    docs = defaultdict(dict)
    for fname in glob.glob('S-ISRI-OCR/scanned_shreds/*'):
        basename = fname.split('/')[-1].split('.')[0]
        side, doc_id = basename.split('_')
        image = cv2.imread(fname)[..., :: -1] # BRG -> RGB
        docs[doc_id][side] = image

    i = 1
    total = len(docs)
    t0 = time.time()
    for doc_id, images in docs.items():
        print('[{}/{}] {}'.format(i, total, doc_id))
        i += 1

        strips, masks = segment(images['front'], images['back'], 50, 0.05, 3, 4, 0.5, 100, 200, 0, nCPU=12)
        save(strips, masks, docname=doc_id, path='S-ISRI-OCR/mechanical')
    print('Elapsed time={:.2f}s'.format(time.time() - t0))
