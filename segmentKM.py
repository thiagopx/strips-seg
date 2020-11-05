import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool
from sklearn.cluster import KMeans
from skimage import measure
from scipy import ndimage as ndi
from multiprocessing import Pool


def save(strips, masks, docname, path='.', ext='jpg'):

    strips_path = '{}/{}/strips'.format(path, docname)
    masks_path = '{}/{}/mask'.format(path, docname)
    for path in [strips_path, masks_path]:
        if not os.path.exists(path):
            os.makedirs(path)
    for i in range(len(strips)):
        strip = strips[i]
        mask = masks[i]
        filename = '{}/{}{:02d}.{}'.format(strips_path, docname, i + 1, ext)
        cv2.imwrite(filename, strip[:, :, :: -1]) # RGB => BGR
        filename = '{}/{}{:02d}.npy'.format(masks_path, docname, i + 1)
        np.save(filename, mask)


def segment(
    front, back, dh=50, bg_sample_perc=0.05, n_classes=3, sample_factor=4,
    min_area_perc=0.5, min_area=100, n_points=200, seed=0, nCPU=4
):

    # vertical crop
    front = front[dh : -dh]
    back = back[dh : -dh]

    # extract background sample
    h, w, _ = back.shape
    bg_sample = back[:, - int(bg_sample_perc * w) :]

    # train data
    sampled = front[:: sample_factor, :: sample_factor]
    train = sampled.reshape(-1, 3)

    # test data
    test_front = front.reshape(-1, 3)
    test_back = back.reshape(-1, 3)

    # kmeans clustering
    bg_color = np.median(bg_sample.reshape((-1, 3)), axis=0)
    centers = np.array([bg_color, [0, 0, 0], [255, 255, 255]])
    kmeans = KMeans(n_clusters=n_classes, n_init=1, random_state=seed, init=centers).fit(train)

    # segmentation
    bg_label = 0
    strips = []
    masks = []
    mean_area = -1
    for test, image in zip([test_front, test_back], [front, back]):
        h, w, _ = image.shape

        # not working in sklearn 0.23
        tests = np.array_split(test, nCPU)
        # with Pool(nCPU) as p: # parallel computation of prediction
            # labels_list = p.map(kmeans.predict, tests)
        # labels = np.concatenate(labels_list)
        # ===============
        labels = kmeans.predict(test)

        # mask
        fg_mask = (labels != bg_label).reshape((h, -1))
        fg_mask = ndi.binary_fill_holes(fg_mask)

        # background margins
        # import matplotlib.pyplot as plt
        #plt.imshow(fg_mask, cmap='gray')
        #plt.show()

        # find margins
        y_data = np.linspace(0, h - 1, n_points).astype(np.int32)
        x_left_data = []
        x_right_data = []
        for y in y_data:
            x = 0
            while fg_mask[y, x]:
                x += 1
            x_left_data.append(x)
            x = w - 1
            while fg_mask[y, x]:
                x -= 1
            x_right_data.append(x)

        poly_left = np.poly1d(np.polyfit(y_data, x_left_data, 1))
        poly_right = np.poly1d(np.polyfit(y_data, x_right_data, 1))

        # fill side margins
        for y in range(h):
            fg_mask[y, : int(poly_left(y))] = False
            fg_mask[y, int(poly_right(y)) :] = False

        #plt.imshow(fg_mask, cmap='gray')
        #plt.show()

        # extract strips
        labels = measure.label(fg_mask)
        props = measure.regionprops(labels)
        if mean_area == -1:
            mean_area = np.mean([region.area for region in props if region.area > min_area])

        for region in props:
            if region.area > (min_area_perc * mean_area): # remove small regions
                min_row, min_col, max_row, max_col = region.bbox
                y, x = np.hsplit(region.coords, 2)
                mask = (255 * region.image).astype(np.uint8)
                hm, wm = mask.shape
                strip = np.zeros((hm, wm, 3), dtype=np.uint8)
                strip[y - min_row, x - min_col] = image[y, x]
                masks.append(mask)
                strips.append(strip)
    return strips, masks


if __name__ == '__main__':

    import glob
    import matplotlib.pyplot as plt
    front_fnames = []
    back_fnames = []
    ids = []

    for fname in glob.glob('test/anotacao_*.txt'):
        for line in open(fname):
            names = line.split()
            front_fname = '/'.join(names[0].split('/')[-3 :])
            back_fname = '/'.join(names[1].split('/')[-3 :])
            id_ = names[0].split('/')[-2]
            front_fnames.append('test/{}'.format(front_fname))
            back_fnames.append('test/{}'.format(back_fname))
            #print(line, id_)
            print(front_fname, back_fname)

    i = 1
    total = len(front_fnames)
    t0 = time.time()
    for front_fname, back_fname in zip(front_fnames, back_fnames):
#        if 'gfu71f00' not in front_fname:
#            continue
        print('[{}/{}] {}'.format(i, total, front_fname))
        i += 1
        front = cv2.imread(front_fname)[:, :, :: -1]
        back = cv2.imread(back_fname)[:, :, :: -1]

        # run in background
        pool = ThreadPool(processes=1)
        #result = pool.apply_async(segment, (front, back, 50, 0.1, 3, 4, 2, 0, 100, 4))
        strips, masks = segment(front, back, 50, 0.05, 3, 4, 0.5, 100, 200, 0, 4)
        # do whatever your need to do
        # strips, masks = result.get()
        id_ = front_fname.split('/')[-2]
        print(id_, len(strips))
        save(strips, masks, docname=id_, path='results')
    print('Elapsed time={:.2f}s'.format(time.time() - t0))
