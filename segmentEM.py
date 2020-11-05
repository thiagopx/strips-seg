import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from skimage import measure
import matplotlib.patches as patches
import os
import sys
from sklearn import mixture


fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)

window = [None, None]
background = [None]
image = [None]
cnt = [0]

# Path to dataset
# assert len(sys.argv) > 2

#src_path = sys.argv[1]
#tgt_path = sys.argv[2]
#files = os.listdir(src_path)
#ext = os.path.splitext(files[0])[1]
#docs = list(set([os.path.splitext(file_)[0].split('_')[1] for file_ in files]))

def update_image(d=30):
    doc = docs[cnt[0]]

    #print 'Processing %s' % doc
    sys.stdout.flush()

    filename = os.path.join(src_path, 'front_%s%s' % (doc, ext))
    front = cv2.imread(filename)[:, : -d]
    filename = os.path.join(src_path, 'back_%s%s' % (doc, ext))
    back = cv2.imread(filename)[:, d : ]
    bgr = np.hstack((front, back))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.GaussianBlur(rgb, (5, 5), 0)
    image[0] = rgb

def save(strips, masks):
    doc = docs[cnt[0]]
    strips_path = os.path.join(tgt_path, doc, 'strips')
    masks_path = os.path.join(tgt_path, doc, 'masks')
    for path in [strips_path, masks_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    for i in range(len(strips)):
        # print 'Saving strip %d' % i
        strip = strips[i]
        mask = masks[i]
        basename = '%s%02d%s' % (doc, i + 1, ext)
        filename = os.path.join(strips_path, basename)
        cv2.imwrite(filename, strip)
        basename = '%s%02d%s' % (doc, i + 1, '.npy')
        filename = os.path.join(masks_path, basename)
        np.save(filename, mask)

def plot():
    ax.clear()
    ax.axis('off')
    ax.imshow(image[0], aspect='auto')
    if None not in window:
        ax.add_patch(
            patches.Rectangle(
                window[0],                   # (x,y)
                window[1][0] - window[0][0], # width
                window[1][1] - window[0][1], # height
                facecolor='none',
                edgecolor='red',
                linestyle='dashed',
                linewidth=0.2
            )
        )
    doc = docs[cnt[0]]
    ax.set_title(doc)
    fig.canvas.draw()

def press(event):
    if event.key.lower() == 'r':
        if None not in window + background:
            strips, masks = segment()
            save(strips, masks)

    elif event.key == 'b':
        background[0] = (int(event.xdata), int(event.ydata))
        plot()
    elif event.key == '1' and event.inaxes is not None: # top-left
        window[0] = (int(event.xdata), int(event.ydata))
        plot()
    elif event.key == '2' and event.inaxes is not None: # bottom-right
        window[1] = (int(event.xdata), int(event.ydata))
        plot()

    elif event.key == 'right':
        cnt[0] = min(cnt[0] + 1, len(docs) - 1)
        update_image()
        plot()
    elif event.key == 'left':
        cnt[0] = max(cnt[0] - 1, 0)
        update_image()
        plot()

    #print window
    #print background
#    draw

def quantize(image, perc=0.05, n_colors=3):
    '''
    Adapted from:
    http://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html
    '''
    norm = np.array(image, dtype=np.float64) / 255
    h, w, d = norm.shape
    assert d == 3
    image_array = np.reshape(norm, (w * h, d))
    sample = shuffle(image_array, random_state=0)[:int(perc * h * w)]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(sample)
    labels = kmeans.predict(image_array)
    colors = (255 * kmeans.cluster_centers_).astype(np.uint8)
    labeled = labels.reshape(h, w)
    quantized = colors[labels].reshape(h, w, d)
    return quantized, labeled, colors

def segment(image, bg_sample):
    strips = []
    masks = []
    x_min, y_min = window[0]
    x_max, y_max = window[1]
    #print x_min, y_min, x_max, y_max
    cropped = image[0]
    cropped = cropped[y_min : y_max + 1, x_min : x_max + 1]
    #print 'Quantization... ',
    sys.stdout.flush()
    quantized, labels, colors = quantize(cropped)
    #print 'done!'
    sys.stdout.flush()
    label = labels[background[0][1] - y_min, background[0][0] - x_min]
    #print label
    bg = (labels == label)

    #print 'Remove small regions... ',
    sys.stdout.flush()
    # Remove small regions
    labels = measure.label(bg)
    props = measure.regionprops(labels)
    for region in props:
        if region.area < 500:
            y, x = np.hsplit(region.coords, 2)
            bg[y, x] = False
    #print 'done!'
    sys.stdout.flush()

    #print 'Foreground segmentation... ',
    sys.stdout.flush()
    # Foreground segmentation
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    fg = (255 * np.logical_not(bg)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, rect)
    fg = cv2.morphologyEx(fg, cv2.MORPH_ERODE, rect)
    temp = np.zeros_like(cropped)
    labels = measure.label(fg)
    props = measure.regionprops(labels)
    for region in props:
        y, x = np.hsplit(region.coords, 2)
        min_row, min_col, max_row, max_col = region.bbox
        temp[y, x] = cropped[y, x]
        strip = temp[min_row : max_row, min_col : max_col].copy()
        strips.append(strip)
        masks.append((255 * region.image).astype(np.uint8))
        temp[y, x, :] = 0 # restore black
    #print('done!')
    sys.stdout.flush()
    return strips, masks

if __name__ == '__main__':

#fig.canvas.mpl_connect('key_press_event', press)
    front = cv2.imread('front_D008.jpeg')[: 1000, : 1000]
    #back = cv2.imread('back_D008.jpeg')
    image = front
    bg_sample = cv2.imread('bg_sample.jpeg')
    h, w, d = front.shape
    assert front.ndim == 3

    window_size = 3
    W = w - window_size + 1
    H = h - window_size + 1

    images= []
    for dx in range(window_size):
        for dy in range(window_size):
            images.append(image[dy : H + dy, dx : W + dx])
    data = np.stack(images).transpose((1, 2, 0, 3))
    data_lin = np.reshape(data, (-1, 3 * (window_size) ** 2))

    # EM
    #gmm = mixture.GaussianMixture(n_components=3, covariance_type='spherical', max_iter=1)
    #gmm.fit(data_lin)
    #predicted = gmm.predict(data_lin)
    #clusters = np.reshape(, (H, -1))

    # KMeans
    kmeans = KMeans(n_clusters=3, random_state=0).fit(data_lin)
    labels = kmeans.predict(data_lin)
    #colors = (255 * kmeans.cluster_centers_).astype(np.uint8)
    predicted = labels.reshape((H, -1))
    #quantized = colors[labels].reshape(h, w, d)


    #mask = np.zeros_like(clusters)
    bg_color = bg_sample.reshape((-1, 3)).mean(axis=0)
    bg_label = 0
    dist = float('inf')
    for label in np.unique(predicted):
        label_color = data_lin[predicted == label].reshape((-1, 3)).mean(axis=0)
        label_dist = np.sum((label_color - bg_color) ** 2)
        if label_dist < dist:
            bg_label = label
            dist = label_dist
        print(label_color, bg_color, label_dist, dist)
        #print(labeled)

        #print(mean_color)

    mask = (predicted.reshape((H, -1)) == bg_label)
    #np.where(predicted == bg_label)
    #mask[]
     #data_lin[predicted == label].reshape((H, -1))

    plt.imshow(mask)
    #print(clusters.shape)
    plt.show()


    #update_image()
    #plot()

