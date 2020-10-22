import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from skimage import feature


class IndexTracker(object):
    def __init__(self, ax, volume, segmentation=None, title="", show_contour=False):
        self.volume = volume
        self.segmentation = segmentation
        if len(volume.shape) == 4:
            rows, cols, channels, self.slices = volume.shape
        elif len(volume.shape) == 3:
            rows, cols, self.slices = volume.shape
        self.ind = self.slices // 2
        self.ax = ax
        self.title = title
        if segmentation is not None:
            self.im = self.ax[0].imshow(self.volume[:, :, self.ind], vmin=0, vmax=1)
            if title != "":
                self.ax[0].set_title(self.title)
            self.show_contour = show_contour
            if show_contour:
                self.contour = np.moveaxis(np.array(
                    [feature.canny(self.segmentation[..., slice_ind]) for slice_ind in range(segmentation.shape[-1])]),
                    0, -1) > 0.5
                self.cont = self.ax[0].imshow(self.contour[..., self.ind], cmap="Reds",
                                              alpha=1.0 * (self.contour[..., self.ind] > 0).astype(float))
            self.seg = self.ax[1].imshow(self.segmentation[:, :, self.ind])
        else:
            self.im = self.ax.imshow(self.volume[..., self.ind], vmin=0, vmax=1)
            if title != "":
                self.ax.set_title(self.title)
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        if self.segmentation is not None:
            self.im.set_data(self.volume[:, :, self.ind])
            self.seg.set_data(self.segmentation[:, :, self.ind])
            self.ax[0].set_ylabel('slice %s' % self.ind)
            if self.show_contour:
                self.cont.set_data(self.contour[..., self.ind])
                self.cont.set_alpha(1.0 * (self.contour[..., self.ind] > 0).astype(float))
            self.im.axes.figure.canvas.draw()
            self.seg.axes.figure.canvas.draw()
            if self.show_contour:
                self.cont.axes.figure.canvas.draw()
        else:
            self.im.set_data(self.volume[..., self.ind])
            self.ax.set_ylabel('slice %s' % self.ind)
            self.im.axes.figure.canvas.draw()


def scroll_slices(volume, title=""):
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, volume, title="")
    fig.suptitle(title)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()


def scroll_slices_and_seg(volume, segmentation, title="", show_contour=None):
    mpl.rc('image', cmap='gray')
    fig, ax = plt.subplots(1, 2)
    tracker = IndexTracker(ax, volume, segmentation, "", show_contour)
    fig.suptitle(title)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()


def create_composition(volume, gt, prediction):
    composition = np.stack([volume, volume, volume], axis=-2)
    gt_contour = get_contour(gt)
    prediction_contour = get_contour(prediction)
    composition[..., 0, :][gt_contour] = gt_contour[gt_contour]
    composition[..., 2, :][prediction_contour] = prediction_contour[prediction_contour]
    return composition


def scroll_slices_volume_gt_and_pred(volume, gt, prediction, title=""):
    composition = create_composition(volume, gt, prediction)
    scroll_slices(composition, title="GT (Red) | Prediction (Blue)")


def get_contour(volume):
    contour = np.moveaxis(np.array([feature.canny(volume[..., slice_ind]) for slice_ind in range(volume.shape[-1])]), 0,
                          -1) > 0.5
    return contour
