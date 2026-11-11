import matplotlib.transforms as mtransforms
from matplotlib.transforms import TransformedBbox, Bbox
from matplotlib.image import AxesImage
import numpy as np
import scipy.ndimage


class CachedAxesImage(AxesImage):
    """
    A subclass of AxesImage that caches the rendered image and updates
    only when the x or y limits of the axes or the figure size changes.
    """

    def __init__(self, ax, rolling_max=False, **kwargs):
        super().__init__(ax, **kwargs)
        self._cached_image = None
        self._cached_extent = None
        self._cached_figure_size = None
        self._cached_lims = None
        self._cached_clim = None
        self._cached_clip = None
        self._cached_transform = None
        self.rolling_max = rolling_max

    def update_data_fast(self, new_data, extent=None):
        """Update image data and extent in-place, invalidating the cache."""
        self.set_data(new_data)
        if extent is not None:
            self._extent = extent
        self.stale = True
        self._cached_image = None

    def make_image(self, renderer, magnification=1.0, unsampled=False):
        if self._A is None or self._A.size == 0 or self._A.shape[0] == 0 or self._A.shape[1] == 0:
            return None, 0, 0, None
        current_extent = self.get_extent()
        current_figure_size = self.axes.figure.get_size_inches()
        current_xlim = self.axes.get_xlim()
        current_ylim = self.axes.get_ylim()
        current_clim = self.get_clim()

        clip = (
            (self.get_clip_box() or self.axes.bbox)
            if self.get_clip_on()
            else self.get_figure(root=True).bbox
        )
        trans = self.get_transform()
        x1, x2, y1, y2 = self.get_extent()
        bbox = Bbox(np.array([[x1, y1], [x2, y2]]))
        transformed_bbox = TransformedBbox(bbox, trans)
        # the transformedbbox is the bounding box of the image in display coordinates
        clipped_transformed_bbox = Bbox.intersection(transformed_bbox, clip)
        if clipped_transformed_bbox is None:
            return None, 0, 0, None
        clipped_bbox = clipped_transformed_bbox.transformed(trans.inverted())
        clip2 = None
        if self._cached_clip is not None:
            clip2 = self._cached_clip.transformed(trans)

        # Check if the cached image is valid
        if (
            self._cached_image is not None
            and self._cached_extent == current_extent
            and self._cached_figure_size == tuple(current_figure_size)
            and self._cached_magnification == magnification
            # and self._cached_lims == (current_xlim, current_ylim)
            # check ratio of xlim and ylim
            and np.isclose(
                (
                    abs(
                        (current_xlim[1] - current_xlim[0])
                        / (current_ylim[1] - current_ylim[0])
                    )
                ),
                abs(
                    (self._cached_lims[0][1] - self._cached_lims[0][0])
                    / (self._cached_lims[1][1] - self._cached_lims[1][0])
                ),
            )
            and self._cached_clim == current_clim
            and (
                self._cached_clip is not None
                and max(self._cached_clip.xmin, 0) <= max(clipped_bbox.xmin, 0)
                and min(self._cached_clip.xmax, self._A.shape[0])
                >= min(clipped_bbox.xmax, self._A.shape[0])
                and max(self._cached_clip.ymin, 0) <= max(clipped_bbox.ymin, 0)
                and min(self._cached_clip.ymax, self._A.shape[1])
                >= min(clipped_bbox.ymax, self._A.shape[1])
            )
            and not self.stale
        ):
            # return self._cached_image
            # write the transform that gets from self._cached_clip to clip
            clip2 = Bbox.intersection(transformed_bbox, clip2)
            x0 = clip2.x0  # self._cached_clip.xmin- transformed_bbox.xmin
            y0 = clip2.y0  # self._cached_clip.ymin- transformed_bbox.ymin
            return self._cached_image[0], x0, y0, None  # self._cached_image[3]

        clip2 = clip.expanded(2, 2)  # 2x the width and height of the window.
        # Cache the new larger cut of the data
        fig = self.get_figure()
        fac = renderer.dpi / fig.dpi
        if self._A is None or self._A.size == 0 or self._A.shape[0] == 0 or self._A.shape[1] == 0:
            return None, 0, 0, None

        img_bbox = bbox
        img_transformed_bbox = transformed_bbox

        x_scale = (abs(bbox.width) / abs(transformed_bbox.width)) / 2 if transformed_bbox.width != 0 else 0
        y_scale = (abs(bbox.height) / abs(transformed_bbox.height)) / 2 if transformed_bbox.height != 0 else 0

        # Viewport slicing when array has many rows
        if self._A.shape[0] > 200:
            data_clip = Bbox.intersection(transformed_bbox, clip2)
            if data_clip is not None:
                data_clip = data_clip.transformed(trans.inverted())
                pad_x = int(x_scale) + 2 if int(x_scale) > 1 else 0
                pad_y = int(y_scale) + 2 if int(y_scale) > 1 else 0

                x_min = max(0, int(np.floor(min(data_clip.x0, data_clip.x1))) - pad_x)
                x_max = min(self._A.shape[1], int(np.ceil(max(data_clip.x0, data_clip.x1))) + pad_x)
                y_min = max(0, int(np.floor(min(data_clip.y0, data_clip.y1))) - pad_y)
                y_max = min(self._A.shape[0], int(np.ceil(max(data_clip.y0, data_clip.y1))) + pad_y)

                if x_max > x_min and y_max > y_min:
                    A = self._A[y_min:y_max, x_min:x_max]
                    if y1 > y2:
                        img_bbox = Bbox(np.array([[x_min, y_max], [x_max, y_min]]))
                    else:
                        img_bbox = Bbox(np.array([[x_min, y_min], [x_max, y_max]]))
                    img_transformed_bbox = TransformedBbox(img_bbox, trans)
                else:
                    A = self._A
            else:
                A = self._A
        else:
            A = self._A

        if self.rolling_max:
            # Safe conversion for NaNs / MaskedArrays
            if np.ma.isMaskedArray(A):
                A_arr = np.nan_to_num(A.filled(0.0), nan=0.0)
            elif np.isnan(A).any():
                A_arr = np.nan_to_num(A, nan=0.0)
            else:
                A_arr = np.array(A, copy=True)

            if int(x_scale) > 1:
                A_arr = scipy.ndimage.maximum_filter1d(A_arr, size=int(x_scale), axis=1, mode="nearest")
            if int(y_scale) > 1:
                A_arr = scipy.ndimage.maximum_filter1d(A_arr, size=int(y_scale), axis=0, mode="nearest")
            A = A_arr

        A_to_pass = np.ma.asarray(A)

        self._cached_image = self._make_image(
            A_to_pass,
            img_bbox,
            img_transformed_bbox,
            clip2,
            magnification=magnification / fac,
            unsampled=unsampled,
        )

        self._cached_clip = Bbox.intersection(transformed_bbox, clip2).transformed(
            trans.inverted()
        )
        self._cached_extent = current_extent
        self._cached_figure_size = tuple(current_figure_size)
        self._cached_magnification = magnification
        self._cached_lims = (current_xlim, current_ylim)
        self._cached_clim = current_clim

        return self._cached_image
