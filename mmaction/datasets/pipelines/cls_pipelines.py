import os.path as osp
import inspect
import math
import random
from numbers import Number
import numpy as np
from collections.abc import Sequence
import torch
import mmcv
from mmcv.parallel import DataContainer as DC
from PIL import Image

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

from mmaction.datasets.builder import PIPELINES


@PIPELINES.register_module()
class CLSRandomErasing(object):
    """Randomly selects a rectangle region in an image and erase pixels.

    Args:
        erase_prob (float): Probability that image will be randomly erased.
            Default: 0.5
        min_area_ratio (float): Minimum erased area / input image area
            Default: 0.02
        max_area_ratio (float): Maximum erased area / input image area
            Default: 0.4
        aspect_range (sequence | float): Aspect ratio range of erased area.
            if float, it will be converted to (aspect_ratio, 1/aspect_ratio)
            Default: (3/10, 10/3)
        mode (str): Fill method in erased area, can be:

            - const (default): All pixels are assign with the same value.
            - rand: each pixel is assigned with a random value in [0, 255]

        fill_color (sequence | Number): Base color filled in erased area.
            Defaults to (128, 128, 128).
        fill_std (sequence | Number, optional): If set and ``mode`` is 'rand',
            fill erased area with random color from normal distribution
            (mean=fill_color, std=fill_std); If not set, fill erased area with
            random color from uniform distribution (0~255). Defaults to None.

    Note:
        See `Random Erasing Data Augmentation
        <https://arxiv.org/pdf/1708.04896.pdf>`_

        This paper provided 4 modes: RE-R, RE-M, RE-0, RE-255, and use RE-M as
        default. The config of these 4 modes are:

        - RE-R: RandomErasing(mode='rand')
        - RE-M: RandomErasing(mode='const', fill_color=(123.67, 116.3, 103.5))
        - RE-0: RandomErasing(mode='const', fill_color=0)
        - RE-255: RandomErasing(mode='const', fill_color=255)
    """

    def __init__(self,
                 erase_prob=0.5,
                 min_area_ratio=0.02,
                 max_area_ratio=0.4,
                 aspect_range=(3 / 10, 10 / 3),
                 mode='const',
                 fill_color=(128, 128, 128),
                 fill_std=None):
        assert isinstance(erase_prob, float) and 0. <= erase_prob <= 1.
        assert isinstance(min_area_ratio, float) and 0. <= min_area_ratio <= 1.
        assert isinstance(max_area_ratio, float) and 0. <= max_area_ratio <= 1.
        assert min_area_ratio <= max_area_ratio, \
            'min_area_ratio should be smaller than max_area_ratio'
        if isinstance(aspect_range, float):
            aspect_range = min(aspect_range, 1 / aspect_range)
            aspect_range = (aspect_range, 1 / aspect_range)
        assert isinstance(aspect_range, Sequence) and len(aspect_range) == 2 \
            and all(isinstance(x, float) for x in aspect_range), \
            'aspect_range should be a float or Sequence with two float.'
        assert all(x > 0 for x in aspect_range), \
            'aspect_range should be positive.'
        assert aspect_range[0] <= aspect_range[1], \
            'In aspect_range (min, max), min should be smaller than max.'
        assert mode in ['const', 'rand']
        if isinstance(fill_color, Number):
            fill_color = [fill_color] * 3
        assert isinstance(fill_color, Sequence) and len(fill_color) == 3 \
            and all(isinstance(x, Number) for x in fill_color), \
            'fill_color should be a float or Sequence with three int.'
        if fill_std is not None:
            if isinstance(fill_std, Number):
                fill_std = [fill_std] * 3
            assert isinstance(fill_std, Sequence) and len(fill_std) == 3 \
                and all(isinstance(x, Number) for x in fill_std), \
                'fill_std should be a float or Sequence with three int.'

        self.erase_prob = erase_prob
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.aspect_range = aspect_range
        self.mode = mode
        self.fill_color = fill_color
        self.fill_std = fill_std

    def _fill_pixels(self, img, top, left, h, w):
        if self.mode == 'const':
            patch = np.empty((h, w, 3), dtype=np.uint8)
            patch[:, :] = np.array(self.fill_color, dtype=np.uint8)
        elif self.fill_std is None:
            # Uniform distribution
            patch = np.random.uniform(0, 256, (h, w, 3)).astype(np.uint8)
        else:
            # Normal distribution
            patch = np.random.normal(self.fill_color, self.fill_std, (h, w, 3))
            patch = np.clip(patch.astype(np.int32), 0, 255).astype(np.uint8)

        img[top:top + h, left:left + w] = patch
        return img

    def __call__(self, results):
        """
        Args:
            results (dict): Results dict from pipeline

        Returns:
            dict: Results after the transformation.
        """
        for key in results.get('img_fields', ['img']):
            if np.random.rand() > self.erase_prob:
                continue
            img = results[key]
            img_h, img_w = img.shape[:2]

            # convert to log aspect to ensure equal probability of aspect ratio
            log_aspect_range = np.log(
                np.array(self.aspect_range, dtype=np.float32))
            aspect_ratio = np.exp(np.random.uniform(*log_aspect_range))
            area = img_h * img_w
            area *= np.random.uniform(self.min_area_ratio, self.max_area_ratio)

            h = min(int(round(np.sqrt(area * aspect_ratio))), img_h)
            w = min(int(round(np.sqrt(area / aspect_ratio))), img_w)
            top = np.random.randint(0, img_h - h) if img_h > h else 0
            left = np.random.randint(0, img_w - w) if img_w > w else 0
            img = self._fill_pixels(img, top, left, h, w)

            results[key] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(erase_prob={self.erase_prob}, '
        repr_str += f'min_area_ratio={self.min_area_ratio}, '
        repr_str += f'max_area_ratio={self.max_area_ratio}, '
        repr_str += f'aspect_range={self.aspect_range}, '
        repr_str += f'mode={self.mode}, '
        repr_str += f'fill_color={self.fill_color}, '
        repr_str += f'fill_std={self.fill_std})'
        return repr_str


@PIPELINES.register_module()
class CLSLoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes()`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 decoding_backend=None):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.decoding_backend = decoding_backend

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        if 'img_bin' in results:  # support read from HDFS
            img_bytes = results['img_bin']
        else:
            if self.file_client is None:
                self.file_client = mmcv.FileClient(**self.file_client_args)

            img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type, backend=self.decoding_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class CLSRandomCrop(object):
    """Crop the given Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. If a sequence of length 4 is provided, it is used to
            pad left, top, right, bottom borders respectively.  If a sequence
            of length 2 is provided, it is used to pad left/right, top/bottom
            borders, respectively. Default: None, which means no padding.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
            Default: False.
        pad_val (Number | Sequence[Number]): Pixel pad_val value for constant
            fill. If a tuple of length 3, it is used to pad_val R, G, B
            channels respectively. Default: 0.
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Default: constant.
            -constant: Pads with a constant value, this value is specified
                with pad_val.
            -edge: pads with the last value at the edge of the image.
            -reflect: Pads with reflection of image without repeating the
                last value on the edge. For example, padding [1, 2, 3, 4]
                with 2 elements on both sides in reflect mode will result
                in [3, 2, 1, 2, 3, 4, 3, 2].
            -symmetric: Pads with reflection of image repeating the last
                value on the edge. For example, padding [1, 2, 3, 4] with
                2 elements on both sides in symmetric mode will result in
                [2, 1, 1, 2, 3, 4, 4, 3].
    """

    def __init__(self,
                 size,
                 padding=None,
                 pad_if_needed=False,
                 pad_val=0,
                 padding_mode='constant'):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        # check padding mode
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.pad_val = pad_val
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (ndarray): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: Params (xmin, ymin, target_height, target_width) to be
                passed to ``crop`` for random crop.
        """
        height = img.shape[0]
        width = img.shape[1]
        target_height, target_width = output_size
        if width == target_width and height == target_height:
            return 0, 0, height, width

        ymin = random.randint(0, height - target_height)
        xmin = random.randint(0, width - target_width)
        return xmin, ymin, target_height, target_width

    def __call__(self, results):
        """
        Args:
            img (ndarray): Image to be cropped.
        """
        for key in results.get('img_fields', ['img']):
            img = results[key]
            if self.padding is not None:
                img = mmcv.impad(
                    img, padding=self.padding, pad_val=self.pad_val)

            # pad the height if needed
            if self.pad_if_needed and img.shape[0] < self.size[0]:
                img = mmcv.impad(
                    img,
                    padding=(0, self.size[0] - img.shape[0], 0,
                             self.size[0] - img.shape[0]),
                    pad_val=self.pad_val,
                    padding_mode=self.padding_mode)

            # pad the width if needed
            if self.pad_if_needed and img.shape[1] < self.size[1]:
                img = mmcv.impad(
                    img,
                    padding=(self.size[1] - img.shape[1], 0,
                             self.size[1] - img.shape[1], 0),
                    pad_val=self.pad_val,
                    padding_mode=self.padding_mode)

            xmin, ymin, height, width = self.get_params(img, self.size)
            results[key] = mmcv.imcrop(
                img,
                np.array([
                    xmin,
                    ymin,
                    xmin + width - 1,
                    ymin + height - 1,
                ]))
        return results

    def __repr__(self):
        return (self.__class__.__name__ +
                f'(size={self.size}, padding={self.padding})')


@PIPELINES.register_module()
class CLSRandomResizedCrop(object):
    """Crop the given image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio
    is made. This crop is finally resized to given size.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        scale (tuple): Range of the random size of the cropped image compared
            to the original image. Default: (0.08, 1.0).
        ratio (tuple): Range of the random aspect ratio of the cropped image
            compared to the original image. Default: (3. / 4., 4. / 3.).
        interpolation (str): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Default:
            'bilinear'.
        backend (str): The image resize backend type, accpeted values are
            `cv2` and `pillow`. Default: `cv2`.
    """

    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 interpolation='bilinear',
                 backend='cv2'):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            raise ValueError('range should be of kind (min, max). '
                             f'But received {scale}')
        if backend not in ['cv2', 'pillow']:
            raise ValueError(f'backend: {backend} is not supported for resize.'
                             'Supported backends are "cv2", "pillow"')

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.backend = backend

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (ndarray): Image to be cropped.
            scale (tuple): Range of the random size of the cropped image
                compared to the original image size.
            ratio (tuple): Range of the random aspect ratio of the cropped
                image compared to the original image area.

        Returns:
            tuple: Params (xmin, ymin, target_height, target_width) to be
                passed to ``crop`` for a random sized crop.
        """
        height = img.shape[0]
        width = img.shape[1]
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            target_width = int(round(math.sqrt(target_area * aspect_ratio)))
            target_height = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < target_width <= width and 0 < target_height <= height:
                ymin = random.randint(0, height - target_height)
                xmin = random.randint(0, width - target_width)
                return xmin, ymin, target_height, target_width

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            target_width = width
            target_height = int(round(target_width / min(ratio)))
        elif in_ratio > max(ratio):
            target_height = height
            target_width = int(round(target_height * max(ratio)))
        else:  # whole image
            target_width = width
            target_height = height
        ymin = (height - target_height) // 2
        xmin = (width - target_width) // 2
        return xmin, ymin, target_height, target_width

    def __call__(self, results):
        """
        Args:
            img (ndarray): Image to be cropped and resized.

        Returns:
            ndarray: Randomly cropped and resized image.
        """
        for key in results.get('img_fields', ['img']):
            img = results[key]
            xmin, ymin, target_height, target_width = self.get_params(
                img, self.scale, self.ratio)
            img = mmcv.imcrop(
                img,
                np.array([
                    xmin, ymin, xmin + target_width - 1,
                    ymin + target_height - 1
                ]))
            results[key] = mmcv.imresize(
                img,
                tuple(self.size[::-1]),
                interpolation=self.interpolation,
                backend=self.backend)
        return results

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(size={self.size}'
        format_string += f', scale={tuple(round(s, 4) for s in self.scale)}'
        format_string += f', ratio={tuple(round(r, 4) for r in self.ratio)}'
        format_string += f', interpolation={self.interpolation})'
        return format_string


@PIPELINES.register_module()
class CLSRandomGrayscale(object):
    """Randomly convert image to grayscale with a probability of gray_prob.

    Args:
        gray_prob (float): Probability that image should be converted to
            grayscale. Default: 0.1.

    Returns:
        ndarray: Grayscale version of the input image with probability
            gray_prob and unchanged with probability (1-gray_prob).
            - If input image is 1 channel: grayscale version is 1 channel.
            - If input image is 3 channel: grayscale version is 3 channel
                with r == g == b.

    """

    def __init__(self, gray_prob=0.1):
        self.gray_prob = gray_prob

    def __call__(self, results):
        """
        Args:
            img (ndarray): Image to be converted to grayscale.

        Returns:
            ndarray: Randomly grayscaled image.
        """
        for key in results.get('img_fields', ['img']):
            img = results[key]
            num_output_channels = img.shape[2]
            if random.random() < self.gray_prob:
                if num_output_channels > 1:
                    img = mmcv.rgb2gray(img)[:, :, None]
                    results[key] = np.dstack(
                        [img for _ in range(num_output_channels)])
                    return results
            results[key] = img
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(gray_prob={self.gray_prob})'


@PIPELINES.register_module()
class CLSRandomFlip(object):
    """Flip the image randomly.

    Flip the image randomly based on flip probaility and flip direction.

    Args:
        flip_prob (float): probability of the image being flipped. Default: 0.5
        direction (str): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    def __init__(self, flip_prob=0.5, direction='horizontal'):
        assert 0 <= flip_prob <= 1
        assert direction in ['horizontal', 'vertical']
        self.flip_prob = flip_prob
        self.direction = direction

    def __call__(self, results):
        """Call function to flip image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """
        flip = True if np.random.rand() < self.flip_prob else False
        results['flip'] = flip
        results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            for key in results.get('img_fields', ['img']):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_prob={self.flip_prob})'


@PIPELINES.register_module()
class CLSResize(object):
    """Resize images.

    Args:
        size (int | tuple): Images scales for resizing (h, w).
            When size is int, the default behavior is to resize an image
            to (size, size). When size is tuple and the second value is -1,
            the short edge of an image is resized to its first value.
            For example, when size is 224, the image is resized to 224x224.
            When size is (224, -1), the short side is resized to 224 and the
            other side is computed based on the short side, maintaining the
            aspect ratio.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos".
            More details can be found in `mmcv.image.geometric`.
        backend (str): The image resize backend type, accpeted values are
            `cv2` and `pillow`. Default: `cv2`.
    """

    def __init__(self, size, interpolation='bilinear', backend='cv2'):
        assert isinstance(size, int) or (isinstance(size, tuple)
                                         and len(size) == 2)
        self.resize_w_short_side = False
        if isinstance(size, int):
            assert size > 0
            size = (size, size)
        else:
            assert size[0] > 0 and (size[1] > 0 or size[1] == -1)
            if size[1] == -1:
                self.resize_w_short_side = True
        assert interpolation in ('nearest', 'bilinear', 'bicubic', 'area',
                                 'lanczos')
        if backend not in ['cv2', 'pillow']:
            raise ValueError(f'backend: {backend} is not supported for resize.'
                             'Supported backends are "cv2", "pillow"')

        self.size = size
        self.interpolation = interpolation
        self.backend = backend

    def _resize_img(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            ignore_resize = False
            if self.resize_w_short_side:
                h, w = img.shape[:2]
                short_side = self.size[0]
                if (w <= h and w == short_side) or (h <= w
                                                    and h == short_side):
                    ignore_resize = True
                else:
                    if w < h:
                        width = short_side
                        height = int(short_side * h / w)
                    else:
                        height = short_side
                        width = int(short_side * w / h)
            else:
                height, width = self.size
            if not ignore_resize:
                img = mmcv.imresize(
                    img,
                    size=(width, height),
                    interpolation=self.interpolation,
                    return_scale=False,
                    backend=self.backend)
                results[key] = img
                results['img_shape'] = img.shape

    def __call__(self, results):
        self._resize_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@PIPELINES.register_module()
class CLSCenterCrop(object):
    """Center crop the image.

    Args:
        crop_size (int | tuple): Expected size after cropping, (h, w).

    Notes:
        If the image is smaller than the crop size, return the original image
    """

    def __init__(self, crop_size):
        assert isinstance(crop_size, int) or (isinstance(crop_size, tuple)
                                              and len(crop_size) == 2)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size

    def __call__(self, results):
        crop_height, crop_width = self.crop_size[0], self.crop_size[1]
        for key in results.get('img_fields', ['img']):
            img = results[key]
            # img.shape has length 2 for grayscale, length 3 for color
            img_height, img_width = img.shape[:2]

            y1 = max(0, int(round((img_height - crop_height) / 2.)))
            x1 = max(0, int(round((img_width - crop_width) / 2.)))
            y2 = min(img_height, y1 + crop_height) - 1
            x2 = min(img_width, x1 + crop_width) - 1

            # crop the image
            img = mmcv.imcrop(img, bboxes=np.array([x1, y1, x2, y2]))
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module()
class CLSNormalize(object):
    """Normalize the image.

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            results[key] = mmcv.imnormalize(results[key], self.mean, self.std,
                                            self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={list(self.mean)}, '
        repr_str += f'std={list(self.std)}, '
        repr_str += f'to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class CLSAlbu(object):
    """Albumentation augmentation.

    Adds custom transformations from Albumentations library.
    Please, visit `https://albumentations.readthedocs.io`
    to get more information.
    An example of ``transforms`` is as followed:

    .. code-block::
        [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]

    Args:
        transforms (list[dict]): A list of albu transformations
        keymap (dict): Contains {'input key':'albumentation-style key'}
    """

    def __init__(self, transforms, keymap=None, update_pad_shape=False):
        if Compose is None:
            raise RuntimeError('albumentations is not installed')

        self.transforms = transforms
        self.filter_lost_elements = False
        self.update_pad_shape = update_pad_shape

        self.aug = Compose([self.albu_builder(t) for t in self.transforms])

        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg):
        """Import a module from albumentations.
        It inherits some of :func:`build_from_cfg` logic.
        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if mmcv.is_str(obj_type):
            if albumentations is None:
                raise RuntimeError('albumentations is not installed')
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """Dictionary mapper. Renames keys according to keymap provided.
        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """

        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)

        results = self.aug(**results)

        if 'gt_labels' in results:
            if isinstance(results['gt_labels'], list):
                results['gt_labels'] = np.array(results['gt_labels'])
            results['gt_labels'] = results['gt_labels'].astype(np.int64)

        # back to the original format
        results = self.mapper(results, self.keymap_back)

        # update final shape
        if self.update_pad_shape:
            results['pad_shape'] = results['img'].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str


def CLSto_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f'Type {type(data)} cannot be converted to tensor.'
            'Supported types are: `numpy.ndarray`, `torch.Tensor`, '
            '`Sequence`, `int` and `float`')


@PIPELINES.register_module()
class CLSToTensor(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = CLSto_tensor(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PIPELINES.register_module()
class CLSImageToTensor(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = CLSto_tensor(img.transpose(2, 0, 1))
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PIPELINES.register_module()
class CLSTranspose(object):

    def __init__(self, keys, order):
        self.keys = keys
        self.order = order

    def __call__(self, results):
        for key in self.keys:
            results[key] = results[key].transpose(self.order)
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(keys={self.keys}, order={self.order})'


@PIPELINES.register_module()
class CLSToPIL(object):

    def __init__(self):
        pass

    def __call__(self, results):
        results['img'] = Image.fromarray(results['img'])
        return results


@PIPELINES.register_module()
class CLSToNumpy(object):

    def __init__(self):
        pass

    def __call__(self, results):
        results['img'] = np.array(results['img'], dtype=np.float32)
        return results


@PIPELINES.register_module()
class CLSCollect(object):
    """
    Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img" and "gt_label".

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('filename', 'ori_shape', 'img_shape', 'flip',
            'flip_direction', 'img_norm_cfg')``

    Returns:
        dict: The result dict contains the following keys
                - keys in``self.keys``
                - ``img_metas`` if avaliable
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'flip', 'flip_direction',
                            'img_norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data['img_metas'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            if key == 'img':
                data['imgs'] = results[key]
            elif key == 'gt_label':
                data['label'] = results[key]
            else:
                data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'


@PIPELINES.register_module()
class CLSWrapFieldsToLists(object):
    """Wrap fields of the data dictionary into lists for evaluation.

    This class can be used as a last step of a test or validation
    pipeline for single image evaluation or inference.

    Example:
        >>> test_pipeline = [
        >>>    dict(type='LoadImageFromFile'),
        >>>    dict(type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
        >>>    dict(type='ImageToTensor', keys=['img']),
        >>>    dict(type='Collect', keys=['img']),
        >>>    dict(type='WrapIntoLists')
        >>> ]
    """

    def __call__(self, results):
        # Wrap dict fields into lists
        for key, val in results.items():
            results[key] = [val]
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}()'
