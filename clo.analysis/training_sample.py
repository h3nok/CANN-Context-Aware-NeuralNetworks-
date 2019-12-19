from ITT import itt
import numpy as np
import skimage
from PIL import Image, ImageStat
import scipy.fftpack as fp
from map_measure import Measure, map_measure_fn
import cv2
import math


class FFT:
    _frame_data = None
    _freq_img = None

    def __init__(self, data):
        self._frame_data = data
        self._convert_to_freq_domain = lambda freq: fp.rfft(fp.rfft(data, axis=0), axis=0)
        self._freq_img = self._convert_to_freq_domain(self._frame_data)

    @property
    def freq_image(self):
        return self._freq_img


class Sample:
    _data = None
    _attributes = None
    _label = None
    _id = None
    _pil_frame_data = None

    def __init__(self, data, label, id=None):
        assert isinstance(data, np.ndarray)
        self._data = data
        self._label = label
        self._id = id
        self._attributes = SampleAttribute(data)

    @property
    def entropy(self):
        return float(self._attributes.entropy())

    @property
    def fft_iq(self):
        return float(self._attributes.fourier_spectrum_iq())

    @property
    def mi(self, other_sample):
        return float(self._attributes.itt_overlap_or_distance(other_sample, metric=Measure.MI))


class SampleAttribute:
    _sample = None
    _pil_frame_data = None

    def __init__(self, sample: np.ndarray):
        self._sample = sample
        self._pil_frame_data = Image.fromarray(sample)

    def entropy(self):
        return round(skimage.measure.shannon_entropy(self._sample), 4)

    def conditional_entropy(self, other_sample):
        return round(itt.entropy_conditional(self._sample, other_sample))

    def ssim(self, other_sample):
        return round(skimage.measure.compare_ssim(self._sample, other_sample))

    def psnr(self, other_sample):
        return round(skimage.measure.compare_psnr(self._sample, other_sample))

    def snmse(self, other_sample):
        return round(skimage.measure.compare_nrmse(self._sample, other_sample))

    def mse(self, other_sample):
        return round(skimage.measure.compare_mse(self._sample, other_sample))

    def joint_entropy(self, other_sample):
        return round(itt.entropy_joint(self._sample, other_sample))

    def itt_overlap_or_distance(self, other_sample, metric):
        assert isinstance(metric, Measure)
        samples = [self._sample, other_sample]
        # return round(itt.information_mutual(self._sample, other_sample))
        return map_measure_fn(metric)

    def fourier_spectrum_iq(self):
        """
        Input: Image I of size M×N.  Output: Image Quality measure (FM) where FM stands for
        Frequency Domain Image Blur
        Measure Step 1: Compute F which is the Fourier Transform representation of image I
        Step 2:  Find Fc which is obtained by shifting the origin of F to centre.
        Step 3: Calculate AF = abs (Fc) where AF is the absolute value of the centered Fourier
        transform of image I.
        Step 4:  Calculate M = max (AF) where M is the maximum value of the frequency component
        where thres = M/1000.Step 6: Calculate Image Quality measure (FM) from equation (1).
        :return:
        """
        _pil_image = Image.fromarray(self._sample)
        width, height = _pil_image.size
        _fft_freq_img = FFT(_pil_image).freq_image
        fft_centered = fp.fftshift(_fft_freq_img)
        af = np.abs(fft_centered)
        m = np.max(af)
        _fft_img_quality_threshold = round(m / 1000)
        th = len(np.where(_fft_freq_img > _fft_img_quality_threshold))
        _fm = round((th / m * width), 4)

        return _fm

    def perceived_brightness(self):
        stat = ImageStat.Stat(self._pil_frame_data)
        r, g, b = stat.rms
        return round(math.sqrt(0.241 * (r ** 2) +
                               0.691 * (g ** 2) +
                               0.068 * (b ** 2)))

    def brightness(self, mode='mean'):
        stat = ImageStat.Stat(self._pil_frame_data.convert('L'))
        if mode == 'mean':
            return round(stat.mean[0])
        else:
            return round(stat.rms[0])

    def colors(self):
        return self._pil_frame_data.getcolors(255)

    def variance(self):
        return round(cv2.Laplacian(self._sample, cv2.CV_64F).var())
