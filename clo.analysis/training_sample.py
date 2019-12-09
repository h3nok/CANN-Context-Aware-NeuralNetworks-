from ITT import itt
import numpy as np
import skimage
from PIL import Image
import scipy.fftpack as fp


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

    def __init__(self, data, label):
        assert isinstance(data, np.ndarray)
        self._data = data
        self._label = label
        self._attributes = SampleStat(data)

    @property
    def entropy(self):
        return float(self._attributes.entropy())

    @property
    def fft_iq(self):
        return float(self._attributes.fourier_spectrum_iq())


class SampleStat:
    _sample = None

    def __init__(self, sample: np.ndarray):
        self._sample = sample

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

    def mutual_information(self, other_sample):
        return round(itt.information_mutual(self._sample, other_sample))

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
