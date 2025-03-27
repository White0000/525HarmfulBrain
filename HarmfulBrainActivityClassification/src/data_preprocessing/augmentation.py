import numpy as np
import torch
import pywt
from concurrent.futures import ThreadPoolExecutor

class EEGAugmentor:
    def __init__(
        self,
        noise_std=0.01,
        shift_ratio=0.1,
        scale_range=(0.9, 1.1),
        drop_prob=0.0,
        wavelet=None,
        device="cpu",
        random_wavelets=None,
        specaugment=False,
        freq_mask_param=5,
        time_mask_param=10
    ):
        self.noise_std = noise_std
        self.shift_ratio = shift_ratio
        self.scale_min, self.scale_max = scale_range
        self.drop_prob = drop_prob
        self.wavelet = wavelet
        self.device = device
        self.random_wavelets = random_wavelets if random_wavelets else []
        self.specaugment = specaugment
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def add_gaussian_noise(self, x):
        return x + np.random.normal(0, self.noise_std, x.shape)

    def time_shift(self, x):
        if x.ndim == 1:
            s = int(x.shape[0] * self.shift_ratio)
            r = np.random.randint(-s, s + 1)
            if r > 0:
                return np.pad(x, (r, 0), mode="constant")[:-r]
            if r < 0:
                r = abs(r)
                return np.pad(x, (0, r), mode="constant")[r:]
            return x
        else:
            s = int(x.shape[-1] * self.shift_ratio)
            r = np.random.randint(-s, s + 1)
            if r > 0:
                return np.pad(x, ((0, 0), (r, 0)), mode="constant")[:, :-r]
            if r < 0:
                r = abs(r)
                return np.pad(x, ((0, 0), (0, r)), mode="constant")[:, r:]
            return x

    def random_scale(self, x):
        return x * np.random.uniform(self.scale_min, self.scale_max)

    def random_drop(self, x):
        m = np.random.rand(*x.shape) > self.drop_prob
        return x * m

    def wavelet_noise(self, x):
        name = self.wavelet
        if not name and self.random_wavelets:
            name = np.random.choice(self.random_wavelets)
        if x.ndim == 1:
            return self.wavelet_augment_1d(x, name)
        else:
            return np.stack([self.wavelet_augment_1d(ch, name) for ch in x], axis=0)

    def wavelet_augment_1d(self, signal, w):
        c = pywt.wavedec(signal, w, mode="symmetric")
        if len(c) > 1:
            i = np.random.randint(1, len(c))
            c[i] = c[i] + np.random.normal(0, self.noise_std, c[i].shape)
        r = pywt.waverec(c, w, mode="symmetric")
        return r[: signal.shape[-1]]

    def spec_augment(self, x):
        if x.ndim != 2:
            return x
        f = x.shape[0]
        t = x.shape[1]
        f_mask = np.random.randint(0, f - self.freq_mask_param) if f > self.freq_mask_param else 0
        t_mask = np.random.randint(0, t - self.time_mask_param) if t > self.time_mask_param else 0
        x_cp = x.copy()
        if f_mask > 0:
            x_cp[f_mask : f_mask + self.freq_mask_param, :] = 0
        if t_mask > 0:
            x_cp[:, t_mask : t_mask + self.time_mask_param] = 0
        return x_cp

    def single_augment(self, x):
        if np.random.rand() < 0.5:
            x = self.add_gaussian_noise(x)
        if np.random.rand() < 0.5:
            x = self.time_shift(x)
        if np.random.rand() < 0.5:
            x = self.random_scale(x)
        if self.drop_prob > 0 and np.random.rand() < 0.5:
            x = self.random_drop(x)
        if (self.wavelet or self.random_wavelets) and np.random.rand() < 0.5:
            x = self.wavelet_noise(x)
        if self.specaugment and np.random.rand() < 0.5:
            x = self.spec_augment(x)
        return torch.tensor(x, dtype=torch.float32, device=self.device).cpu().numpy()

    def augment(self, x):
        y = self.single_augment(x)
        return torch.tensor(y, dtype=torch.float32, device=self.device).numpy()

    def augment_batch(self, batch, num_threads=4):
        r = []
        with ThreadPoolExecutor(max_workers=num_threads) as e:
            f = [e.submit(self.single_augment, b) for b in batch]
            for i in f:
                r.append(i.result())
        return np.stack(r, axis=0)

def demo():
    x = np.sin(np.linspace(0, 6.28, 1000))
    a = EEGAugmentor(
        noise_std=0.02,
        shift_ratio=0.05,
        scale_range=(0.95, 1.05),
        drop_prob=0.1,
        wavelet="db1",
        device="cpu",
        random_wavelets=["db1","db2","coif1"],
        specaugment=True,
        freq_mask_param=10,
        time_mask_param=20
    )
    y = a.augment(x)
    b = np.array([x for _ in range(8)])
    yb = a.augment_batch(b, num_threads=4)
    print(y.mean(), yb.mean())

if __name__ == "__main__":
    demo()
