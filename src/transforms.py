import random
import numpy as np
import librosa


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, trg=None):
        if trg is None:
            for t in self.transforms:
                image = t(image)
            return image
        else:
            for t in self.transforms:
                image, trg = t(image, trg)
            return image, trg


class UseWithProb:
    def __init__(self, transform, prob=0.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image, trg=None):
        if trg is None:
            if random.random() < self.prob:
                image = self.transform(image)
            return image
        else:
            if random.random() < self.prob:
                image, trg = self.transform(image, trg)
            return image, trg


class OneOf:
    def __init__(self, transforms, p=None):
        self.transforms = transforms
        self.p = p

    def __call__(self, image, trg=None):
        transform = np.random.choice(self.transforms, p=self.p)
        if trg is None:
            image = transform(image)
            return image
        else:
            image, trg = transform(image, trg)
            return image, trg


class PitchShift:
    def __init__(self, pitch_range, sr):
        self.pr_low, self.pr_hi = pitch_range
        self.sr = sr

    def __call__(self, audio):
        shift = np.random.choice(np.linspace(self.pr_low, self.pr_hi, 100))
        return librosa.effects.pitch_shift(audio, self.sr, shift)


class TimeStretch:
    def __init__(self, stretch_param):
        self.stretch = stretch_param

    def __call__(self, audio):
        """
        if self.stretch is one number, use it as stretch param.
        if it's 2nums array, use it as limits for uniform distribution
        from which we sample stretch param
        """
        if type(self.stretch) in (int, float):
            return librosa.effects.time_stretch(audio, self.stretch)
        else:
            low, hi = self.stretch
            s = np.random.choice(np.linspace(low, hi, 100))
            return librosa.effects.time_stretch(audio, s)


class AddNoise:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def __call__(self, audio):
        noise = np.random.normal(loc=self.loc, scale=self.scale, size=audio.shape)
        return noise + audio


class ScaleAmp:
    def __init__(self, share, scale):
        self.share = share
        self.scale = scale

    def __call__(self, audio):
        # choose period:
        audio = audio.copy()
        share_int = int(self.share * len(audio))
        start_seg = random.randint(0, len(audio) - share_int)
        end_seg = start_seg + share_int

        # choose scale
        if type(self.scale) in (int, float):
            audio[start_seg:end_seg] = audio[start_seg:end_seg] * self.scale
            return audio
        random_scale = np.random.choice(np.linspace(self.scale[0], self.scale[1], 200))
        audio[start_seg:end_seg] = audio[start_seg:end_seg] * random_scale
        return audio


class TrimAug:
    def __init__(self):
        pass

    def __call__(self, audio):
        raw, _ = librosa.effects.trim(audio)
        return raw
