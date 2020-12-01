import python_speech_features as psf
import numpy as np
import librosa
import cv2

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

def normalize(image, mean, std):
    image = (image / 255.0).astype(np.float32)
    image = (image - mean) / std
    return image

def audio2image(signal, sample_rate, window_length=0.05, window_step=0.0045, NFFT=2205):

    # preemphasis
    signal = psf.sigproc.preemphasis(signal, coeff=0.95)

    # get specrogram
    # Get the frames
    frames = psf.sigproc.framesig(
        signal, window_length * sample_rate, window_step * sample_rate, lambda x: np.ones((x,))
    )  # Window function

    # magnitude Spectrogram
    spectrogram = np.rot90(psf.sigproc.magspec(frames, NFFT))

    # get rid of high frequencies
    spectrogram = spectrogram[512:, :]

    # normalize in [0, 1]
    spectrogram -= spectrogram.min(axis=None)
    spectrogram /= spectrogram.max(axis=None)

    spectrogram = spectrogram[:591, :549]

    return spectrogram


def build_spectrogram(raw_audio):

    # Use our function from earlier
    spectrogram = audio2image(raw_audio)  # a 2D array

    # Pad to make sure it is 512 x 512
    h, w = spectrogram.shape
    spectrogram = np.pad(spectrogram, [(591 - h, 0), (549 - w, 0)])

    # Scale to (0, 255)
    spectrogram -= spectrogram.min()
    spectrogram *= 255.0 / spectrogram.max()

    # Make it uint8
    im_arr = np.array(spectrogram, np.uint8)

    # Make it rgb (hint - some fun tricks you can do here!)
    r = im_arr
    g = im_arr
    b = im_arr

    return np.stack([r, g, b], axis=-1)


def new_generate_spec(audio, config):
    arr_limit = int(config.max_len_sec * config.sr)

    # trim according to the config
    if config.trim:
        audio, trim_idx = librosa.effects.trim(audio)
    #     print("audio shape", audio.shape)
    # for long audios: trim:
    if len(audio) >= arr_limit:
        audio = audio[:arr_limit]
    # for short: pad:
    else:

        to_add = arr_limit - len(audio)
        # pad either on the left:
        if not config.pad_center:
            audio = np.concatenate((np.zeros(to_add), audio))
        # or on both sides (simmetrically):
        else:
            add_l = to_add // 2
            add_r = to_add - add_l
            audio = np.concatenate((np.zeros(add_l), audio, np.zeros(add_r)))

    X = librosa.stft(audio, n_fft=config.n_fft, hop_length=config.hop_size)
    Xdb = librosa.amplitude_to_db(abs(X), ref=np.max)
    return Xdb


def new_build_image(audio, config):
    spec = new_generate_spec(audio, config)

    # Scale to (0, 255)
    spec -= spec.min()
    spec *= 255.0 / spec.max()

    # Make it uint8
    im_arr = np.array(spec, np.uint8)

    # Make it rgb (hint - some fun tricks you can do here!)
    r = im_arr
    g = im_arr
    b = im_arr

    image = np.stack([r, g, b], axis=-1)
    image = cv2.resize(image, (config.img_size, config.img_size))

    return image
