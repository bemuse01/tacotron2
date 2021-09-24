import tensorflow as tf
from text import symbols


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=500,
        iters_per_checkpoint=1000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding.weight'],

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=True,
        training_files='/content/tacotron2/filelists/meian/train.txt',
        validation_files='/content/tacotron2/filelists/meian/val.txt',
        text_cleaners=['english_cleaners'],

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        sample_rate = 22050,                  # sample rate of source .wavs, used while computing spectrograms, MFCCs, etc.
        num_fft = 1102,                       # number of frequency bins used during computation of spectrograms
        num_mels = 80,                       # number of mel bins used during computation of mel spectrograms
        num_mfcc = 13,                        # number of MFCCs, used just for MCD computation (during training)
        stft_window_ms = 50,                  # size in ms of the Hann window of short-time Fourier transform, used during spectrogram computation
        stft_shift_ms = 12.5,                 # shift of the window (or better said gap between windows) in ms
        griffin_lim_iters = 60,               # used if vocoding using Griffin-Lim algorithm (synthesize.py), greater value does not make much sense
        griffin_lim_power = 1.5,              # power applied to spectrograms before using GL
        normalize_spectrogram = True,         # if True, spectrograms are normalized before passing into the model, a per-channel normalization is used
                                            # statistics (mean and variance) are computed from dataset at the start of training  
        use_preemphasis = True,               # if True, a preemphasis is applied to raw waveform before using them (spectrogram computation)
        preemphasis = 0.97,      

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=32,
        mask_padding=True  # set model's padded outputs to padded values
    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
