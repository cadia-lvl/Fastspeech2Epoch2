import os
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt


matplotlib.use("Agg")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(data, device):
    
    (
        ids,
        raw_texts,
        speakers,
        texts,
        text_lens,
        max_text_len,
        mels,
        phases,
        acoustic_lens,
        max_acoustic_len,
        epochdurs,
        epochlens
    )    = data

    # I write the shape of each item (just an example), to have an idea of the data
    speakers = torch.from_numpy(speakers).long().to(device) # torch.Size([1])
    texts = torch.from_numpy(texts).long().to(device)   # torch.Size([1, 339]), 1 batch, 339 phonemes
    text_lens = torch.from_numpy(text_lens).to(device)  # torch.Size([1, 1])
    mels = torch.from_numpy(mels).float().to(device)    # torch.Size([1, 80, 1725])
    phases = torch.from_numpy(phases).float().to(device)    # torch.Size([1, 80, 1725]), same as mel
    acoustic_lens = torch.from_numpy(acoustic_lens).to(device)  # torch.Size([1, 1])
    epochdurs = torch.from_numpy(epochdurs).long().to(device) # torch.Size([1, 106])
    epochlens = torch.from_numpy(epochlens).float().to(device)   # torch.Size([1, 1725])
    
    

    return (
        ids,
        raw_texts,
        speakers,
        texts,
        text_lens,
        max_text_len,   # just a number
        mels,
        phases,
        acoustic_lens,
        max_acoustic_len,   # just a number
        epochdurs,
        epochlens,
    )




def log(
    logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/mel_postnet_loss", losses[2], step)
        logger.add_scalar("Loss/pitch_loss", losses[3], step)
        logger.add_scalar("Loss/energy_loss", losses[4], step)
        logger.add_scalar("Loss/duration_loss", losses[5], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def synth_one_sample(targets, predictions, vocoder, model_config, preprocess_config):

    basename = targets[0][0]
    src_len = predictions[8][0].item()
    mel_len = predictions[9][0].item()
    mel_target = targets[6][0, :mel_len].detach().transpose(0, 1)
    mel_prediction = predictions[1][0, :mel_len].detach().transpose(0, 1)
    duration = targets[11][0, :src_len].detach().cpu().numpy()
    if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
        pitch = targets[9][0, :src_len].detach().cpu().numpy()
        pitch = expand(pitch, duration)
    else:
        pitch = targets[9][0, :mel_len].detach().cpu().numpy()
    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
        energy = targets[10][0, :src_len].detach().cpu().numpy()
        energy = expand(energy, duration)
    else:
        energy = targets[10][0, :mel_len].detach().cpu().numpy()

    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        stats = stats["pitch"] + stats["energy"][:2]

    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), pitch, energy),
            (mel_target.cpu().numpy(), pitch, energy),
        ],
        stats,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction, basename


def synth_samples(targets, predictions, vocoder, model_config, preprocess_config, path):

    basenames = targets[0]
    for i in range(len(predictions[0])):
        basename = basenames[i]
        src_len = predictions[8][i].item()
        mel_len = predictions[9][i].item()
        mel_prediction = predictions[1][i, :mel_len].detach().transpose(0, 1)
        duration = predictions[5][i, :src_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = predictions[2][i, :src_len].detach().cpu().numpy()
            pitch = expand(pitch, duration)
        else:
            pitch = predictions[2][i, :mel_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = predictions[3][i, :src_len].detach().cpu().numpy()
            energy = expand(energy, duration)
        else:
            energy = predictions[3][i, :mel_len].detach().cpu().numpy()

        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"][:2]

        fig = plot_mel(
            [
                (mel_prediction.cpu().numpy(), pitch, energy),
            ],
            stats,
            ["Synthetized Spectrogram"],
        )
        plt.savefig(os.path.join(path, "{}.png".format(basename)))
        plt.close()

    from .model import vocoder_infer

    mel_predictions = predictions[1].transpose(1, 2)
    lengths = predictions[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, basenames):
        wavfile.write(os.path.join(path, "{}.wav".format(basename)), sampling_rate, wav)


def plot_mel(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, pitch, energy = data[i]
        pitch = pitch * pitch_std + pitch_mean
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato")
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    # item shape in inputs: [80, len]
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[1] > max_len:
            raise ValueError("not max_len")

        x_padded = np.pad(
            x, 
            ((0, 0), (0, max_len - np.shape(x)[1])), 
            mode="constant", constant_values=PAD
        )
        
        return x_padded

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[1] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def modify_length(input, maxlen):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[1] > max_len:
            raise ValueError("not max_len")

        x_padded = np.pad(
            x, 
            ((0, 0), (0, max_len - np.shape(x)[1])), 
            mode="constant", constant_values=PAD
        )
        
        return x_padded
    
    input_len = input.shape[-1]
    
    if input_len > maxlen:
        return input[:, :maxlen]
    elif input_len == maxlen:
        return input
    else:
        return pad(input, maxlen)


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded

def plot_spectrograms(ground_truth, prediction, save_path, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # Create a figure with 2 subplots, one for each spectrogram

    # Plotting the ground truth spectrogram
    if title is not None:
        axs[0].set_title(f'{title} (Ground Truth)')
    axs[0].set_ylabel(ylabel)
    axs[0].imshow(ground_truth, origin="lower", aspect="auto", interpolation="nearest")

    # Plotting the prediction spectrogram
    axs[1].set_title('Prediction' if title is None else f'{title} (Prediction)')
    axs[1].set_ylabel(ylabel)
    axs[1].imshow(prediction, origin="lower", aspect="auto", interpolation="nearest")

    plt.tight_layout()  # Adjust the layout so that plots do not overlap
    plt.savefig(save_path)  # Save the figure to the specified path
    plt.close()  # Close the figure after saving to free up memory
    
def plot_lines(groundtruth, prediction, save_path):
    
    gt_len = list(range(groundtruth.shape[0]))
    pred_len = list(range(prediction.shape[0]))
    
    # Plotting groundtruth (red) and prediction (blue) lines
    plt.plot(gt_len, groundtruth, 'r', label='groundtruth')
    plt.plot(pred_len, prediction, 'b', label='prediction')
    
    # Setting legends, title, and labels
    plt.legend(loc='upper right')
    plt.title('Groundtruth vs Prediction')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    
    # Save the figure to the specified path
    plt.savefig(save_path)
    plt.close()