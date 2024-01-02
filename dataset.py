import json
import math
import os

import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence, symbols
from utils.tools import pad_1D, pad_2D


class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(filename)
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
            
        # The reason we use sort is to allow the model for learn from the short samples
        self.sort = sort
        self.drop_last = drop_last
        
        # self.phase_min = -540.3069518961524
        # self.phase_max = 470.14039400175034

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx]))

        # mel
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        # mel = np.log(mel+1)
        mel = mel
        
        # phase
        phase_path = os.path.join(
            self.preprocessed_path,
            "phase",
            "{}-phase-{}.npy".format(speaker, basename),
        )
        phase = np.load(phase_path)
        # phase = (phase - self.phase_min) / (self.phase_max-self.phase_min)
        # phase = phase * 10
        # phase = 1 / (1 + np.exp(-phase))
        
        # epoch amount for each phoneme
        epochdur_path = os.path.join(
            self.preprocessed_path,
            "epoch_dur",
            "{}-epochdur-{}.npy".format(speaker, basename),
        )
        epochdur = np.load(epochdur_path)
        
        # epoch length
        epochlen_path = os.path.join(
            self.preprocessed_path,
            "epoch_len",
            "{}-epochlen-{}.npy".format(speaker, basename),
        )
        epochlen = np.load(epochlen_path)

        
        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "phase": phase,
            "epochdur": epochdur,
            "epochlen": epochlen,
        }

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f:
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        phases = [data[idx]["phase"] for idx in idxs]
        epochdurs = [data[idx]["epochdur"] for idx in idxs]
        epochlens = [data[idx]["epochlen"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts]).reshape(-1, 1)
        mel_lens = np.array([mel.shape[-1] for mel in mels]).reshape(-1, 1)

        speakers = np.array(speakers)
        
        texts = pad_1D(texts)
        
        mels = pad_2D(mels)
        phases = pad_2D(phases)
        
        epochdurs = pad_1D(epochdurs)
        epochlens = pad_1D(epochlens)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens)[0],  # Just need a number
            mels,
            phases,
            mel_lens,
            max(mel_lens)[0],
            epochdurs,
            epochlens
        )

    def collate_fn(self, data):
        
        '''
            Output:
                ids (e.g. LJ008-0031),
                raw_texts (word-based transcription),
                speakers (speaker id),
                texts (phoneme ids),
                text_lens (lenght of texts),
                max_text_lengths,
                mels,
                phases,
                acoustic_lens (the length of mels or phases),
                max_acoustic_lengths,
                epochdurs,
                epochlens
        '''
        
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))

        return (basename, speaker_id, phone, raw_text)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])

        texts = pad_1D(texts)

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens)


if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from utils.utils import to_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_config = yaml.load(
        open("./config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/LJSpeech/train.yaml", "r"), Loader=yaml.FullLoader
    )

    train_dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )