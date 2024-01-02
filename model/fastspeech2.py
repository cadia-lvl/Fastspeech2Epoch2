import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        # Phoneme Encoder
        self.encoder = Encoder(model_config)
        
        # Predict the epoch duration
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        
        # Decoder to generate mel, phase and epoch length
        self.decoder = Decoder(model_config)

        # Convert the decoder output into mel
        self.mel_postnet = PostNet(n_input=model_config['transformer']['decoder_hidden'], n_output=151)
        
        # Convert the decoder output into phase
        self.phase_postnet = PostNet(n_input=model_config['transformer']['decoder_hidden'], n_output=151)
        
        # Predict epoch lenghts from decoder output
        self.epolen_postnet = PostNet(n_input=model_config['transformer']['decoder_hidden'], n_output=256)

        self.speaker_emb = None
        
        # self.epolen_layer = nn.Linear(20, 30)
        
        # For ljspeech, this is False
        if model_config["multi_speaker"]:
            with open(
                os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json"), "r",
            ) as f:
                n_speaker = len(json.load(f))
                    
            self.speaker_emb = nn.Embedding(n_speaker, model_config["transformer"]["encoder_hidden"])

    def forward(
        self,
        speakers, 
        texts,
        text_lens,
        max_text_len,
        mels=None, 
        phases=None, 
        acoustic_lens=None, 
        max_acoustic_len=None,
        epochdurs=None, 
        epochlens=None, 
    ):
        text_masks = get_mask_from_lengths(text_lens, max_text_len)
        acoustic_masks = (
            get_mask_from_lengths(acoustic_lens, max_acoustic_len)
            if acoustic_lens is not None
            else None
        )

        output = self.encoder(texts, text_masks)    # torch.Size([2, 374, 256])

        (
            output,
            log_d_predictions,
            d_rounded,
            acoustic_lens,
            acoustic_masks,
        ) = self.variance_adaptor(
            output,
            text_masks,
            acoustic_masks,
            max_acoustic_len,
            epochdurs
        )

        output, mel_masks = self.decoder(output, acoustic_masks)

        mel_prediction = self.mel_postnet(output)
        phase_prediction = self.phase_postnet(output)
        epochlen_prediction = self.epolen_postnet(output)
        
        # epochlen_prediction = self.sp(epochlen_prediction)

        return (
            log_d_predictions,
            mel_prediction,
            phase_prediction,
            epochlen_prediction,
            d_rounded,
            text_masks,
            acoustic_masks,
            text_lens,
            acoustic_lens,
        )