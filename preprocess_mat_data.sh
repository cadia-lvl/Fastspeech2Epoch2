#!/bin/bash
python3 preprocess_mat_data.py --mat_path=/work/gfang/ljspeech/LJSpeech_raw \
    --tgt_path=/work/gfang/FastSpeech2Epoch2/preprocessed_data/LJSpeech/TextGrid/TextGrid/LJSpeech \
    --raw_transcrption_path=/work/gfang/ljspeech/LJSpeech-1.1/metadata.csv \
    --save_path=/work/gfang/ljspeech/epoch_processed/output