#!/usr/bin/env bash
python main.py \
    --title base_train \
    --model BaseModel \
    --gpu-ids 1 2 3 \
    --workers 12 \
    --save-model-dir trained_base_2020_0306