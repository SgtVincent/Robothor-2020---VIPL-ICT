#!/usr/bin/env bash
python main.py \
    --title savn_train \
    --model SAVN \
    --gpu-ids 0 1 2 \
    --workers 18 \
    --action_space 4
