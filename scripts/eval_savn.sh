#!/usr/bin/env bash
python /home/chenjunting/repos/savn/main.py --eval \
    --test_or_val test \
    --episode_type TestValEpisode \
    --load_model pretrained_models/savn_pretrained.dat \
    --model SAVN \
    --results_json savn_test.json

cat /home/chenjunting/repos/savn/savn_test.json