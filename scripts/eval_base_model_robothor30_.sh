#!/usr/bin/env bash
python /home/chenjunting/repos/savn/main.py \
  --eval \
  --test_or_val test \
  --episode_type TestValEpisode \
  --load_model pretrained_models/nonadaptivea3c_pretrained.dat \
  --model BaseModel \
  --scene_types FloorPlan_Val1 FloorPlan_Val2 FloorPlan_Val3 \
  --results_json savn_test.json

cat /home/chenjunting/repos/savn/savn_test.json