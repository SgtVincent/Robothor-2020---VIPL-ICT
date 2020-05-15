#!/usr/bin/env bash
python main.py \
  --title demo_trajectory_0514_073 \
  --model BaseModel \
  --offline_data_dir /home/chenjunting/Robothor_data \
  --gpu-ids  0 1  \
  --workers 6 \
  --save-model-dir demo_trajectory_0514_073 \
  --data_source robothor \
  --grid_size 0.125 --rotate_by 30 --state_decimal 3 \
  --images_file_name resnet18.hdf5 \
  --train_scenes [1-5] \
  --seed 10 \
  --lr 0.0001 \
  --step_penalty -0.005 \
  --max-episode-length 50 \
  --max_ep 3100000 \
  --pinned_scene \
  --curriculum_learning \
  --demo_trajectory \
  --demo_trajectory_freq 1 \
  --verbose \
  --scene_types FloorPlan_Train1
#  FloorPlan_Train2 FloorPlan_Train3
#  FloorPlan_Train4 FloorPlan_Train5 FloorPlan_Train6 FloorPlan_Train7 \
#  FloorPlan_Train8 FloorPlan_Train9 FloorPlan_Train10 FloorPlan_Train11  FloorPlan_Train12\

#  --load_checkpoint checkpoint.dat \
#  --load_checkpoint /home/chenjunting/trained_sp1e-3_decay08_1m/base_train_sp1e-3_decay08_51295441_3000000_2020-04-28_10_58_43.dat \
