#!/usr/bin/env bash

python data_utils/create_curriculum_meta_Robothor.py \
  --out_dir /home/ubuntu/chenjunting/savn/data/curriculum_meta \
  --num_process 15

#python data_utils/create_curriculum_meta_Robothor.py \
#  --out_dir /home/ubuntu/chenjunting/savn/data/valmeta \
#  --num_process 3 \
#  --scenes FloorPlan_Val1_1 FloorPlan_Val1_2 FloorPlan_Val1_3 FloorPlan_Val1_4 FloorPlan_Val1_5 FloorPlan_Val2_1 FloorPlan_Val2_2 FloorPlan_Val2_3 FloorPlan_Val2_4 FloorPlan_Val2_5 FloorPlan_Val3_1 FloorPlan_Val3_2 FloorPlan_Val3_3 FloorPlan_Val3_4 FloorPlan_Val3_5