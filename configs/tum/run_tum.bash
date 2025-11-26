#!/bin/bash

SEED=$1
export SEED

cd /storage2/datasets/nkeetha/4d/4DTrack

for scene in 0 1 2 4 3
do
    SCENE_NUM=${scene}
    export SCENE_NUM
    echo "Running scene number ${SCENE_NUM} with seed ${SEED}"
    python3 -u scripts/slam.py configs/tum/tum_eval.py
done