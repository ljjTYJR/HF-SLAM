#!/bin/bash

SCENE=$1
export SCENE

cd /home/airlab/nkeetha/4d/4DTrack

echo "Evaluating scene number ${SCENE} with seed 0"
python3 -u scripts/eval_novel_view.py configs/scannetpp/slam_eval.py