#!/bin/bash

PIPELINE_CONFIG_PATH="models/model/sketchup.config"
MODEL_DIR="models/model"
NUM_TRAIN_STEPS=5
SAMPLE_1_OF_N_EVAL_EXAMPLES=1

export PYTHONPATH=$PYTHONPATH:`pwd`/..:`pwd`/../slim

echo "Starting to train... `date`"

pipenv run python ../object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr \
    &> output.txt

echo "Done training! refer to output.txt `date`"

