#!/bin/bash
# soundReal s1/s3 学習 (GPU1): act All-s3 -> diffusion All-s3 -> diffusion Shake-s3
export CUDA_VISIBLE_DEVICES=1
export STEPS=200000
export SAVE_FREQ=10000
export SEED=0

export POLICY=act
export DATASET_NAME=soundRealAll-m4-f10-s3-p0
uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME}_seed${SEED} --job_name=${POLICY}_${DATASET_NAME}_seed${SEED} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --dataset.video_backend=pyav --batch_size=8 --steps=$STEPS --save_freq=$SAVE_FREQ --seed=$SEED --num_workers=8

export POLICY=diffusion
export DATASET_NAME=soundRealAll-m4-f10-s3-p0
uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME}_seed${SEED} --job_name=${POLICY}_${DATASET_NAME}_seed${SEED} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --dataset.video_backend=pyav --batch_size=32 --steps=$STEPS --save_freq=$SAVE_FREQ --seed=$SEED --num_workers=8 --policy.use_separate_rgb_encoder_per_camera=true

export POLICY=diffusion
export DATASET_NAME=soundRealShake-m4-f10-s3-p0
uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME}_seed${SEED} --job_name=${POLICY}_${DATASET_NAME}_seed${SEED} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --dataset.video_backend=pyav --batch_size=32 --steps=$STEPS --save_freq=$SAVE_FREQ --seed=$SEED --num_workers=8 --policy.use_separate_rgb_encoder_per_camera=true
