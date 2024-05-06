#!/usr/bin/bash -l

diff_steps=$1
gpu_num=$2
in_channels=$3
num_channels=$4
out_channels=$5
batch_size=$6
num_subjects=$7
ckpt_num=$8

image_size=256
humanliff_path=$(pwd)"/.."
JOB_NAME="TightCap_185_view_"$num_subjects"_subject_triplane_${image_size}x${image_size}x27_ckpt_"$ckpt_num"_diff_steps_"$diff_steps"_cond_controlnet_scale_tv_loss_nineplane"


EXPS="exps"
if ! [ -d "$EXPS/$JOB_NAME" ]; then
   mkdir -p $EXPS/$JOB_NAME
fi

DATA_FLAGS="--data_name tightcap"
MODEL_FLAGS="--image_size $image_size --num_channels $num_channels --num_res_blocks 3 --learn_sigma False --use_scale_shift_norm True --attention_resolutions 32,16,8 --class_cond True --dropout 0"
DIFFUSION_FLAGS="--in_channels $in_channels --out_channels $out_channels --diffusion_steps $diff_steps --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 5e-5 --batch_size $batch_size --save_interval 50000 --use_amp True --num_subjects $num_subjects --use_cond True --cond_type controlnet --microbatch 2"

cp $humanliff_path"/recon_NeRF/data/TightCap/human_list.txt" $humanliff_path"/recon_NeRF/logs/TightCap_185_view_100_subject_triplane_256x256x27_tv_loss_1e-2_l1_loss_5e-4/human_list.txt"

python -m torch.distributed.launch --nproc_per_node=$gpu_num scripts/image_train.py --data_dir $humanliff_path"/recon_NeRF/logs/TightCap_185_view_100_subject_triplane_256x256x27_tv_loss_1e-2_l1_loss_5e-4/"$ckpt_num".tar" --log_dir $EXPS/$JOB_NAME $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

