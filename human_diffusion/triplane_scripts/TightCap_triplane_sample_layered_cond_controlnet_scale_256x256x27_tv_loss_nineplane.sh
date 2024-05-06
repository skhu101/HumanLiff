#!/usr/bin/bash -l

diff_steps=$1
in_channels=$2
num_channels=$3
out_channels=$4
start_id=$5
master_port=$6

image_size=256
ckpt_num=200000
diff_ckpt=250000
humanliff_path=$(pwd)"/.."
JOB_NAME="TightCap_185_view_107_subject_triplane_${image_size}x${image_size}x27_ckpt_"$ckpt_num"_diff_steps_"$diff_steps"_cond_controlnet_scale_tv_loss_nineplane"

EXPS="exps"
if ! [ -d "$EXPS/$JOB_NAME" ]; then
   mkdir -p $EXPS/$JOB_NAME
fi

num_samples=2

DATA_FLAGS="--data_name tightcap"
MODEL_FLAGS="--image_size $image_size --num_channels $num_channels --num_res_blocks 3 --learn_sigma False --use_scale_shift_norm True --attention_resolutions 32,16,8 --class_cond True --dropout 0"
DIFFUSION_FLAGS="--in_channels $in_channels --out_channels $out_channels --diffusion_steps $diff_steps --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --use_cond True --cond_type controlnet"
SAMPLE_FLAGS="--batch_size 1 --num_samples $num_samples --timestep_respacing 250"

python -m torch.distributed.launch --nproc_per_node=1 --master_port $master_port scripts/triplane_sample_layered.py --log_dir $EXPS/$JOB_NAME --model_path $EXPS/$JOB_NAME/ema_0.9999_$diff_ckpt.pt $SAMPLE_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS --layer_index 0 --data_dir $humanliff_path/recon_NeRF/logs/TightCap_185_view_100_subject_triplane_256x256x27_mlp_2_layer_N_coarse_samples_128_fine_sample_128_no_human_sample_lr_5e-3_tri_plane_lr_1e-1_scale_tv_loss_1e-2_l1_loss_5e-4/200000.tar --start_id $start_id $DATA_FLAGS

wait

python -m torch.distributed.launch --nproc_per_node=1 --master_port $master_port scripts/triplane_sample_layered.py --log_dir $EXPS/$JOB_NAME --model_path $EXPS/$JOB_NAME/ema_0.9999_$diff_ckpt.pt $SAMPLE_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS --sample_npz $EXPS/$JOB_NAME/samples_person_${num_samples}x27x256x256_ckpt_${diff_ckpt}_ema_start_id_$start_id.npz --layer_index 1 --data_dir $humanliff_path/recon_NeRF/logs/TightCap_185_view_100_subject_triplane_256x256x27_mlp_2_layer_N_coarse_samples_128_fine_sample_128_no_human_sample_lr_5e-3_tri_plane_lr_1e-1_scale_tv_loss_1e-2_l1_loss_5e-4/200000.tar --start_id $start_id $DATA_FLAGS

wait 

python -m torch.distributed.launch --nproc_per_node=1 --master_port $master_port scripts/triplane_sample_layered.py --log_dir $EXPS/$JOB_NAME --model_path $EXPS/$JOB_NAME/ema_0.9999_$diff_ckpt.pt $SAMPLE_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS --sample_npz $EXPS/$JOB_NAME/samples_person_pant_${num_samples}x27x256x256_ckpt_${diff_ckpt}_ema_start_id_$start_id.npz --layer_index 2 --data_dir $humanliff_path/recon_NeRF/logs/TightCap_185_view_100_subject_triplane_256x256x27_mlp_2_layer_N_coarse_samples_128_fine_sample_128_no_human_sample_lr_5e-3_tri_plane_lr_1e-1_scale_tv_loss_1e-2_l1_loss_5e-4/200000.tar --start_id $start_id $DATA_FLAGS

wait 

python -m torch.distributed.launch --nproc_per_node=1 --master_port $master_port scripts/triplane_sample_layered.py --log_dir $EXPS/$JOB_NAME --model_path $EXPS/$JOB_NAME/ema_0.9999_${diff_ckpt}.pt $SAMPLE_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS --sample_npz $EXPS/$JOB_NAME/samples_person_pant_shirt_${num_samples}x27x256x256_ckpt_${diff_ckpt}_ema_start_id_$start_id.npz --layer_index 3 --data_dir $humanliff_path/recon_NeRF/logs/TightCap_185_view_100_subject_triplane_256x256x27_mlp_2_layer_N_coarse_samples_128_fine_sample_128_no_human_sample_lr_5e-3_tri_plane_lr_1e-1_scale_tv_loss_1e-2_l1_loss_5e-4/200000.tar --start_id $start_id $DATA_FLAGS
