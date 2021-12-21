python3 train_dalle.py \
--text_seq_len 200 --batch_size 8 --epochs 5 \
--learning_rate 0.0006  --lr_decay \
--depth 8  \
--image_text_folder ~/datasets/mannequin/images \
--data_file ~/datasets/mannequin/pose.pickle \
--data_type pose \
--pose_format keypoint \
--pose_dim 6 \
--merge_images \
--wandb_name dalle_mannequin_pose_multi \
--taming \
--vqgan_config_path ../VQGAN-CLIP/checkpoints/vqgan_gumbel_f8_8192.yaml \
--vqgan_model_path ../VQGAN-CLIP/checkpoints/vqgan_gumbel_f8_8192.ckpt \
--cuda cuda:1  \
--hug --bpe_path /home/soon/github/PoseGuidedTextToImage/tokenizer-mannequin.json \
--dalle_output_file_name dalle_mannequin_pose_multi \
--display_freq 200 
#--dalle_path dalle_mannequin_pose_keypoint_aug.pt
