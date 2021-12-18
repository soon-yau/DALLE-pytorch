python3 train_dalle.py \
--text_seq_len 128 --batch_size 4 --epochs 20 \
--learning_rate 0.0006  --lr_decay \
--depth 12  \
--image_text_folder ~/datasets/deepfashion/img_256/ \
--data_file ~/datasets/deepfashion/pose_deepfashion.pickle \
--data_type pose \
--pose_format keypoint \
--pose_dim 3 \
--wandb_name dalle_deepfashion \
--taming \
--vqgan_config_path ../VQGAN-CLIP/checkpoints/vqgan_gumbel_f8_8192.yaml \
--vqgan_model_path ../VQGAN-CLIP/checkpoints/vqgan_gumbel_f8_8192.ckpt \
--cuda cuda:2  \
--hug --bpe_path /home/soon/github/PoseGuidedTextToImage/tokenizer-mannequin.json \
--dalle_output_file_name dalle_deepfashion_depth_12 \
--display_freq 100 
#--dalle_path dalle_mannequin_pose_keypoint_aug.pt
