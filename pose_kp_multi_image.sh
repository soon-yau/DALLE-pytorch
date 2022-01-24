python3 train_dalle.py \
--text_seq_len 225 --batch_size 12 --epochs 5 \
--learning_rate 0.0006  --lr_decay \
--depth 8  \
--image_text_folder ~/datasets/ \
--data_file ~/datasets/mannequin_multi/pose_multi_train.pickle \
--test_data_file ~/datasets/mannequin_multi/pose_multi_test.pickle \
--data_type pose \
--pose_format image \
--pose_seq_len 256 \
--pose_dim 256 \
--wandb_name dalle_mannequin_pose_final \
--taming \
--vqgan_config_path ../VQGAN-CLIP/checkpoints/vqgan_mannequin.yaml \
--vqgan_model_path ../VQGAN-CLIP/checkpoints/vqgan_mannequin.ckpt \
--cuda cuda:2  \
--hug --bpe_path /home/soon/github/PoseGuidedTextToImage/tokenizer-mannequin.json \
--dalle_output_file_name checkpoint/final/dalle_mannequin_pose_image_256 \
--display_freq 50 \
#--dalle_path dalle_mannequin_pose_multi_4.pt
