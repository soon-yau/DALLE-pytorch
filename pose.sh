python3 train_dalle.py --text_seq_len 128 --batch_size 8 --epochs 5 \
--learning_rate 0.003  --lr_decay \
--depth 8  \
--image_text_folder ~/datasets/mannequin/ \
--data_file ~/datasets/mannequin/pose.pickle \
--data_type pose \
--wandb_name dalle_mannequin_pose \
--taming \
--vqgan_config_path ../VQGAN-CLIP/checkpoints/vqgan_imagenet_f16_16384.yaml \
--vqgan_model_path ../VQGAN-CLIP/checkpoints/vqgan_imagenet_f16_16384.ckpt \
--cuda cuda:1  \
--hug --bpe_path /home/soon/github/PoseGuidedTextToImage/tokenizer-mannequin.json \
--dalle_output_file_name dalle_mannequin_pose
#--dalle_path dalle_mannequin_depth_8_gumbel.pt



