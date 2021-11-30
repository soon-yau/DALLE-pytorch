python3 train_dalle.py --text_seq_len 128 --batch_size 8 --epochs 5 \
--learning_rate 0.003  --lr_decay \
--depth 8  \
--image_text_folder ~/datasets/mannequin/ \
--wandb_name dalle_mannequin \
--taming \
--vqgan_config_path ../VQGAN-CLIP/checkpoints/vqgan_gumbel_f8_8192.yaml \
--vqgan_model_path ../VQGAN-CLIP/checkpoints/vqgan_gumbel_f8_8192.ckpt \
--cuda cuda:1  \
--hug --bpe_path /home/soon/github/PoseGuidedTextToImage/tokenizer-mannequin.json \
--dalle_output_file_name dalle_mannequin_gumbel_lr_3e_3
#--dalle_path dalle_mannequin_depth_8_gumbel.pt



