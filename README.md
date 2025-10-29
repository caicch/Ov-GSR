# Ov-GSR
The official code for From Semantics, Scene to Instance-awareness: Distilling Foundation Model for Open-vocabulary Grounded Situation Recognition

# Dataset
Download the [images](https://swig-data-weights.s3.us-east-2.amazonaws.com/images_512.zip), and [datasets](https://drive.google.com/drive/folders/1ftpQVou9zgPWqL2X7bbqeNPXwoM50hGv?usp=sharing).

# Pre-process image and text to train on low-memory GPU
  1. Run BLIP_feat.py to encode and save the BLIP image features
  2. Run CLIP_feat.py to encode and save the CLIP image features
  3. Run clip_4_text.py to encode and save caption features
  4. Run action_embedding.py and class_embedding.py to save action and object text features

# Training
Use train_L10_with_obj_cap_refined.json for training.
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py   --batch_size 32 --epochs 10 --num_workers 4 --num_tr_dec_layers 1 --dropout 0.10 --hidden_dim 512 --output_dir Ov-GSR
```

# Evaluation
Change the testing dataset in datasets/swig.py line 421 for val and line 422 for test. dev.json, dev_L10_unseen_with_obj_cap_refined.json, dev_L20_rare_with_obj_cap_refined.json for dev set. test.json, test_L10_unseen_with_obj_cap_refined.json, test_L20_rare_with_obj_cap_refined.json for test set.
```
python main.py --saved_model ckpt/checkpoint.pth --output_dir Ov-GSR --dev True  # Evaluation on develpment set
python main.py --saved_model ckpt/checkpoint.pth --output_dir Ov-GSR --test True # Evaluation on test set
```

# Acknowledgement
This code is built based on the:
Collaborative Transformers for Grounded Situation Recognition [Coformer](https://github.com/jhcho99/CoFormer). 

Open Scene Understanding: Grounded Situation Recognition Meets Segment Anything for Helping People with Visual Impairments [OpenSU](https://github.com/RuipingL/OpenSU?tab=readme-ov-file)

# Citation
```
@inproceedings{cai2025semantics,
  title={From Semantics, Scene to Instance-awareness: Distilling Foundation Model for Open-vocabulary Grounded Situation Recognition},
  author={Cai, Chen and Liu, Tianyi and Gao, Jianjun and Liu, Wenyang and Wu, Kejun and Wang, Ruoyu and Wang, Yi and Liew, Soo Chin},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={392--401},
  year={2025}
}
```
