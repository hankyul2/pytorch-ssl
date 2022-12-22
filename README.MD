# Pytorch-SSL

This is pytorch-self-supervised-learning repository.



Table of contents:

1. tutorial
2. experimental result



## Tutorial

### 1. how to pre-train your model

1. clone this repo.

   ```bash
   git clone https://github.com/hankyul2/pytorch-ssl.git
   cd pytorch-ssl
   pip3 install requirements.txt 
   ```
   
2. run following command.



### 2. how to fine-tune your model

1. run following command.



### 3. how to validate your model

1. run following command.
   
   KNN classifier
   
   ```bash
   # Single-GPU
   python3 valid.py -es imagenet1k_knn_224_v1 -ws dino_official
   # Multi-GPU 
   torchrun --nproc_per_node=4 --master_port=12345 valid.py -es imagenet1k_knn_224_v1 -ws dino_official
   ```

   FC classifier

   ```bash
   # Single-GPU
   python3 valid.py -es imagenet1k_fc_224_v1 -ws dino_official
   # Multi-GPU 
   torchrun --nproc_per_node=4 --master_port=12345 valid.py -es imagenet1k_fc_224_v1 -ws dino_official
   ```
   
   *Tips*

   1. For large sized train dataset, we recommend to run multi-gpu command because, extracting whole features takes too much time in single gpu.
   2. If you have extracted features once, please specify the extracted feature path at `feature_path` in `config/valid.json`.
   3. If you don't specify weight settings (`-ws`), whole weight settings will be used in validation.
   4. If you don't specify model name (`-m`), whole model in `model_weight` will be used in validation.