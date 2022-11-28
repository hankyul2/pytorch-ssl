# Self-Supervised Learning 

## Tutorial
1. clone
2. run following (# gpu=4, # batch=32)
    ```python
    torchrun --nproc_per_node=4 multi_train.py mocov1 -c 0,1,2,3 --use-wandb
    ```