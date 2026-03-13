# Supplementary materials

**TQ-ViT-main.zip contains all the code for the paper "TQ-ViTs: Accelerating Vision Transformer via Token-wise Reparametrization".**

### Model file
`TQ_ViT`: pytorch-image-models/timm/models/vision_transformer_tq.py

`TQ-Swin`: pytorch-image-models/timm/models/swin_transformer_tq.py

`TQ-Convformer` `TQ-CAformer` `TQ-PoolformerV2`: pytorch-image-models/timm/models/metaformer_tq.py

`TQ Block`: pytorch-image-models/timm/models/tq_block.py

`ToMe`: pytorch-image-models/ToMe.py

### Train and test scripts
`train_tq_model.py`: pytorch-image-models/train_tq_model.py

`validate.py`: pytorch-image-models/validate.py

`only_val_speed.py`:pytorch-image-models/only_val_speed.py

Please note that before testing the speed with the `only_val_speed.py` script, you should comment out line 76 in the tq_block.py file. This line is specifically used for counting codebook utility statistics.

## Training
Hardware:

 `
 2*Intel(R) Xeon(R) Gold 6346 CPU @ 3.10GHz; 
 4*Nvidia A100-80G
 `

Environments:

 `
 pytorch==3.8 torch==2.4.1 torchvision==0.19.1
 `

```sh
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node=2 --master_port=29579 train_tq_model.py -j 24 --dict-dim 4 --tq-level 3 3 3 3 --model tq_vit_small_patch16_224 --teacher-model vit_small_patch16_224 --output ../output --dataset imagenet1k --initial-checkpoint ./pretrained_weights/vit_small_patch16_224.augreg_in21k_ft_in1k.pth  --amp -b 1024 --grad-accum-steps 2 --lr 8e-4 --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5 --data-dir /path/imagenet1k
```

## Test

All efficiency metrics are tested on an `NVIDIA GeForce RTX 3080 Ti`.

Please note that the **`--reparam`** option will perform token reparameterization on TQ-ViT, corresponding to the "inference phase" architecture in paper. **`--tome --tome-r 16`** option will perform token merging on TQ-ViT,  corresponding to the "TQ-ViT+ToMe(r=16)" in paper.

For instance, you can test the model using the provided weights: *../tq_weight/tq_vit_small_patch16_224-79.34.pth.tar*. Due to supplement file size limitations, we have only uploaded the TQ-ViT/S weights. The weights for the other TQ models will be publicly available on GitHub.

```sh
CUDA_VISIBLE_DEVICES=1 python validate.py --model tq_vit_small_patch16_224  --dataset imagenet1k --model-kwargs dic_dim=4 tq_level=[3,3,3,3] --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5 --checkpoint ../tq_weight/tq_vit_small_patch16_224-79.34.pth.tar --data-dir /path/imagenet1k --reparam  
```
then, you will get result of `TQ-ViT/S` inference phase:
```sh
{
    "dataset": "imagenet1k",
    "model": "tq_vit_small_patch16_224",
    "top1": 79.338,
    "param_count": "12.37M",
    "FLOPs": "2.523G",
    "codebook_utilization": "91.69%"
}
```

```sh
CUDA_VISIBLE_DEVICES=1 python validate.py --model tq_vit_small_patch16_224  --dataset imagenet1k --model-kwargs dic_dim=4 tq_level=[3,3,3,3] --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5 --checkpoint ../tq_weight/tq_vit_small_patch16_224-79.34.pth.tar --data-dir /path/imagenet1k --reparam  --tome --tome-r 16
```

then, you will get rsult of `TQ-ViT/S+ToMe(r=16)` inference phase:
```sh
{
    "dataset": "imagenet1k",
    "model": "tq_vit_small_patch16_224",
    "top1": 78.4,
    "param_count": "12.37M",
    "FLOPs": "2.082G",
    "codebook_utilization": "91.63%"
}

```

```sh
CUDA_VISIBLE_DEVICES=1 python only_val_speed.py --model tq_vit_small_patch16_224 --model-kwargs dic_dim=4 tq_level=[3,3,3,3] --reparam
```
then, you will get **speed (images/sec)** of TQ-ViT/S inference phase:
```sh
{
    "model": "tq_vit_small_patch16_224",
    "param_count": "12.37M",
    "FLOPs": "2.523G",
    "Speed": "4407.24 images/sec"
}
```