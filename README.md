# Supplementary materials

**TQViT-main.zip contains all the code for the paper "TQ-ViTs: Accelerating Vision Transformer via Token-wise Reparametrization".**

### Model file
`TQ_ViT`: pytorch-image-models/timm/models/vision_transformer_tq.py

`TQ-Swin`: pytorch-image-models/timm/models/swin_transformer_tq.py

`TQ-Poolformer`: pytorch-image-models/timm/models/poolformer_tq.py

`TQ-Convformer` `TQ-CAformer` `TQ-PoolformerV2`: pytorch-image-models/timm/models/metaformer_tq.py

`TQ Block`: pytorch-image-models/timm/models/TQ_block.py

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

### ImageNet-1k:
```sh
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node=2 --master_port=29579 trainvqvit.py -j 12 --tqtype TQ --dict-dim 4 --tq-level 3 3 3 3 --model vqvit_base_patch16_224 --teacher-model vit_base_patch16_224 --output ../output/official_dis --dataset imagenet1k --initial-checkpoint ./pretrained_weights/vit_base_patch16_224.augreg_in21k_ft_in1k.pth  --amp -b 800 --grad-accum-steps 3 --lr 8e-4 --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5 --data-dir /path/imagenet1k
```

## Test
Please note that the *--reparam* option will perform token reparameterization on TQ-ViT, corresponding to the "inference phase" architecture in paper.

### ImageNet-1k:
```sh
CUDA_VISIBLE_DEVICES=3 python validate.py --model tqvit_small_patch16_224 --dataset imagenet1k --data-dir /path/imagenet1k --checkpoint /path/TQViT/tqvit_small_patch16_224.pth.tar --model-kwargs tq_type='TQ' dic_dim=4 tq_levle=[3,3,3,3] --reparam
```

```sh
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --master_port=29577 trainvqvit.py -j 12 --tqtype TQ --dict-dim 4 --tq-level 5 5 5 5 --model tq_convformer_s18 --teacher-model convformer_s18 --output ../output/swin --dataset imagenet1k --initial-checkpoint ./pretrained_weights/convformer_s18.sail_in22k_ft_in1k.pth  --amp -b 512 --grad-accum-steps 4 --lr 3e-4 --crop-pct 0.9 --epochs 100
```

For instance, you can test the model using the provided weights: *output/tqvit_tiny_patch16_224-Acc7072-QuantLevel555-lr4e-4-wd1e-3-310epoch.pth.tar.*
```sh
CUDA_VISIBLE_DEVICES=0 python validate.py --model tqvit_tiny_patch16_224  --dataset imagenet1k --model-kwargs dic_dim=3 tq_type='TQ' tq_levle=[5,5,5] -j 24 --checkpoint ../output/tqvit_tiny_patch16_224-Acc70.72-QuantLevel555-lr4e-4-wd1e-3-310epoch.pth.tar --data-dir /in1k_data_path/ImageNet2012/  --reparam
```

then, you will get:
```sh
 "dataset": "imagenet1k",
    "checkpoint": "../output/tqvit_tiny_patch16_224-Acc70.72-QuantLevel555-lr4e-4-wd1e-3-310epoch.pth.tar",
    "model": "tqvit_tiny_patch16_224",
    "top1": 70.724,
    "param_count": "3.78M",
    "FLOPs": "738.026M",
    "img_size": [
        3,
        224,
        224
    ],
    "crop_pct": 0.875,
    "interpolation": "bicubic",
    "params_thop": "3.025M",
    "average_batchtime": "0.082s,   12.27/s, FPS=3140.4",
    "codebook_utilization": "81.20%"
```