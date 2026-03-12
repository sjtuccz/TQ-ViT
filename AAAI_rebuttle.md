 **TQ:** The codebook is predefined and does not introduce any learnable parameters.To integrate the TQ block with Layernorm, we enhance the vanilla TQ by introducing a trainable scale factor, initialized within [0, 1].The quantization level is set to [3,3,3,3].More details will be included.
||Throughput(imgs/s)|Memory(G)|
|-|-|-|
|ViT/T-16|755.3|1.80|
|TQ-ViT-T/16|911.3 (+17.1%)|1.70|
|ViT/S-16|313.4|2.45|
|TQ-ViT/S-16|431.1(+27.3%)|2.31|
|ViT/B/16|108|3.83|
|TQ-ViT-B/16|171.1(+36.9%)|3.48|

**Metrics:** As shown above,our method achieves promising efficiency gains in throughput and memory footprint.
||Acc(%)|FLOPs(G)|#Param(M)|Throughput(imgs/s)|
|-|-|-|-|-|
|PoolFormer-S12|77.2|1.813|11.92|614.2|
|TQ-PoolFormer-S12|75.67|1.025(-47%)|7.4(-38%)|686.7(+11%)|
|ViT/S-32 (384)|77.26|3.45|22.92|403.1|
|TQ-ViT/S-32 (384)|76.38|1.913(-45%)|13.24(-42%)|514.4(+22%)|

**High-resolution inputs:** As shown above,our method demonstrates competitive performance with efficiency gains on high-resolution inputs.
||in1kAcc(%)|FLOPs(G)|#Param(M)|
|-|-|-|-|
|**TQ-Poolformer-S24**|78.5|1.8|12.4|
|PoolFormer-S12(2022)|77.2|1.9|12.0|
|**TQ-ViT/S**|77.7|2.5|12.4|
|STViT (2023)|79.8|1.91|22.1|
|ToMe(2023)|79.4|2.7|22.1|
|ToFu(2023)|79.6|2.7|22.1|
|Zero-TP-b(2024)|79.1|2.5|22.1|
|LF-ViT(2024)|79.8|1.7|22.1|
|Q-ViT|79|0.4|8.7|

**Compatibility:** Our method is compatible with different Transformer-based models and consistently produces promising performance (Table above). Note that, our method provides a new paradigm for ViT acceleration, which is compatible to previous approaches (PTQ, token sparsification, etc) to achieve further efficiency gains. Due to time limits,results on Swin and CLIP will be included.

**Baseline:** As shown above, our method demonstrates effectiveness agaisnt SOTA methods (including token sparsification,pruning,merging,etc).
 
**Training cost:** Training time of TQ-ViT is ~20% longer than ViT with only ~5% additional memory footprint due to distillation strategy,while TQ-block has only very small overhead.

**Expression:** All typos will be revised.We will add a figure to better illustrate token-wise rep and TQ.

**Inconsistent parameters:**  ViT in Table 4 and 5 are respectively developed for ImageNet and CIFAR such that their parameter counts are slightly different.

**Downstream tasks:** Following previous evaluation protocol,we mainly evaluate our method on image classification tasks.Due to time limits, results on more tasks will be added.