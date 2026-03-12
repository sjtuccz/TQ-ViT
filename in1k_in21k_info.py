def calculate_dataset_ratios():
    """计算数据集比例"""
    
    # 官方数据统计
    dataset_stats = {
        'ImageNet-1K (IN-1K)': {
            '训练图片数': 1_281_167,
            '验证图片数': 50_000,
            '总图片数': 1_331_167,
            '类别数': 1_000,
            '每类平均图片数': 1_281
        },
        'ImageNet-21K (IN-21K)': {
            '训练图片数': 14_197_122,  # 约14.2百万
            '验证图片数': 52_460,     # 约5.2万
            '总图片数': 14_249_582,  # 约14.25百万
            '类别数': 21_843,
            '每类平均图片数': 650
        }
    }
    
    # 计算比例
    ratios = {}
    
    for metric in ['训练图片数', '总图片数', '类别数']:
        in21k_val = dataset_stats['ImageNet-21K (IN-21K)'][metric]
        in1k_val = dataset_stats['ImageNet-1K (IN-1K)'][metric]
        ratio = in21k_val / in1k_val
        
        ratios[metric] = {
            'IN-21K': in21k_val,
            'IN-1K': in1k_val,
            '倍数': ratio,
            '百分比': f"{ratio:.1f}倍"
        }
    
    return dataset_stats, ratios

# 计算比例
dataset_stats, ratios = calculate_dataset_ratios()

print("ImageNet数据集详细对比:")
print("=" * 80)
for dataset, stats in dataset_stats.items():
    print(f"\n{dataset}:")
    for key, value in stats.items():
        if '图片数' in key:
            print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value}")

print(f"\n{'='*80}")
print("数据量倍数对比:")
print(f"{'指标':<15} | {'IN-21K':>15} | {'IN-1K':>15} | {'倍数':>10} | {'比例':>10}")
print("-" * 80)
for metric, data in ratios.items():
    print(f"{metric:<15} | {data['IN-21K']:>15,} | {data['IN-1K']:>15,} | {data['倍数']:>9.1f}x | {data['百分比']:>10}")