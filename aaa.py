import torch.nn as nn
import time
import numpy as np
from collections import OrderedDict
import torch
class LightweightProfiler:
    """轻量级性能分析器"""
    
    def __init__(self, model):
        self.model = model
        self.timings = OrderedDict()
        self.handles = []
        
    def _get_time(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()
    
    def _add_hooks(self):
        """添加前向hook"""
        def make_hook(name):
            def forward_pre_hook(module, input):
                self.start_times[name] = self._get_time()
            
            def forward_hook(module, input, output):
                end_time = self._get_time()
                if name in self.start_times:
                    elapsed = (end_time - self.start_times[name]) * 1000
                    if name not in self.timings:
                        self.timings[name] = []
                    self.timings[name].append(elapsed)
            
            return forward_pre_hook, forward_hook
        
        self.start_times = {}
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # 叶子节点
                pre_hook, hook = make_hook(name)
                pre_handle = module.register_forward_pre_hook(pre_hook)
                handle = module.register_forward_hook(hook)
                self.handles.extend([pre_handle, handle])
    
    def profile(self, input_tensor, num_iterations=100, warmup=10):
        """性能分析"""
        # 添加hook
        self._add_hooks()
        
        # 预热
        self.model.eval()
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(input_tensor)
        
        # 清空计时
        self.timings.clear()
        
        # 正式测试
        with torch.no_grad():
            for i in range(num_iterations):
                _ = self.model(input_tensor)
        
        # 移除hook
        self._remove_hooks()
        
        # 分析结果
        return self.analyze()
    
    def _remove_hooks(self):
        """移除所有hook"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
    
    def analyze(self):
        """分析结果"""
        results = {}
        total_time = 0
        
        for name, times in self.timings.items():
            if times:
                mean_time = np.mean(times)
                std_time = np.std(times)
                results[name] = {
                    'mean_ms': mean_time,
                    'std_ms': std_time,
                    'calls': len(times)
                }
                total_time += mean_time
        
        # 排序
        sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_ms'], reverse=True)
        
        # 打印
        print(f"\n{'Layer':<40} {'Mean (ms)':<12} {'Std (ms)':<10} {'%':<8} {'Calls':<6}")
        print("-" * 80)
        
        for name, stats in sorted_results:
            percentage = (stats['mean_ms'] / total_time * 100) if total_time > 0 else 0
            print(f"{name:<40} {stats['mean_ms']:<12.3f} {stats['std_ms']:<10.3f} "
                  f"{percentage:<8.1f} {stats['calls']:<6}")
        
        print("-" * 80)
        print(f"{'TOTAL':<40} {total_time:<12.3f} ms")
        
        return results

# 使用示例
model = nn.Sequential(
    nn.Conv2d(3, 64, 3, 1, 1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, 1, 1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(128 * 8 * 8, 10)
).cuda()

profiler = LightweightProfiler(model)
input_tensor = torch.randn(32, 3, 32, 32).cuda()

results = profiler.profile(input_tensor, num_iterations=100, warmup=10)