# 创建一个测试脚本 test_mmcv.py
import sys
import mmcv

print("Python 路径:")
for path in sys.path:
    print(f"  {path}")

print(f"\n导入的 mmcv 模块位置: {mmcv.__file__}")
print(f"mmcv 模块属性列表:")
for attr in dir(mmcv):
    if 'version' in attr.lower() or 'VERSION' in attr:
        print(f"  {attr}: {getattr(mmcv, attr, 'N/A')}")

# 尝试所有可能的版本属性
possible_attrs = ['__version__', 'VERSION', 'version', 'mmcv_version']
for attr in possible_attrs:
    if hasattr(mmcv, attr):
        print(f"✅ 找到 {attr}: {getattr(mmcv, attr)}")