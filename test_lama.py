import sys
import os

# 手动设置LAMA仓库的绝对路径（请务必替换为你实际的路径）
lama_folder = r"C:\Users\48690\lama"

# 验证路径是否存在
if not os.path.exists(lama_folder):
    print(f"错误：LAMA文件夹不存在 - {lama_folder}")
    sys.exit(1)

# 检查saicinpainting文件夹是否存在
saicinpainting_folder = os.path.join(lama_folder, "saicinpainting")
if not os.path.exists(saicinpainting_folder):
    print(f"错误：未找到saicinpainting文件夹 - {saicinpainting_folder}")
    print("请检查仓库是否完整克隆，或路径是否正确")
    sys.exit(1)

# 创建__init__.py文件（如果不存在）
init_file = os.path.join(saicinpainting_folder, "__init__.py")
if not os.path.exists(init_file):
    with open(init_file, "w", encoding="utf-8") as f:
        f.write("")  # 创建空的__init__.py文件
    print(f"已在saicinpainting文件夹中创建__init__.py文件")

# 添加路径
sys.path.append(lama_folder)
sys.path.append(saicinpainting_folder)

# 最后尝试导入
try:
    import saicinpainting

    print("成功导入saicinpainting模块！")
except ImportError as e:
    print(f"导入失败: {e}")
    print("\n当前Python搜索路径:")
    for p in sys.path:
        if "lama" in p or "saicinpainting" in p:
            print(f"- {p}")
    sys.exit(1)
