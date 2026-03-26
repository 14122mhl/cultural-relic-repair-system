#!/usr/bin/env python3
import logging
import os
import sys
import traceback
import cv2
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
import torch.nn.functional as F

# 设置环境变量以控制线程数量
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# 确保LAMA库可以被导入
try:
    # 尝试导入LAMA相关模块
    from saicinpainting.evaluation.utils import move_to_device
    from saicinpainting.evaluation.refinement import refine_predict
    from saicinpainting.training.data.datasets import make_default_val_dataset
    from saicinpainting.training.trainers import load_checkpoint
    from saicinpainting.utils import register_debug_signal_handlers
except ImportError as e:
    # 提示用户可能需要调整LAMA库路径
    print(f"导入LAMA库失败: {e}")
    print("请确保LAMA库路径已正确设置")
    sys.exit(1)

# 配置日志
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class FastRestorationModel:
    """快速修复算法模型封装类，基于LAMA实现"""

    def __init__(self, model_path, device='cpu', refine=False):
        """
        初始化快速修复模型

        参数:
            model_path: LAMA模型目录路径（直接包含config.yaml和best.ckpt）
            device: 运行设备 ('cpu' 或 'cuda')
            refine: 是否使用结果优化
        """
        self.model_path = model_path
        self.device = torch.device(device)
        self.refine = refine
        self.model = None
        self.train_config = None
        # 模型通常要求尺寸为32的倍数，这是根据LAMA模型特性设置的
        self.size_divisor = 32

        # 加载模型
        self._load_model()

    def _load_model(self):
        """加载模型和配置"""
        try:
            # 注册调试信号处理器（非Windows系统）
            if sys.platform != 'win32':
                register_debug_signal_handlers()

            # 加载训练配置
            train_config_path = os.path.join(self.model_path, 'config.yaml')
            if not os.path.exists(train_config_path):
                raise FileNotFoundError(f"配置文件不存在: {train_config_path}")

            with open(train_config_path, 'r') as f:
                self.train_config = OmegaConf.create(yaml.safe_load(f))

            # 设置模型为仅预测模式
            self.train_config.training_model.predict_only = True
            self.train_config.visualizer.kind = 'noop'

            # 加载模型检查点
            checkpoint_path = os.path.join(self.model_path, 'best.ckpt')
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"模型检查点不存在: {checkpoint_path}")

            # 加载模型
            self.model = load_checkpoint(
                self.train_config,
                checkpoint_path,
                strict=False,
                map_location='cpu'
            )
            self.model.freeze()

            # 如果不使用优化，将模型移动到指定设备
            if not self.refine:
                self.model.to(self.device)

            LOGGER.info(f"模型已成功加载: {checkpoint_path}")

        except Exception as ex:
            LOGGER.error(f"模型加载失败: {ex}\n{traceback.format_exc()}")
            raise

    def _resize_to_valid_size(self, image, mask):
        """
        将图像和掩码调整为模型要求的尺寸（size_divisor的倍数）

        参数:
            image: 输入图像张量 (C, H, W)
            mask: 掩码张量 (1, H, W)

        返回:
            调整后的图像和掩码张量
        """
        # 获取当前尺寸
        c, h, w = image.shape
        _, mask_h, mask_w = mask.shape

        # 检查图像和掩码尺寸是否匹配
        if h != mask_h or w != mask_w:
            # 如果不匹配，将掩码调整为与图像相同尺寸
            mask = F.interpolate(
                mask.unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # 计算调整后的尺寸（向上取整到最近的size_divisor倍数）
        new_h = ((h + self.size_divisor - 1) // self.size_divisor) * self.size_divisor
        new_w = ((w + self.size_divisor - 1) // self.size_divisor) * self.size_divisor

        # 如果需要调整尺寸
        if new_h != h or new_w != w:
            # 调整图像尺寸
            image = F.interpolate(
                image.unsqueeze(0),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            # 调整掩码尺寸
            mask = F.interpolate(
                mask.unsqueeze(0),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        return image, mask

    def predict(self, image_path, mask_path, output_path=None, out_ext='.png'):
        """
        对单张图像进行快速修复

        参数:
            image_path: 输入图像路径
            mask_path: 掩码图像路径
            output_path: 输出图像保存路径，为None则不保存
            out_ext: 输出图像扩展名

        返回:
            修复后的图像（BGR格式的numpy数组）
        """
        try:
            # 验证输入文件是否存在
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"输入图像不存在: {image_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"掩码图像不存在: {mask_path}")

            # 创建临时数据集来处理单张图像
            class SingleImageDataset:
                def __init__(self, image_path, mask_path):
                    self.image_path = image_path
                    self.mask_path = mask_path
                    self.mask_filenames = [mask_path]

                def __len__(self):
                    return 1

                def __getitem__(self, idx):
                    # 读取图像和掩码
                    image = cv2.imread(self.image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

                    mask = cv2.imread(self.mask_path, 0)  # 读取为灰度图
                    mask = (mask > 0) * 1.0  # 二值化

                    # 添加批次维度
                    return {
                        'image': torch.from_numpy(image.transpose(2, 0, 1)).float(),
                        'mask': torch.from_numpy(mask).float().unsqueeze(0),
                        'unpad_to_size': image.shape[:2]  # 保存原始尺寸 (H, W)
                    }

            # 创建数据集和数据批次
            dataset = SingleImageDataset(image_path, mask_path)
            batch = default_collate([dataset[0]])

            # 关键修复：调整图像和掩码尺寸以匹配模型要求
            # 提取图像和掩码
            image = batch['image'][0]  # 取第一个样本
            mask = batch['mask'][0]  # 取第一个样本

            # 调整尺寸
            resized_image, resized_mask = self._resize_to_valid_size(image, mask)

            # 更新批次中的图像和掩码
            batch['image'] = resized_image.unsqueeze(0)
            batch['mask'] = resized_mask.unsqueeze(0)

            # 执行修复
            if self.refine:
                # 使用优化模式
                cur_res = refine_predict(batch, self.model, **{'refiner': {}})
                cur_res = cur_res[0].permute(1, 2, 0).detach().cpu().numpy()
            else:
                # 标准模式
                with torch.no_grad():
                    batch = move_to_device(batch, self.device)
                    batch['mask'] = (batch['mask'] > 0) * 1
                    batch = self.model(batch)

                    # 检查并调整预测结果尺寸与输入匹配
                    if batch['predicted_image'].shape != batch['image'].shape:
                        batch['predicted_image'] = F.interpolate(
                            batch['predicted_image'],
                            size=batch['image'].shape[2:],
                            mode='bilinear',
                            align_corners=False
                        )

                    cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()

                    # 调整大小到原始尺寸 - 修复：将张量转换为整数
                    unpad_to_size = batch.get('unpad_to_size', None)
                    if unpad_to_size is not None:
                        # 将张量转换为CPU上的numpy数组，再转为整数
                        orig_height = int(unpad_to_size[0].cpu().numpy())
                        orig_width = int(unpad_to_size[1].cpu().numpy())
                        cur_res = cv2.resize(cur_res, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)

            # 处理结果
            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)  # 转换回BGR格式

            # 保存结果（如果指定了输出路径）
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, cur_res)
                LOGGER.info(f"修复结果已保存至: {output_path}")

            return cur_res

        except Exception as ex:
            LOGGER.error(f"修复过程失败: {ex}\n{traceback.format_exc()}")
            raise

    def batch_predict(self, input_dir, output_dir, out_ext='.png'):
        """
        批量处理图像

        参数:
            input_dir: 输入目录，包含图像和掩码
            output_dir: 输出目录
            out_ext: 输出图像扩展名
        """
        try:
            if not os.path.isdir(input_dir):
                raise NotADirectoryError(f"输入目录不存在: {input_dir}")

            os.makedirs(output_dir, exist_ok=True)

            # 创建数据集
            dataset = make_default_val_dataset(input_dir, **{'img_suffix': '.png'})

            # 批量处理
            for img_i in range(len(dataset)):
                mask_fname = dataset.mask_filenames[img_i]
                # 构建输出文件名
                cur_out_fname = os.path.join(
                    output_dir,
                    os.path.splitext(mask_fname[len(input_dir):])[0] + out_ext
                )
                os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)

                # 处理单张图像
                batch = default_collate([dataset[img_i]])

                # 关键修复：调整图像和掩码尺寸以匹配模型要求
                image = batch['image'][0]
                mask = batch['mask'][0]
                resized_image, resized_mask = self._resize_to_valid_size(image, mask)
                batch['image'] = resized_image.unsqueeze(0)
                batch['mask'] = resized_mask.unsqueeze(0)

                if self.refine:
                    # 使用优化模式
                    cur_res = refine_predict(batch, self.model, **{'refiner': {}})
                    cur_res = cur_res[0].permute(1, 2, 0).detach().cpu().numpy()
                else:
                    # 标准模式
                    with torch.no_grad():
                        batch = move_to_device(batch, self.device)
                        batch['mask'] = (batch['mask'] > 0) * 1
                        batch = self.model(batch)

                        # 检查并调整预测结果尺寸与输入匹配
                        if batch['predicted_image'].shape != batch['image'].shape:
                            batch['predicted_image'] = F.interpolate(
                                batch['predicted_image'],
                                size=batch['image'].shape[2:],
                                mode='bilinear',
                                align_corners=False
                            )

                        cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()

                        # 调整大小到原始尺寸 - 修复：批量处理同样转换为整数
                        unpad_to_size = batch.get('unpad_to_size', None)
                        if unpad_to_size is not None:
                            orig_height = int(unpad_to_size[0][0].cpu().numpy())
                            orig_width = int(unpad_to_size[0][1].cpu().numpy())
                            cur_res = cv2.resize(cur_res, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)

                # 处理结果
                cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
                cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
                cv2.imwrite(cur_out_fname, cur_res)

                if (img_i + 1) % 10 == 0:
                    LOGGER.info(f"已处理 {img_i + 1}/{len(dataset)} 张图像")

            LOGGER.info(f"批量处理完成，结果保存至: {output_dir}")

        except Exception as ex:
            LOGGER.error(f"批量处理失败: {ex}\n{traceback.format_exc()}")
            raise


# 测试代码
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='快速图像修复工具')
    parser.add_argument('--model-path', required=True, help='LAMA模型目录路径（包含config.yaml和best.ckpt）')
    parser.add_argument('--image', help='输入图像路径（单张处理）')
    parser.add_argument('--mask', help='掩码图像路径（单张处理）')
    parser.add_argument('--output', help='输出图像路径（单张处理）')
    parser.add_argument('--indir', help='输入目录（批量处理）')
    parser.add_argument('--outdir', help='输出目录（批量处理）')
    parser.add_argument('--device', default='cpu', help='运行设备（cpu或cuda）')
    parser.add_argument('--refine', action='store_true', help='是否使用结果优化')

    args = parser.parse_args()

    try:
        # 初始化模型
        model = FastRestorationModel(
            model_path=args.model_path,
            device=args.device,
            refine=args.refine
        )

        # 单张处理
        if args.image and args.mask:
            output_path = args.output if args.output else 'restored.png'
            result = model.predict(
                image_path=args.image,
                mask_path=args.mask,
                output_path=output_path
            )
            print(f"单张图像修复完成，结果保存至: {output_path}")

        # 批量处理
        elif args.indir and args.outdir:
            model.batch_predict(
                input_dir=args.indir,
                output_dir=args.outdir
            )
            print(f"批量图像修复完成，结果保存至: {args.outdir}")

        else:
            print("请提供正确的参数组合：")
            print("单张处理: --image <图像路径> --mask <掩码路径> [--output <输出路径>]")
            print("批量处理: --indir <输入目录> --outdir <输出目录>")
            sys.exit(1)

    except Exception as e:
        print(f"处理失败: {e}")
        sys.exit(1)