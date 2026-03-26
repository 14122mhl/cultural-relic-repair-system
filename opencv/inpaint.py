import cv2 as cv
import os


def inpaint_image(src_path, mask_path, output_path=None, inpaint_radius=3, method=cv.INPAINT_TELEA):
    """
    使用OpenCV的inpaint算法修复图像

    参数:
        src_path: 原始图像路径
        mask_path: 掩码图像路径（白色区域表示需要修复的区域）
        output_path: 修复结果保存路径，为None时不保存
        inpaint_radius: 修复半径
        method: 修复方法，cv.INPAINT_TELEA或cv.INPAINT_NS

    返回:
        修复后的图像（BGR格式），如果出错则返回None
    """
    try:
        # 读取原始图像和掩码
        src = cv.imread(src_path)
        mask = cv.imread(mask_path, 0)  # 以灰度模式读取

        # 检查图像是否读取成功
        if src is None:
            raise ValueError(f"无法读取待修复图片，请检查路径：{src_path}")
        if mask is None:
            raise ValueError(f"无法读取掩码图片，请检查路径：{mask_path}")

        # 确保掩码与原始图像尺寸一致
        if src.shape[:2] != mask.shape[:2]:
            mask = cv.resize(mask, (src.shape[1], src.shape[0]))

        # 二值化掩码（确保掩码只有0和255两个值）
        _, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)

        # 执行图像修复
        dst = cv.inpaint(src, mask, inpaint_radius, method)

        # 保存结果（如果指定了输出路径）
        if output_path:
            # 创建输出目录（如果不存在）
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            cv.imwrite(output_path, dst)

        return dst

    except Exception as e:
        print(f"图像修复出错: {str(e)}")
        return None


# 测试代码
if __name__ == "__main__":
    # 测试用例
    src_path = r"D:\python_chuangxin\jiemianpythonProject\test_pictue_done\cat_damaged.png"
    mask_path = r"D:\python_chuangxin\jiemianpythonProject\test_pictue_done\cat_damaged_mask.png"
    output_path = r"D:\python_chuangxin\jiemianpythonProject\test_pictue_done\result_saved.png"

    result = inpaint_image(src_path, mask_path, output_path)

    if result is not None:
        print(f"修复结果已保存至：{output_path}")
        cv.imshow("待修复图", cv.imread(src_path))
        cv.imshow("掩码图", cv.imread(mask_path, 0))
        cv.imshow("修复结果", result)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print("修复失败")
