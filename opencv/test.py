import cv2 as cv

# 图片路径（请确保正确）
src_path = r"D:\python_chuangxin\jiemianpythonProject\opencv\stained_images\cat_stained.png"  # 待修复图片路径
mask_path = r"D:\python_chuangxin\jiemianpythonProject\opencv\masks\cat_mask.png"    # 掩码图片路径

# 图片路径（请确保正确）
src_path = r"D:\python_chuangxin\jiemianpythonProject\opencv\stained_images\cat_stained.png"  # 待修复图片路径
mask_path = r"D:\python_chuangxin\jiemianpythonProject\opencv\masks\cat_mask.png"    # 掩码图片路径

# 读取图片
src = cv.imread(src_path)          # 读取待修复图片（彩色模式）
mask = cv.imread(mask_path, 0)     # 读取掩码图片（灰度模式）

# 核心检查：确保图片读取成功
if src is None:
    print(f"错误：无法读取待修复图片，请检查路径：{src_path}")
    exit()
if mask is None:
    print(f"错误：无法读取掩码图片，请检查路径：{mask_path}")
    exit()

# 核心处理：确保尺寸匹配
if src.shape[:2] != mask.shape[:2]:
    mask = cv.resize(mask, (src.shape[1], src.shape[0]))  # 调整掩码尺寸与原图一致

# 核心处理：确保掩码为有效二值图（0或255）
_, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)

# 核心功能：图像修复
dst = cv.inpaint(src, mask, 3, cv.INPAINT_TELEA)

# 保存结果到指定路径
output_path = r"D:\python_chuangxin\jiemianpythonProject\opencv\result_saved.png"
cv.imwrite(output_path, dst)
print(f"修复结果已保存至：{output_path}")

# 窗口显示功能（如需禁用，直接注释掉以下4行即可）
cv.imshow("待修复图", src)       # 显示原始待修复图像
cv.imshow("掩码图", mask)         # 显示掩码图像
cv.imshow("修复结果", dst)        # 显示修复后的图像
cv.waitKey(0)                     # 等待按键关闭窗口
cv.destroyAllWindows()            # 释放窗口资源


# 读取图片
src = cv.imread(src_path)
mask = cv.imread(mask_path, 0)

# 基础读取检查
if src is None:
    print(f"错误：无法读取待修复图片，请检查路径：{src_path}")
    exit()
if mask is None:
    print(f"错误：无法读取掩码图片，请检查路径：{mask_path}")
    exit()

# 尺寸匹配检查
if src.shape[:2] != mask.shape[:2]:
    print(f"错误：尺寸不匹配！待修复图 {src.shape[:2]}，掩码图 {mask.shape[:2]}")
    mask = cv.resize(mask, (src.shape[1], src.shape[0]))  # 强制调整尺寸
    print(f"已调整掩码尺寸为 {mask.shape[:2]}")

# 掩码有效性检查
if cv.countNonZero(mask) == 0:
    print("警告：掩码图无白色修复区域，添加测试区域")
    cv.rectangle(mask, (50, 50), (150, 150), 255, -1)  # 画白色矩形

# 确保掩码是二值图
_, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)

# 图像修复
dst = cv.inpaint(src, mask, 3, cv.INPAINT_TELEA)

# 保存结果（优先通过文件查看）
cv.imwrite("src_saved.png", src)
cv.imwrite("mask_saved.png", mask)
cv.imwrite("result_saved.png", dst)
print("已保存处理结果到代码目录")

# 显示窗口
cv.imshow("待修复图", src)
cv.imshow("掩码图", mask)
cv.imshow("修复结果", dst)
cv.waitKey(0)
cv.destroyAllWindows()