import cv2
import numpy as np
import os  # 导入os库，用于处理文件夹和文件名

# ---------------------- 路径配置（无需手动输入，直接定义） ----------------------
# 1. 待检测污痕的图片路径（具体到图片文件，包含文件名和扩展名）
image_path = r"D:\python_chuangxin\jiemianpythonProject\test_pictue_done\cat_damaged.png"
# 2. mask图的保存文件夹（仅文件夹路径，无需写文件名）
mask_save_dir = r"D:\python_chuangxin\jiemianpythonProject\test_pictue_done"
# -----------------------------------------------------------------------------

# 自动生成mask图的完整保存路径（原图片名 + _mask.png，避免覆盖原文件）
# 例如：cat_damaged.png → cat_damaged_mask.png
filename = os.path.basename(image_path)  # 提取原图片文件名（如cat_damaged.png）
name_without_ext = os.path.splitext(filename)[0]  # 提取文件名（不含扩展名，如cat_damaged）
mask_save_path = os.path.join(mask_save_dir, f"{name_without_ext}_mask.png")  # 拼接完整保存路径

# ---------------------- 核心：污痕检测与mask生成 ----------------------
# 1. 读取图片
img = cv2.imread(image_path)
# 检查图片是否读取成功
if img is None:
    print(f"❌ 无法读取图片，请检查路径是否正确：{image_path}")
    exit()  # 读取失败则退出程序，避免后续报错

# 2. 图像预处理（去噪+灰度化，提升污痕检测准确性）
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 彩色图转灰度图
blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯模糊：减少图像噪声，避免误判污痕

# 3. 自适应阈值分割（核心步骤：识别污痕区域，白色为污痕，黑色为背景）
# 参数说明：11=邻域大小，2=阈值偏移量（污痕浅可减小为1，污痕深可增大为3-5）
thresh = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
)

# 4. 形态学优化（去除小噪声点，让污痕区域更连贯）
kernel = np.ones((3, 3), np.uint8)  # 3x3核心（噪声多可改为5x5，污痕细保持3x3）
mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # 闭运算：填充污痕区域的小空洞
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)     # 开运算：去除背景中的小亮点噪声

# ---------------------- 保存与结果提示 ----------------------
# 保存生成的mask图
cv2.imwrite(mask_save_path, mask)
# 打印处理结果
print(f"✅ 处理完成！")
print(f"📷 原图路径：{image_path}")
print(f"🎭 Mask图保存路径：{mask_save_path}")

# ---------------------- 窗口显示（可选，不需要可直接注释掉以下4行） ----------------------
cv2.imshow("Original Image (待检测图)", img)  # 显示原图
cv2.imshow("Stain Mask (污痕掩码图)", mask)    # 显示生成的mask图
cv2.waitKey(0)  # 等待任意按键（0表示无限等待，按键盘任意键关闭窗口）
cv2.destroyAllWindows()  # 关闭所有OpenCV窗口，释放资源