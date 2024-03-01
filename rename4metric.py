import os
from PIL import Image

def rename_images(input_folder):
    if not os.path.exists(input_folder):
        print("does not exsist")
        return

    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    images.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

    for index, img_name in enumerate(images):
        extension = os.path.splitext(img_name)[1].lower()
        img_path = os.path.join(input_folder, img_name)
        img = Image.open(img_path)
        
        # 转换为PNG格式并重新命名
        new_name = f"{index:05d}.png"
        new_path = os.path.join(input_folder, new_name)
        img.save(new_path, 'PNG')
        
        # 删除旧文件
        os.remove(img_path)
        
        print(f"Converted {img_name} to {new_name}")

# 指定图片所在的文件夹路径
input_folder_path = 'output/banana_Loss_colmapMask/train/ours_30000/images'  # 替换为你的文件夹路径

rename_images(input_folder_path)

