from PIL import Image
import os

def resize_images(input_folder, output_folder, width, height):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')): 
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            resized_img = img.resize((width, height), Image.ANTIALIAS)  
            output_path = os.path.join(output_folder, filename)
            resized_img.save(output_path)  

# 指定输入文件夹、输出文件夹以及目标宽度和高度
input_folder_path = 'Dataset/banana/input_fake_og'  
output_folder_path = 'Dataset/banana/input_fake'  
target_width = 960 
target_height = 720  

resize_images(input_folder_path, output_folder_path, target_width, target_height)