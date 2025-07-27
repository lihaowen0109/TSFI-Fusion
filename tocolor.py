import os
from PIL import Image

out_folder = r'./MSRS/'
gray_folder = r'./MSRS1/'
vi_folder = r'./MSRS/vi/'

os.makedirs(out_folder, exist_ok=True)

for gray_name in os.listdir(gray_folder):
    if gray_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        gray_base_name = os.path.splitext(gray_name)[0]

        for vi_name in os.listdir(vi_folder):
            if vi_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                vi_base_name = os.path.splitext(vi_name)[0]
                if gray_base_name == vi_base_name:
                    gray_img_path = os.path.join(gray_folder, gray_name)
                    gray_img = Image.open(gray_img_path).convert('L')

                    vi_img_path = os.path.join(vi_folder, vi_name)
                    vi_img = Image.open(vi_img_path).convert('YCbCr')
                    vi, cb, cr = vi_img.split()
                    combine_img = Image.merge(
                        "YCbCr", (gray_img, cb, cr)).convert('RGB')
                    output_path = os.path.join(
                        out_folder, f"{gray_base_name}.png")
                    combine_img.save(output_path)
                    print(f"Saved combined image to {output_path}")
                    break

print("Images have been saved.")
