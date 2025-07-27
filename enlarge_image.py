import os
import cv2
from PIL import Image, ImageDraw

class ImagePlotter:
    def __init__(self, image_path):
        self.image_path = image_path
        self.refPt = []
        self.cropping = False
        self.regions = []
        self.image = None
        self.clone = None

    def click_and_crop(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt = [(x, y)]
            self.cropping = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.refPt.append((x, y))
            self.cropping = False
            color = (255, 165, 0) if len(self.regions) < 2 else (255, 255, 0)
            cv2.rectangle(self.image, self.refPt[0], self.refPt[1], color, 2)
            cv2.imshow("Image", self.image)
            if len(self.refPt) == 2:
                self.regions.append((self.refPt[0][0], self.refPt[0][1], self.refPt[1][0], self.refPt[1][1]))

    def setup_image_selection(self):
        self.image = cv2.imread(self.image_path)
        self.clone = self.image.copy()
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.click_and_crop)
        while len(self.regions) < 2:
            cv2.imshow("Image", self.image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                self.image = self.clone.copy()
                self.regions.clear()
        cv2.destroyAllWindows()


class ImageCropper:
    def __init__(self, dir_path, regions, out_path):
        self.dir_path = dir_path
        self.regions = regions
        self.out_path = out_path

    def process_image(self, magnifications):
        file_ls = os.listdir(self.dir_path)
        file_ls = [i for i in file_ls if i.endswith(('.png', '.jpg', '.jpeg'))]
        for i in file_ls:
            image_path = os.path.join(self.dir_path, i)
            print(image_path)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            self.adjust_sizes_and_create_image(pil_image, pil_image.width, pil_image.height, magnifications,
                                               file_name=i)

    def adjust_sizes_and_create_image(self, pil_image, original_width, original_height, magnifications, file_name):
        new_sizes = self.calculate_new_sizes(magnifications)
        total_new_width = sum(size[0] for size in new_sizes)
        width_scale = original_width / total_new_width
        adjusted_sizes = [(int(width * width_scale), int(height * width_scale)) for (width, height) in new_sizes]
        new_height = original_height + max(size[1] for size in adjusted_sizes)
        new_image = Image.new("RGB", (original_width, new_height))
        new_image.paste(pil_image, (0, 0))
        self.paste_crops_and_draw_boxes(pil_image, new_image, adjusted_sizes, original_height)

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        new_image.save(os.path.join(self.out_path, file_name))

    def calculate_new_sizes(self, magnifications):
        # 确保两个区域大小相同
        base_size = max(magnifications)
        return [(int(base_size), int(base_size)) for _ in self.regions]

    def paste_crops_and_draw_boxes(self, pil_image, new_image, adjusted_sizes, original_height):
        draw = ImageDraw.Draw(new_image)
        colors = [(255, 0, 0), (0, 255, 0)]

        # 计算两个裁剪区域的总宽度
        total_width = sum(size[0] for size in adjusted_sizes)

        # 计算居中所需的起始 x 坐标
        center_x = (new_image.width - total_width) // 2

        for (region, size), color in zip(zip(self.regions, adjusted_sizes), colors):
            crop_img = pil_image.crop(region).resize(size, Image.LANCZOS)
            new_image.paste(crop_img, (center_x, original_height))
            draw.rectangle([center_x, original_height, center_x + size[0], original_height + size[1]], outline=color,
                           width=6)
            draw.rectangle([region[0], region[1], region[2], region[3]], outline=color, width=4)
            center_x += size[0]

def main(dir_path, save_dir='box'):
    # 选择RoI图像
    roi_image_path = os.path.join(dir_path, 'our.png')  # 假设 roi_image.png 是用于绘制 RoI 的图像
    plotter = ImagePlotter(roi_image_path)
    plotter.setup_image_selection()

    # 获取用户输入的放大倍数
    magnification1 = float(input("Enter magnification for the first region: "))
    magnification2 = float(input("Enter magnification for the second region: "))

    # 创建 ImageCropper 实例，处理目录中的所有图片
    cropper = ImageCropper(dir_path, plotter.regions, os.path.join(dir_path, save_dir))
    cropper.process_image([magnification1, magnification2])


if __name__ == "__main__":
    main(dir_path=r'./data_magni/MSRS_Night/',
         save_dir=r'./data_magni/result_msrs_night/')