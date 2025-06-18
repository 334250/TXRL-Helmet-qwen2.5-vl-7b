import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

def draw_boxes_on_image(image_path, xml_path, output_path):
    try:
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        font = None
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            font = None
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name_elem = obj.find('name')
            if name_elem is None:
                continue
            name = name_elem.text
            bndbox_elem = obj.find('bndbox')
            if bndbox_elem is not None:
                xmin = int(bndbox_elem.find('xmin').text)
                ymin = int(bndbox_elem.find('ymin').text)
                xmax = int(bndbox_elem.find('xmax').text)
                ymax = int(bndbox_elem.find('ymax').text)
                if name in ['hat', 'helmet']:
                    color = (0, 255, 0)
                elif name == 'no-helmet':
                    color = (255, 0, 0)
                else:
                    color = (0, 0, 255)
                draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
                label = name
                if font:
                    draw.text((xmin, max(ymin-20, 0)), label, fill=color, font=font)
                else:
                    draw.text((xmin, max(ymin-20, 0)), label, fill=color)
        image.save(output_path)
        return True
    except Exception as e:
        print(f"Error processing {image_path} and {xml_path}: {e}")
        return False

def main():
    xml_dir = r'f:\TXRL\code\qwen2.5_VL_7b\input\dataset\Helmet\an2'
    img_dir = r'f:\TXRL\code\qwen2.5_VL_7b\input\dataset\Helmet\jpg2'
    output_dir = r'f:\TXRL\code\qwen2.5_VL_7b\output\dataset\vision'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith('.xml'):
            continue
        xml_path = os.path.join(xml_dir, xml_file)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            image_filename = root.find('filename').text
            # 优先使用<path>字段
            path_elem = root.find('path')
            if path_elem is not None and os.path.exists(path_elem.text):
                image_path = path_elem.text
            else:
                image_path = os.path.join(img_dir, image_filename)
                if not os.path.exists(image_path):
                    print(f"Image not found for {xml_file}: {image_filename}")
                    continue
            output_path = os.path.join(output_dir, image_filename)
            draw_boxes_on_image(image_path, xml_path, output_path)
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")

if __name__ == '__main__':
    main()