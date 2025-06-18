import os
import xml.etree.ElementTree as ET
import json


def convert_xml_to_json_with_annotations(xml_file_path, output_dir):
    """Converts a single XML annotation file to JSON, including bounding box info for people with/without helmets."""
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        image_filename = root.find('filename').text

        people_boxes = []
        helmet_boxes = []
        people_with_helmet = []
        people_without_helmet = []

        # 收集所有 helmet 和 people 的框
        for obj in root.findall('object'):
            name_elem = obj.find('name')
            if name_elem is None:
                continue
            name = name_elem.text
            bndbox_elem = obj.find('bndbox')
            if bndbox_elem is not None:
                bndbox = {
                    'xmin': int(bndbox_elem.find('xmin').text),
                    'ymin': int(bndbox_elem.find('ymin').text),
                    'xmax': int(bndbox_elem.find('xmax').text),
                    'ymax': int(bndbox_elem.find('ymax').text)
                }
                if name in ['helmet', 'hat']:
                    helmet_boxes.append(bndbox)
                elif name == 'people':
                    people_boxes.append(bndbox)

        # 判断每个人是否佩戴头盔
        def iou(boxA, boxB):
            xA = max(boxA['xmin'], boxB['xmin'])
            yA = max(boxA['ymin'], boxB['ymin'])
            xB = min(boxA['xmax'], boxB['xmax'])
            yB = min(boxA['ymax'], boxB['ymax'])
            interW = max(0, xB - xA)
            interH = max(0, yB - yA)
            interArea = interW * interH
            if interArea == 0:
                return 0.0
            boxAArea = (boxA['xmax'] - boxA['xmin']) * (boxA['ymax'] - boxA['ymin'])
            boxBArea = (boxB['xmax'] - boxB['xmin']) * (boxB['ymax'] - boxB['ymin'])
            iou_val = interArea / float(boxAArea + boxBArea - interArea)
            return iou_val

        # 判断每个人是否佩戴头盔
        def helmet_in_person(person_box, helmet_box):
            # 判断头盔中心点是否在人员框内
            cx = (helmet_box['xmin'] + helmet_box['xmax']) // 2
            cy = (helmet_box['ymin'] + helmet_box['ymax']) // 2
            return (person_box['xmin'] <= cx <= person_box['xmax']) and (person_box['ymin'] <= cy <= person_box['ymax'])

        for pbox in people_boxes:
            matched = False
            for hbox in helmet_boxes:
                # 只要头盔中心点在人员框内即可判定为佩戴
                if helmet_in_person(pbox, hbox):
                    matched = True
                    break
            if matched:
                people_with_helmet.append(pbox)
            else:
                people_without_helmet.append(pbox)

        hat_count = len(people_with_helmet)
        no_helmet_count = len(people_without_helmet)

        assistant_response_value = f"有{hat_count}人佩戴了安全头盔，有{no_helmet_count}人没有佩戴安全头盔。"
        if people_with_helmet:
            bnd_box_strings = []
            for box in people_with_helmet:
                bnd_box_strings.append(
                    f"[xmin={box['xmin']}, ymin={box['ymin']}, xmax={box['xmax']}, ymax={box['ymax']}]")
            assistant_response_value += " 佩戴安全头盔的人员位置：" + ", ".join(bnd_box_strings) + "。"
        if people_without_helmet:
            bnd_box_strings = []
            for box in people_without_helmet:
                bnd_box_strings.append(
                    f"[xmin={box['xmin']}, ymin={box['ymin']}, xmax={box['xmax']}, ymax={box['ymax']}]")
            assistant_response_value += " 未佩戴安全头盔的人员位置：" + ", ".join(bnd_box_strings) + "。"

        output_data = {
            "image": image_filename,
            "conversations": [
                {
                    "from": "user",
                    "value": "<img>这张图中有几个人佩戴了安全头盔，有几个人没有佩戴安全头盔</img>"
                },
                {
                    "from": "assistant",
                    "value": assistant_response_value
                }
            ]
        }

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        base_xml_filename = os.path.basename(xml_file_path)
        json_filename = os.path.splitext(base_xml_filename)[0] + '.json'
        output_json_path = os.path.join(output_dir, json_filename)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        return True
    except Exception as e:
        print(f"Error processing {xml_file_path}: {e}")
        return False


def main():
    input_xml_dir = r'f:\TXRL\code\qwen2.5_VL_7b\input\dataset\Helmet\an2'
    output_json_dir = r'f:\TXRL\code\qwen2.5_VL_7b\output\dataset\json'

    if not os.path.isdir(input_xml_dir):
        print(f"Input directory '{input_xml_dir}' does not exist or is not a directory.")
        return

    if not os.path.exists(output_json_dir):
        os.makedirs(output_json_dir)
        print(f"Created output directory: {output_json_dir}")

    processed_count = 0
    error_count = 0
    xml_files = [f for f in os.listdir(input_xml_dir) if f.endswith('.xml')]

    if not xml_files:
        print(f"No XML files found in '{input_xml_dir}'.")
        return

    print(f"Found {len(xml_files)} XML files to process in '{input_xml_dir}'.")

    for filename in xml_files:
        xml_file_path = os.path.join(input_xml_dir, filename)
        if convert_xml_to_json_with_annotations(xml_file_path, output_json_dir):
            processed_count += 1
        else:
            error_count += 1

    print(f"\nProcessing complete.")
    print(f"Successfully converted {processed_count} files.")
    if error_count > 0:
        print(f"Failed to convert {error_count} files.")


if __name__ == '__main__':
    main()