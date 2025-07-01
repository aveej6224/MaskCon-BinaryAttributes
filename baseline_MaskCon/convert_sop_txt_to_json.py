import os
import json

def convert_txt_to_json(txt_file, save_dir, is_train=True):
    image_paths = []
    coarse_labels = []
    fine_labels = []

    with open(txt_file, 'r') as f:
        lines = f.readlines()[1:]  # skip header

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        class_id = int(parts[1])        # Fine label
        super_class_id = int(parts[2])  # Coarse label
        path = parts[3]

        image_paths.append(path)
        coarse_labels.append(super_class_id - 1)  # 0-based
        fine_labels.append(class_id - 1)          # 0-based

    split = "train" if is_train else "test"
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, f"{split}_path.json"), 'w') as f:
        json.dump(image_paths, f)

    with open(os.path.join(save_dir, f"{split}_coarse_label.json"), 'w') as f:
        json.dump(coarse_labels, f)

    with open(os.path.join(save_dir, f"{split}_fine_label.json"), 'w') as f:
        json.dump(fine_labels, f)

    print(f"[✓] Saved {split} JSONs to: {save_dir}")


if __name__ == "__main__":
    root = "./datasets/Stanford_Online_Products"
    out_dir = os.path.join(root, "sop_split1")

    convert_txt_to_json(os.path.join(root, "Ebay_train.txt"), out_dir, is_train=True)
    convert_txt_to_json(os.path.join(root, "Ebay_test.txt"), out_dir, is_train=False)
