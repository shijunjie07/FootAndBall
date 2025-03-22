import os
import shutil
import configparser

from tqdm import tqdm
from sklearn.model_selection import train_test_split

class SoccerNetTrackingConverter:
    CLASS_NAME_TO_ID = {
        "ball": 0,
        "person": 1
    }

    def __init__(self, input_root, output_root, seed=42):
        self.input_root = input_root
        self.output_root = output_root
        self.seed = seed

    def canonicalize_object_type(self, obj_type):
        obj_type = obj_type.lower()
        if obj_type in ['goalkeeper', 'goalkeepers', 'player', 'referee', 'other']:
            return 'person'
        elif obj_type == 'ball':
            return 'ball'
        else:
            return 'person'

    def parse_gameinfo(self, gameinfo_path):
        config = configparser.ConfigParser(strict=False)
        config.read(gameinfo_path)
        tracklet_map = {}

        for key, value in config["Sequence"].items():
            if key.startswith("trackletid_"):
                parts = value.split(";")
                obj_type = parts[0].strip().split()[0].lower()
                canonical_type = self.canonicalize_object_type(obj_type)
                class_id = self.CLASS_NAME_TO_ID[canonical_type]
                tracklet_id = int(key.split("_")[1])
                tracklet_map[tracklet_id] = class_id

        return tracklet_map

    def convert_gt(self, gt_path, output_dir, sequence, tracklet_map):
        os.makedirs(output_dir, exist_ok=True)
        with open(gt_path, "r") as f:
            for line in f:
                parts = list(map(float, line.strip().split(",")))
                frame_id, tracklet_id = map(int, parts[:2])
                l, t, w, h = parts[2:6]

                if tracklet_id not in tracklet_map:
                    continue

                class_id = tracklet_map[tracklet_id]
                x1, y1, x2, y2 = l, t, l + w, t + h
                label_path = os.path.join(output_dir, f"{sequence}_{frame_id:06d}.txt")
                with open(label_path, "a") as label_file:
                    label_file.write(f"{class_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f}\n")

    def copy_images(self, image_dir, sequence, output_image_dir):
        os.makedirs(output_image_dir, exist_ok=True)
        for filename in sorted(os.listdir(image_dir)):
            if filename.endswith(".jpg"):
                new_filename = f'{sequence}_{filename}'
                shutil.copy(os.path.join(image_dir, filename), os.path.join(output_image_dir, new_filename))

    def run(self, splits=['train', 'test', 'valid']):
        for dset in splits:
            if dset != 'train':
                all_sequences = os.listdir(os.path.join(self.input_root, 'test'))
                test_sequences, val_sequences = train_test_split(all_sequences, test_size=0.2, shuffle=True, random_state=self.seed)
                dset_sequences = test_sequences if dset == 'test' else val_sequences
                dataset_base_dir = os.path.join(self.input_root, 'test')
            else:
                dset_sequences = os.listdir(os.path.join(self.input_root, 'train'))
                dataset_base_dir = os.path.join(self.input_root, 'train')

            output_images_dir = os.path.join(self.output_root, dset, 'images')
            output_labels_dir = os.path.join(self.output_root, dset, 'labels')
            dset_sequences.sort()

            print(f"Processing split: {dset} ({len(dset_sequences)} sequences)")
            for seq in tqdm(dset_sequences, desc=dset):
                dataset_dir = os.path.join(dataset_base_dir, seq)
                gt_path = os.path.join(dataset_dir, "gt", "gt.txt")
                image_dir = os.path.join(dataset_dir, "img1")
                gameinfo_path = os.path.join(dataset_dir, "gameinfo.ini")

                tracklet_map = self.parse_gameinfo(gameinfo_path)

                seqinfo = configparser.ConfigParser(strict=False)
                seqinfo.read(os.path.join(dataset_dir, "seqinfo.ini"))
                # img_width = int(seqinfo["Sequence"]["imWidth"])
                # img_height = int(seqinfo["Sequence"]["imHeight"])

                self.convert_gt(gt_path, output_labels_dir, seq, tracklet_map)
                self.copy_images(image_dir, seq, output_images_dir)

            print(f"Finished converting {dset} set. Output: {os.path.join(self.output_root, dset)}")


if __name__ == "__main__":
    converter = SoccerNetTrackingConverter(
        input_root="F:/soccer_tracking/sports_datasets/tracking-2023",
        output_root="F:/soccer_tracking/sports_datasets/sntracking"
    )
    converter.run(splits=['train', 'test', 'valid'])
