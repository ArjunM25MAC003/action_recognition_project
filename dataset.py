# ============================================================
# FILE: dataset.py
# PURPOSE: Load videos and convert them into model-ready tensors
# ============================================================

import os
from collections import Counter

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from extract_frames import load_video


# Final labels required by the project
FINAL_LABELS = [
    "normal_activity",
    "fight",
    "fall",
    "crowd_anomaly",
    "running_panic",
]


# Mapping public dataset classes → our target labels
# NOTE: These are approximations (important to explain in interview)
CLASS_MAPPING = {
    # Fight-like motion
    "BoxingPunchingBag": "fight",
    "BoxingSpeedBag": "fight",
    "SumoWrestling": "fight",
    "Fencing": "fight",

    # Normal activities
    "WalkingWithDog": "normal_activity",
    "Walking": "normal_activity",
    "BabyCrawling": "normal_activity",
    "BrushingTeeth": "normal_activity",

    # Running / panic
    "Running": "running_panic",
    "Biking": "running_panic",
    "HorseRace": "running_panic",

    # Crowd-like activity (approximation)
    "JumpingJack": "crowd_anomaly",
    "BandMarching": "crowd_anomaly",
    "MilitaryParade": "crowd_anomaly",
}


class VideoDataset(Dataset):
    def __init__(self, data_folder, train=True):
        """
        data_folder:
            train → data/train/
            test  → data/test/
        """
        self.data_folder = data_folder
        self.train = train

        self.samples = []

        # label → index mapping
        self.label_to_index = {
            label: i for i, label in enumerate(FINAL_LABELS)
        }

        self.index_to_label = {
            i: label for label, i in self.label_to_index.items()
        }

        # Normalize images like ImageNet (important for MobileNet)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        if self.train:
            self.load_training_data()
        else:
            self.load_test_data()

    # -------------------------------
    # Helper functions
    # -------------------------------

    def is_video(self, file_name):
        return file_name.lower().endswith((".mp4", ".avi", ".mov"))

    def map_class_name(self, folder_name):
        """
        Convert dataset class → final label
        """
        # Case 1: already correct label (structured dataset)
        if folder_name in FINAL_LABELS:
            return folder_name

        # Case 2: map from UCF101
        return CLASS_MAPPING.get(folder_name)

    # -------------------------------
    # Load data
    # -------------------------------

    def load_training_data(self):
        if not os.path.exists(self.data_folder):
            print("Training folder not found!")
            return

        for folder_name in sorted(os.listdir(self.data_folder)):
            folder_path = os.path.join(self.data_folder, folder_name)

            if not os.path.isdir(folder_path):
                continue

            final_label = self.map_class_name(folder_name)

            if final_label is None:
                print(f"Skipping class: {folder_name}")
                continue

            label_index = self.label_to_index[final_label]

            for file_name in os.listdir(folder_path):
                if self.is_video(file_name):
                    video_path = os.path.join(folder_path, file_name)
                    self.samples.append((video_path, label_index))

        print("\nDataset loaded!")
        self.print_class_distribution()

    def load_test_data(self):
        for file_name in sorted(os.listdir(self.data_folder)):
            if self.is_video(file_name):
                video_path = os.path.join(self.data_folder, file_name)
                video_id = os.path.splitext(file_name)[0]
                self.samples.append((video_path, video_id))

        print(f"Loaded {len(self.samples)} test videos")

    # -------------------------------
    # Info
    # -------------------------------

    def print_class_distribution(self):
        counter = Counter()

        for _, label_idx in self.samples:
            label_name = self.index_to_label[label_idx]
            counter[label_name] += 1

        print("\nClass distribution:")
        for label in FINAL_LABELS:
            print(f"{label}: {counter[label]}")

        if counter["fall"] == 0:
            print("\nNote: No 'fall' data found (expected with UCF101)")

    # -------------------------------
    # Core Dataset functions
    # -------------------------------

    def __len__(self):
        return len(self.samples)

    def convert_frames_to_tensor(self, frames):
        """
        Convert list of frames → tensor
        Shape: (16, 3, 224, 224)
        """
        frame_list = []

        for frame in frames:
            frame_tensor = self.transform(frame)
            frame_list.append(frame_tensor)

        return torch.stack(frame_list)

    def __getitem__(self, index):
        video_path, target = self.samples[index]

        # Load RGB + motion frames
        frames, diff_frames = load_video(video_path)

        # Convert to tensors
        frames = self.convert_frames_to_tensor(frames)
        diff_frames = self.convert_frames_to_tensor(diff_frames)

        if self.train:
            label = torch.tensor(target)
            return frames, diff_frames, label

        return frames, diff_frames, target


# -------------------------------
# Quick test
# -------------------------------

if __name__ == "__main__":
    dataset = VideoDataset("data/train", train=True)

    print("\nTotal samples:", len(dataset))

    if len(dataset) > 0:
        frames, diff_frames, label = dataset[0]

        print("Frames shape:", frames.shape)
        print("Diff shape:", diff_frames.shape)
        print("Label:", label.item())
        print("Label name:", dataset.index_to_label[label.item()])