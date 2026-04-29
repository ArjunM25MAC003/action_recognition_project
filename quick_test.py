import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from extract_frames import load_video
from model import ActionRecognizer


image_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def show_frames(frames, title):
    plt.figure(figsize=(8, 8))

    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(frames[i])
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def convert_to_tensor(frames):
    frame_list = []

    for frame in frames:
        frame_tensor = image_transform(frame)
        frame_list.append(frame_tensor)

    tensor = torch.stack(frame_list)
    tensor = tensor.unsqueeze(0)

    return tensor


def main():
    video_path = "sample_video.mp4"

    frames, diff_frames = load_video(video_path)

    print("Raw frames shape:", frames.shape)
    print("Raw diff frames shape:", diff_frames.shape)

    show_frames(frames, "RGB Frames")
    show_frames(diff_frames, "Motion Frames")

    frames_tensor = convert_to_tensor(frames)
    diff_tensor = convert_to_tensor(diff_frames)

    print("Frames tensor shape:", frames_tensor.shape)
    print("Diff tensor shape:", diff_tensor.shape)

    model = ActionRecognizer(num_classes=5)
    model.eval()

    with torch.no_grad():
        logits = model(frames_tensor, diff_tensor)

    print("Output shape:", logits.shape)
    print("Logits:", logits)


if __name__ == "__main__":
    main()