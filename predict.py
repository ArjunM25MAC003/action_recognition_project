from collections import Counter, deque

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import FINAL_LABELS, VideoDataset
from model import ActionRecognizer


def smooth_prediction(current_label, history, window_size=2):
    """
    Temporal smoothing reduces flickering predictions across nearby videos or
    clips by using the most common label in a small recent window.
    Smaller window = less bias.
    """
    history.append(current_label)

    if len(history) > window_size:
        history.popleft()

    counts = Counter(history)
    return counts.most_common(1)[0][0]


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    # Load model (we saved only state_dict)
    model = ActionRecognizer(num_classes=5)
    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Label mapping
    idx_to_label = {i: label for i, label in enumerate(FINAL_LABELS)}

    return model, idx_to_label


def main():
    test_folder = "data/test"
    model_path = "best_model.pth"
    output_file = "submission.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_dataset = VideoDataset(test_folder, train=False)

    if len(test_dataset) == 0:
        print("No test videos found.")
        return

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model, idx_to_label = load_model(model_path, device)

    history = deque()
    results = []
    prediction_counts = Counter()

    with torch.no_grad():
        for frames, diff_frames, video_ids in test_loader:
            frames = frames.to(device)
            diff_frames = diff_frames.to(device)

            logits = model(frames, diff_frames)

            pred_idx = torch.argmax(logits, dim=1).item()
            pred_label = idx_to_label[pred_idx]

            # Apply smoothing only after first prediction
            if len(history) > 0:
                pred_label = smooth_prediction(pred_label, history)

            video_id = video_ids[0]

            results.append({
                "video_id": video_id,
                "predicted_label": pred_label
            })

            prediction_counts[pred_label] += 1

            print(f"{video_id} -> {pred_label}")

    submission = pd.DataFrame(results)
    submission.to_csv(output_file, index=False)

    print(f"\nSaved submission to {output_file}")
    print("\nPrediction distribution:")

    for label, count in prediction_counts.items():
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()