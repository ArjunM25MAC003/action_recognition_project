'''Read a video, extract frames, and compute motion frames'''

import cv2
import numpy as np


def get_frames_from_video(video_path, num_frames=16, size=224):
    """
    Extract fixed number of frames from a video.

    Why fixed frames?
    → So every video has the same shape (important for model input)

    Why 16?
    → Enough to capture short actions (fight, fall) without being too heavy
    """
    extracted_frames = []

    # fallback (in case video fails)
    empty_frames = np.zeros((num_frames, size, size, 3), dtype=np.uint8)

    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return empty_frames

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            print(f"Empty or broken video: {video_path}")
            cap.release()
            return empty_frames

        # pick evenly spaced frames across the video
        frame_positions = np.linspace(0, total_frames - 1, num_frames).astype(int)

        for pos in frame_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
            success, frame = cap.read()

            if not success or frame is None:
                # if frame fails, replace with black frame
                frame = np.zeros((size, size, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (size, size))

            extracted_frames.append(frame)

        cap.release()

        if len(extracted_frames) != num_frames:
            print(f"Frame mismatch: expected {num_frames}, got {len(extracted_frames)}")
            return empty_frames

        return np.array(extracted_frames, dtype=np.uint8)

    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        return empty_frames


def get_motion_frames(frames):
    """
    Create motion frames using frame differencing.

    Idea:
    → Instead of learning motion from scratch, we explicitly show movement

    Why useful?
    → Helps detect:
        - fights (fast motion)
        - falls (sudden change)
        - panic (rapid movement)
    """
    motion_frames = []

    # first frame has no previous frame → use zero
    first_frame = np.zeros_like(frames[0])
    motion_frames.append(first_frame)

    for i in range(1, len(frames)):
        current = frames[i].astype(np.int16)
        previous = frames[i - 1].astype(np.int16)

        diff = np.abs(current - previous)
        diff = np.clip(diff, 0, 255).astype(np.uint8)

        motion_frames.append(diff)

    return np.array(motion_frames, dtype=np.uint8)


def load_video(video_path):
    """
    Main helper function used in dataset.py

    Returns:
    - RGB frames
    - Motion (diff) frames
    """
    frames = get_frames_from_video(video_path)
    motion_frames = get_motion_frames(frames)
    return frames, motion_frames


if __name__ == "__main__":
    test_video = "sample_video.mp4"

    frames, motion_frames = load_video(test_video)

    print("RGB frames shape:", frames.shape)
    print("Motion frames shape:", motion_frames.shape)