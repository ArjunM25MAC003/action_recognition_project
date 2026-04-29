import torch
import torch.nn as nn
from torchvision import models


class ActionRecognizer(nn.Module):
    def __init__(self, num_classes=5, hidden_size=128):
        super(ActionRecognizer, self).__init__()

        try:
            mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        except AttributeError:
            mobilenet = models.mobilenet_v2(pretrained=True)

        # MobileNetV2 is frozen because it already learned strong image features
        # from ImageNet. This makes training faster and more real-time friendly.
        for param in mobilenet.parameters():
            param.requires_grad = False

        self.feature_extractor = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # The two streams improve accuracy because appearance and motion answer
        # different questions: what is visible, and how it is moving.
        self.lstm = nn.LSTM(
            input_size=2560,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # The LSTM is used because action recognition depends on frame order.
        # A fall or fight is not just one image; it is a short motion sequence.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def extract_features(self, frame):
        features = self.feature_extractor(frame)
        features = self.pool(features)
        features = torch.flatten(features, 1)
        return features

    def forward(self, frames, diff_frames):
        spatial_features = []
        motion_features = []

        # Loop through each time step (each frame)
        for t in range(frames.shape[1]):
            rgb_frame = frames[:, t]
            diff_frame = diff_frames[:, t]

            spatial_feature = self.extract_features(rgb_frame)
            motion_feature = self.extract_features(diff_frame)

            spatial_features.append(spatial_feature)
            motion_features.append(motion_feature)

        spatial_seq = torch.stack(spatial_features, dim=1)
        motion_seq = torch.stack(motion_features, dim=1)

        combined_seq = torch.cat([spatial_seq, motion_seq], dim=2)

        lstm_output, _ = self.lstm(combined_seq)

        last_output = lstm_output[:, -1]
        logits = self.classifier(last_output)

        return logits


if __name__ == "__main__":
    model = ActionRecognizer(num_classes=5)

    frames = torch.randn(2, 16, 3, 224, 224)
    diff_frames = torch.randn(2, 16, 3, 224, 224)

    logits = model(frames, diff_frames)
    print("Output shape:", logits.shape)