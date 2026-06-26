import torch
import torch.nn as nn
import joblib
import pandas as pd

# MODEL ARCHITECTURE
class YogaPoseModel(nn.Module):
    def __init__(self, input_size=80, num_classes=10):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# LOAD MODEL
device = torch.device("cpu")

model = YogaPoseModel(input_size=80)

model.load_state_dict(
    torch.load(
        "models/yoga_pose_model.pth",
        map_location=device
    )
)

model.eval()

# LOAD SCALER
scaler = joblib.load("models/scaler.save")

# LOAD LABELS
label_df = pd.read_csv("models/label_map.csv")

label_map = {}

for _, row in label_df.iterrows():
    label_map[int(row["pose"])] = row["label"]