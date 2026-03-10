import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append('/content/gcp_pose_estimation')

from utils.dataset import GCPDataset
from models.gcp_model import GCPModel


def train():

    dataset_dir = "/content/drive/MyDrive/GCP_Assignment_Datasets/train_dataset"

    annotation_file = dataset_dir + "/gcp_marks.json"

    train_dataset = GCPDataset(dataset_dir, annotation_file)

    print("Total training samples:", len(train_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    model = GCPModel().to(device)

    coord_loss_fn = nn.MSELoss()
    shape_loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 1

    for epoch in range(epochs):

        model.train()

        running_loss = 0

        for patches, coords, shapes in train_loader:

            patches = patches.to(device)
            coords = coords.to(device)
            shapes = shapes.to(device)

            pred_coords, pred_shapes = model(patches)

            coord_loss = coord_loss_fn(pred_coords, coords)
            shape_loss = shape_loss_fn(pred_shapes, shapes)

            loss = coord_loss + shape_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "gcp_model.pth")

    print("Model saved as gcp_model.pth")


if __name__ == "__main__":
    train()