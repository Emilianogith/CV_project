from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from trainer import train_test_split
from utils import plot_losses
import json
import numpy as np

from model import HybridFusionNetwork

"""
    -Plot the Training and Validation losses during training
    -Evaluate performance of the trained model on the test set
"""

#Plot Training and Validation Losses during training
with open('./training_info.json', 'r') as f:
    training_info = json.load(f)

num_epochs = training_info['num_epochs']
train_losses = training_info['train_losses']
val_losses = training_info['val_losses']

plot_losses(train_losses, val_losses, num_epochs)


#Evaluate performance on test set
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using the device: {device}")


checkpoint_path = './definitive_pose_checkpoints'
train_dataset, val_dataset, test_dataset = train_test_split(checkpoint_path)

batch_size=4
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

model = HybridFusionNetwork(visual_input_shape=(16, 3, 224, 224),
                            pose_input_shape=(16, 64),
                            trajectory_input_shape=(16, 4),
                            speed_input_shape=(16, 1),
                            hidden_size=256).to(device)
criterion = nn.BCELoss()

# test
model.load_state_dict(torch.load('./model_weights.pth'))
model.eval()
test_loss = 0
correct_predictions = 0
all_labels = []
all_predictions = []
with torch.no_grad():
    for batch in test_loader:
        test_local_context = batch['local_context'].to(device)
        test_speed = batch['speed'].to(device)
        test_bbox = batch['bbox'].to(device)
        test_pose = batch['pose'].to(device)
        test_labels = batch['label'].to(device)

        output = model(test_local_context, test_pose, test_bbox, test_speed)
        test_loss += criterion(output, test_labels).item()

        predicted_labels = (output > 0.5).float()

        correct_predictions += (predicted_labels == test_labels).sum().item()

        all_labels.append(test_labels.cpu().numpy())
        all_predictions.append(predicted_labels.cpu().numpy())

avg_test_loss = test_loss / len(test_loader)
test_accuracy = correct_predictions / len(test_dataset)

all_labels = np.concatenate(all_labels)
all_predictions = np.concatenate(all_predictions)

class_report = classification_report(all_labels, all_predictions)
print(class_report)