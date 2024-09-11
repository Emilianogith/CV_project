import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json

from model import HybridFusionNetwork
from trainer import train_test_split

"""
    Train the model:
        Hyperparameters used:
        -epochs: 10
        -batch_size: 4
        -lr: 0.00005
        -L2_regularization: 0.01

    Finally save the parameters in '/content/drive/MyDrive/CV_Project/model_weights.pth'
    and Training and Validation Losses in '/content/drive/MyDrive/CV_Project/training_info.json'

"""



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using the device: {device}")



checkpoint_path = './definitive_pose_checkpoints'
train_dataset, val_dataset, test_dataset = train_test_split(checkpoint_path)

print('len of train dataset',train_dataset.__len__())


batch_size = 4

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)


model = HybridFusionNetwork(visual_input_shape=(16, 3, 224, 224),
                            pose_input_shape=(16, 64),
                            trajectory_input_shape=(16, 4),
                            speed_input_shape=(16, 1),
                            hidden_size=256).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=0.01)


num_epochs = 10
train_losses = []
val_losses = []

#train loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    print(f'Epoch [{epoch+1}/{num_epochs}]')

    for step,batch in enumerate(train_loader):
        batch_local_context = batch['local_context'].to(device)
        batch_speed = batch['speed'].to(device)
        batch_bbox = batch['bbox'].to(device)
        batch_pose = batch['pose'].to(device)
        batch_labels = batch['label'].to(device)

        optimizer.zero_grad()
        output = model(batch_local_context, batch_pose, batch_bbox, batch_speed)
        loss = criterion(output, batch_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()


    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Average Train Loss: {avg_train_loss}")

    #validation
    model.eval()
    val_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for batch in val_loader:
            val_local_context = batch['local_context'].to(device)
            val_speed = batch['speed'].to(device)
            val_bbox = batch['bbox'].to(device)
            val_pose = batch['pose'].to(device)
            val_labels = batch['label'].to(device)

            output = model(val_local_context, val_pose, val_bbox, val_speed)
            val_loss += criterion(output, val_labels).item()

            predicted_labels = (output > 0.5).float()

            correct_predictions += (predicted_labels == val_labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    val_accuracy = correct_predictions / len(val_dataset)
    print(f"Validation Loss: {avg_val_loss} | Validation accuracy: {val_accuracy}")

torch.save(model.state_dict(), './training_info.json')

# Save losses and epochs information
training_info = {
    'num_epochs': num_epochs,
    'train_losses': train_losses,
    'val_losses': val_losses
}

with open('./model_weights.pth', 'w') as f:
    json.dump(training_info, f)


