import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_local_context(image, center_point, context_size=(224, 224)):
    half_w, half_h = context_size[0] // 2, context_size[1] // 2
    cx, cy = center_point
    x1, y1 = max(0, int(cx - half_w)), max(0, int(cy - half_h))
    x2, y2 = min(image.shape[1], x1 + context_size[0]), min(image.shape[0], y1 + context_size[1])
    cropped_image = image[y1:y2, x1:x2]

    if cropped_image.shape[0] < context_size[1] or cropped_image.shape[1] < context_size[0]:
        cropped_image = cv2.copyMakeBorder(
            cropped_image,
            top=0, bottom=context_size[1] - cropped_image.shape[0],
            left=0, right=context_size[0] - cropped_image.shape[1],
            borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
    return cropped_image


def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    return {'start_idx': 0, 'speed': [], 'bbox': [], 'pose': [], 'local_context': [], 'labels': []}


def save_checkpoint(checkpoint_path, checkpoint_data):
    existing_data = load_checkpoint(checkpoint_path)

    existing_data['start_idx'] = checkpoint_data['start_idx']
    existing_data['speed'].extend(checkpoint_data['speed'])
    existing_data['bbox'].extend(checkpoint_data['bbox'])
    existing_data['pose'].extend(checkpoint_data['pose'])
    existing_data['local_context'].extend(checkpoint_data['local_context'])
    existing_data['labels'].extend(checkpoint_data['labels'])

    with open(checkpoint_path, 'wb') as f:
        pickle.dump(existing_data, f)


def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    return {'start_idx': 0, 'speed': [], 'bbox': [], 'pose': [], 'local_context': [], 'labels': []}


def show_an_example(local_context, pose, labels, random_idx, random_fr):
    image = local_context[random_idx][random_fr]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("label:", labels[random_idx])

    plt.figure()
    plt.imshow(image)

    x=[]
    y=[]
    print("pose:", pose[random_idx][random_fr])

    h,w,z = image.shape

    for i,point in enumerate(pose[random_idx][random_fr]):
        if i %2 ==0:
            x.append(int(point*w))
        else:
            y.append(int(point*h))
    for i in range(len(x)):
        plt.scatter(x,y)

def enumerate_pos_idx(labels):
    pos=0
    for index,i in enumerate(labels):
        if i[0]==1:
            print('positive index:',index)
            pos+=1
    print('positive labels:', pos )
    print('negative labels:', len(labels)-pos )

def balance_dataset(local_context, pose, bbox, speed, labels, remove_n_samples=10):
    local_context = np.array(local_context)
    pose = np.array(pose)
    bbox = np.array(bbox)
    speed = np.array(speed)
    labels = np.array(labels)

    negative_indices = np.where(labels == 0)[0]

    if len(negative_indices) < remove_n_samples:
        remove_n_samples = len(negative_indices)

    indices_to_remove = np.random.choice(negative_indices, size=remove_n_samples, replace=False)
    keep_indices = np.setdiff1d(np.arange(len(labels)), indices_to_remove)

    local_context = local_context[keep_indices]
    pose = pose[keep_indices]
    bbox = bbox[keep_indices]
    speed = speed[keep_indices]
    labels = labels[keep_indices]

    return local_context, pose, bbox, speed, labels


def plot_losses(train_losses, val_losses, num_epochs):
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

