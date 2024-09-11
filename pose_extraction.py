import numpy as np
import os
import mediapipe as mp
import cv2
import pickle
from utils import *
from jaad_data import JAAD

"""
    Pose extraction with mediapipe and Dataset organization using JAAD Dataset annotations

"""

jaad = JAAD(data_path='/JAAD')

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.6)

checkpoint_path = './definitive_pose_checkpoints'


def generate_trajectories(data, annotations, start_idx=0, num_frames=16, prediction_frame_step=8 ):
    global_context, speed, bbox, pose, local_context, labels = [], [], [], [], [], []

    for idx in range(start_idx, len(data['image'])):
        print(f'iteration: {idx} of ',len(data['image']))

        #exclude frames smaller than the frame prediction length
        if len(data['image'][idx]) < num_frames+prediction_frame_step:
          print(f'skip index: {idx}, because contains only:', len(data['image'][idx]), 'frames')
          continue

        # Exclude occluded frames
        #elif any(frame == 2 for frame in data['occlusion'][idx][:num_frames]):
        #  print(f"Skip index: {idx}, because contains frames with occlusion: {data['occlusion'][idx]}")
        #  print(f"Skipped labels {data['intent'][idx]}")
        #  continue


        image_sequence = data['image'][idx]
        #global_context_seq = image_sequences[:num_frames]

        #local_context_extraction
        local_context_seq = [extract_local_context(cv2.imread(path), data['center'][idx][j])
                              for j,path in enumerate(image_sequence[:num_frames]) ]
        #FOR DEBUGGING
        #for im in local_context_seq :
        #  plt.figure()
        #  plt.imshow(im)
        #print(len(local_context_seq))

        # Extract video id and frame number
        speed_seq = []
        for image_path in image_sequence[:num_frames]:
          for string in image_path.split(os.sep):
              if 'video_' in string:
                  video_id = string  # Extract the current video
                  img = os.path.basename(image_path)
                  img = int(img[:-4])  # Extract the current image number
                  break

          # Speed extraction
          speed_annot = annotations[video_id]['vehicle_annotations'].get(img, 0)
          speed_seq.append(speed_annot)

        #bbox extraction
        bbox_seq = data['bbox'][idx][:num_frames]


        #print(f'labels on index {idx}:',data['intent'][idx])
        intent_label = data['intent'][idx][num_frames+prediction_frame_step-1]
        #print('label on this sample:',intent_label)

        # Pose extraction
        pose_seq = []
        for img in local_context_seq:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose_model.process(img_rgb)
            if results.pose_landmarks:
                keypoints = [[lmk.x, lmk.y] for lmk in results.pose_landmarks.landmark[:32]]
            else:
                keypoints = np.zeros((32, 2))
            pose_seq.append(np.ravel(keypoints).tolist())

        # Append to sequences
        #global_context.append(local_context_seq)
        local_context.append(local_context_seq)
        speed.append(speed_seq)
        bbox.append(bbox_seq)
        pose.append(pose_seq)
        labels.append(intent_label)


        # Save checkpoint after processing 10 trajectories
        if idx % 10 ==0:
          save_checkpoint(checkpoint_path, {
              'start_idx': idx + 1,
              #'global_context': global_context,
              'speed': speed,
              'bbox': bbox,
              'pose': pose,
              'local_context': local_context,
              'labels': labels
          })
          #Reinitialization
          global_context, speed, bbox, pose, local_context, labels = [], [], [], [], [], []




if __name__ == "__main__":
  # Load dataset
  data = jaad.generate_data_trajectory_sequence('train')
  annotations = jaad.generate_database()

  # Load checkpoint data
  checkpoint_data = load_checkpoint(checkpoint_path)
  start_idx = checkpoint_data['start_idx']

  # Generate trajectories and process data
  generate_trajectories(data, annotations, start_idx=start_idx)
