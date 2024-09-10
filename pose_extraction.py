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


def generate_trajectories(data_train, annotations, start_idx=0, num_frames=16, prediction_frame_step=8 ):
    global_context, speed, bbox, pose, local_context, labels = [], [], [], [], [], []

    def trim_to_divisible_by_16(trajectory, num_frames, prediction_frame_step ):
        trajectory=trajectory[:-prediction_frame_step-1]
        length = len(trajectory)
        return trajectory[:length - (length % num_frames)] if length % num_frames != 0 else trajectory

    for idx in range(start_idx, len(data_train['image'])):
        print(f'iteration: {idx} of ',len(data_train['image']))

        if len(data_train['image'][idx]) < num_frames+prediction_frame_step:
          print(f'skip index: {idx}, because contains only:', len(data_train['image'][idx]), 'frames')
          continue
        image_sequence = trim_to_divisible_by_16(data_train['image'][idx], num_frames,prediction_frame_step)
        num_sequences = len(image_sequence) // num_frames

        print(f'image to process: {len(image_sequence)}')

        for i in range(num_sequences):
            start_idx, end_idx = i * num_frames, (i + 1) * num_frames
            #images = [cv2.imread(path) for path in image_sequence[start_idx:end_idx]]
            local_context_seq = [extract_local_context(cv2.imread(path), data_train['center'][idx][start_idx + j])
                                 for j, path in enumerate(image_sequence[start_idx:end_idx])]
            #FOR DEBUGGING
            #for im in local_context_seq :
            #  plt.figure()
            #  plt.imshow(im)
            #print(len(local_context_seq))

            # Extract video id and frame number
            speed_seq = []
            for image_path in image_sequence[start_idx:end_idx]:
              for string in image_path.split(os.sep):
                  if 'video_' in string:
                      video_id = string  # Extract the current video
                      img = os.path.basename(image_path)
                      img = int(img[:-4])  # Extract the current image number
                      break

              # Speed extraction
              speed_annot = annotations[video_id]['vehicle_annotations'].get(img, 0)
              speed_seq.append(speed_annot)


            bbox_seq = data_train['bbox'][idx][start_idx:end_idx]

            
            #print('labels: ',data_train['intent'][idx])
            intent_label = data_train['intent'][idx][end_idx+prediction_frame_step]
            #print('label on this sample:',intent_label)

            # Pose estimation
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
            #global_context.append(images)
            local_context.append(local_context_seq)
            speed.append(speed_seq)
            bbox.append(bbox_seq)
            pose.append(pose_seq)
            labels.append(intent_label)


        # Save checkpoint after processing each trajectory
        save_checkpoint(checkpoint_path, {
            'start_idx': idx + 1,
            #'global_context': global_context,
            'speed': speed,
            'bbox': bbox,
            'pose': pose,
            'local_context': local_context,
            'labels': labels
        })




if __name__ == "__main__":
  # Load dataset
  data = jaad.generate_data_trajectory_sequence('train')
  annotations = jaad.generate_database()

  # Load checkpoint data
  checkpoint_data = load_checkpoint(checkpoint_path)
  start_idx = checkpoint_data['start_idx']

  # Generate trajectories and process data
  generate_trajectories(data, annotations, start_idx=start_idx)
