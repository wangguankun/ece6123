import cv2
import torch

def load_images(number_of_frames):
  images_1 = get_images_from_folder('marple2/marple2_', 75, number_of_frames)
  images_2 = get_images_from_folder('marple17/marple17_', 234, number_of_frames)
  #images_3 = get_images_from_folder('shoe/images/shoe_', 600)
  #images_4 = get_images_from_folder('monkey5/monkey5_', 577)
  #images_5 = get_images_from_folder('birdhouse/images/birdhouse_', 270)
  #images_6 = get_images_from_folder('head/images/head_', 251)
  # images_7 = get_images_from_folder('tennis/tennis_', )
  images = images_1 + images_2 #+ images_3 + images_4 + images_5 + images_6
  return images

def get_images_from_folder(name, total_number_of_images, number_of_frames):
  number_of_frames += number_of_frames + 1
  sequences = list()
  for i in range(1, total_number_of_images - number_of_frames):
    sequence_of_images = list()
    for j in range(number_of_frames):
      frame_name = get_name_for_image(name, i + j, total_number_of_images)
      image = cv2.imread(frame_name).astype(float) #.reshape((384, 512, 3))
      image = cv2.resize(image, dsize=(384, 512), interpolation=cv2.INTER_CUBIC)
      sequence_of_images.append(image)
    
    interpolated_image = sequence_of_images.pop(int(number_of_frames / 2))
    interpolated_image = cv2.resize(interpolated_image, dsize=(96, 128), interpolation=cv2.INTER_CUBIC)
    length_of_sequence = len(sequence_of_images)
    for i in range(1, int(length_of_sequence / 2) - 1):
      sequence_of_images.pop(i)
      sequence_of_images.pop(len(sequence_of_images) - 1 - i)

    sequence_of_images_torch_tensor = torch.tensor(sequence_of_images[0]).permute(2, 0, 1)
    for i in range(1, len(sequence_of_images)):
      sequence_of_images_torch_tensor = torch.cat((sequence_of_images_torch_tensor, torch.tensor(sequence_of_images[i]).permute(2, 0, 1)), 0).type(torch.FloatTensor)
    interpolated_image = torch.tensor(interpolated_image).permute(2, 0, 1).type(torch.FloatTensor)
    sequences.append([sequence_of_images_torch_tensor, interpolated_image])
  return sequences

def get_name_for_image(name, i, number_of_images):
  number_string = ''
  if i < 10 and number_of_images >= 10 and number_of_images < 100:
    number_string = '0' + str(i)
  elif i >= 10 and number_of_images >= 10 and number_of_images < 100:
    number_string = str(i)
  elif i < 10 and number_of_images >= 100:
    number_string = '00' + str(i)
  elif i >= 10 and i < 100 and number_of_images >= 100:
    number_string = '0' + str(i)
  else:
    number_string = str(i)
  return name + number_string + '.jpg'