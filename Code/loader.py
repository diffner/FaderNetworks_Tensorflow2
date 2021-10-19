#Load and pre-process data
import numpy as np
import os
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from ipywidgets import IntProgress
from Code.loader import *
import pickle 

#For normalizing images
#print(normalize_images(np.uint8(np.load('./Data/test_data.npy')[:1000, :, :, :])))

AVAILABLE_ATTR = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
    "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
    "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
    "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
    "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
    "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young"
]

def normalize_images(images):
  """
  Normalize image values.
  """

  return ((images / 255.0) * 2.0) -1.0

def unnormalize_images(images):
  """
  Reverse the normalized image values.
  """
  return np.uint8(((images + 1.0) / 2.0) * 255.0)
  
def two_hot(data, attribute_list, number_of_images = 0):
  """
  Parameters:
    - data: A dictionary of the attributes for each image. Key is a string and value is a boolean array
    - attribute_list: A list of the attributes that we wish to convert to a two hot encoding
    - number_of_images: The amount of images that we desire to have two hot encoded

  Returns: 
    - two_hot: A 3d matrix, attribute x 2 for each image. Each row is either [1,0] or [0,1]. 
  """  
  if(number_of_images == 0):
      print("Error: No number of images was given!")
      return

  two_hot = np.zeros((number_of_images, 2*len(attribute_list), 1), dtype=np.int8)
  for img in range(number_of_images):
    for i, att in enumerate(attribute_list): 
        two_hot[img, 2*i, 0] = int(np.invert(data[att][img]) * 1)
        two_hot[img, 2*i + 1, 0] = int(data[att][img] * 1) #Converts the boolean to 0 or 1. 

  return tf.cast(two_hot, 'float32')

'''
#For reading in attributes and changing to binary from Boolean. 
a_file = open("./Data/test_attributes_dict.pkl", "rb")
test_att = pickle.load(a_file)

#print(test_att['Big_Lips'][:3])
#print(np.invert(test_att['Big_Lips'][:3]))

attributes = ["Big_Lips", "Wearing_Hat", "Smiling", "Young", "No_Beard"]
test_two_hot = two_hot(test_att, attributes, 3)
#print(test_two_hot[0, :, :])
#print(test_att['Big_Lips'][:3] * 1)
#test_att = pickle.load()
'''
