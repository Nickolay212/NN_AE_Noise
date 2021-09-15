# Используемые библиотеки
import numpy as np
from tensorflow.keras.optimizers import Adam,Adamax
from tensorflow.keras.layers import InputLayer,Dense, MaxPooling2D, UpSampling2D,\
  Conv2D, Dropout, Reshape, Flatten, Conv2DTranspose,Input,BatchNormalization,Activation,\
  concatenate
from tensorflow.keras.models import Model, Sequential,load_model
from sklearn.preprocessing import StandardScaler
from keras.preprocessing import image
from PIL import Image

import matplotlib.pyplot as plt
from PIL import ImageFilter


