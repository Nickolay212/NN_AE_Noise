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

import processing_sample # предварительная обработка выборки
import processing_sample_test 
import merge_predict



# Тестовая выборка (преобразование в матрицу)
test1 =         processing_sample.test1
test2 =         processing_sample.test2

# Проверочная выборка (преобразование в матрицу, 
#                      разделение на фрагменты размером (32,32),
#                      объединение в массив)
trainx =        processing_sample.trainx
trainy =        processing_sample.trainy
print('trainx = {}, trainy = {}'.format(trainx.shape, trainy.shape))
processing_sample.show_sample(trainx, trainy, trainx.shape[0], 10)

# Проверочная выборка (преобразование в матрицу, 
#                      разделение на фрагменты размером (32,32),
#                      объединение в массив)
trainx_test =   processing_sample.trainx_test
trainy_test =   processing_sample.trainy_test
print('trainx_test = {}, trainy_test = {}'.format(trainx_test.shape, trainy_test.shape))
processing_sample.show_sample(trainx_test, trainy_test, trainx_test.shape[0], 10)


# Деление тестовой выборки на фрагменты размером (32,32)
img_parts_test = part_img_test(test1)
img_parts_test2 = part_img_test(test2)

# Predict
predict = modelUnet_main2.predict(img_parts_test)
predict1 = modelUnet_main2.predict(img_parts_test2)

# Целая тестовая картинка после predict
predict11 = compare_part_img(predict, test1)
predict22 = compare_part_img(predict1, test2)