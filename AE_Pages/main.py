# Используемые библиотеки
import numpy as np
import os,sys
from tensorflow.keras.optimizers import Adam,Adamax
from tensorflow.keras.layers import InputLayer,Dense, MaxPooling2D, UpSampling2D,\
  Conv2D, Dropout, Reshape, Flatten, Conv2DTranspose,Input,BatchNormalization,Activation,\
  concatenate
from tensorflow.keras.models import Model, Sequential,load_model
from sklearn.preprocessing import StandardScaler
from keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt

import processing_sample            # предварительная обработка выборки
import processing_sample_test       # предварительная обработка тестовой выборки
import merge_predict                # тестовая выборка после predict

path_files = '../text_cleaning/'    # путь к файлам
variants_sample = os.listdir(path_files)

trainx, trainy, trainx_test, trainy_test, test1, test2 = processing_sample.processing(path_files)
## Тестовая выборка (преобразование в матрицу)
#test1 test2 

## Обучающая выборка (преобразование в матрицу, 
##                      разделение на фрагменты размером (32,32),
##                      объединение в массив)
#trainx trainy 
#print('trainx = {}, trainy = {}'.format(trainx.shape, trainy.shape))
#processing_sample.show_sample(trainx, trainy, trainx.shape[0], 10)

## Проверочная выборка (преобразование в матрицу, 
##                      разделение на фрагменты размером (32,32),
##                      объединение в массив)
#trainx_test trainy_test 
#print('trainx_test = {}, trainy_test = {}'.format(trainx_test.shape, trainy_test.shape))
#processing_sample.show_sample(trainx_test, trainy_test, trainx_test.shape[0], 10)

## Деление тестовой выборки на фрагменты размером (32,32) - два вида размеров
img_parts_test = processing_sample_test.part_img_test(test1)
img_parts_test2 = processing_sample_test.part_img_test(test2)

# Predict2
# Модель НС Unet обучена в google colab (больше вычислительной мощности)
model = load_model('./modelUnet.h5')                # загрузка модели НС
predict = model.predict(img_parts_test)
predict1 = model.predict(img_parts_test2)

# Целая тестовая картинка после predict
predict11 = merge_predict.compare_part_img(predict, test1)
predict22 = merge_predict.compare_part_img(predict1, test2)

# Отрисовка результатов
processing_sample.show_sample(predict11, test1, predict11.shape[0], 4)
processing_sample.show_sample(predict22, test2, predict22.shape[0], 4)




