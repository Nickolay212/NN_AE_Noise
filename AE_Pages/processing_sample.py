
import os,sys
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt


def img2matrix(img, path):
  '''
  Преобразование пикселей в матрицу
  В данной задаче используются картинки двух видов размера
  img - картинка без преобразования
  path - путь к картинке
  '''

  path = os.path.join(path, img)
  x = image.load_img(path)
  x = image.img_to_array(x)

  # Изменим размеры картинок, чтобы можно было использовать
  # кратное число для разбиения на маленькие фрагменты
  if x.shape == (258, 540, 3):                 # текущий размер картинки (1)
    img_sh = (256, 544, 3)                     # измененный размер картинки (1)
  else:
    img_sh = (416, 544, 3)                     # измененный размер картинки (2)
  x = image.load_img(path, target_size=img_sh) # загружаем картинку с измененным размером
  x = image.img_to_array(x)
  x = x.astype('float32')/255                  # нормализация, для уменьшения нагрузки на НС
  return x

def part_img(img): 
  '''
  Подготовка выборки, деление картинки на части
  img - картинки в матричном виде с измененным размером
  '''
  x_max = img.shape[1]                          # выделение размера по оси x
  y_max = img.shape[0]                          # выделение размера по оси y
  img_parts = []
  step = 32                                     # шаг разбиения
  i, j = 0, 0
  # разбиение картинки на фрагменты с размерами (32,32)
  while j < y_max:
    while i < x_max:
      img_parts.append(img[j:j+step, i:i+step])
      i+=step
    i=0
    j+=step
  img_parts = np.array(img_parts)
  return img_parts 

def division_img(x, y):
  '''
  Объединение полученных фрагментов в выборки,
  Для обучающей и проверочной выборки сопоставляем 
  соответствующие истинные фрагменты (выход нейронки)
  x - обучающая и проверочная выборка
  y - истинные фрагменты
  '''
  trainx = np.array([])
  trainy = np.array([])

  for i,img in enumerate(x):                    # перебор x
    img_y = y[i]                                # перебор y
    img_parts_x = part_img(img)
    img_parts_y = part_img(img_y)
    try:                                        # Создание выборки, где y соответствует x
      trainx = np.concatenate([trainx, img_parts_x])
      trainy = np.concatenate([trainy, img_parts_y])
    except:
      trainx = img_parts_x                      # не совпадение размеров при объединении
      trainy = img_parts_y                      # не совпадение размеров при объединении
      print('!!!!!!!!!!! concatenate at the begining time', i, img_parts_x.shape, img_parts_y.shape)
  
  trainx = np.array(trainx)
  trainy = np.array(trainy)
  return trainx, trainy

def show_sample(x, y, num_sample, num_img):
  '''
  Отрисовка фрагментов
  x - обучающая и проверочная выборка
  y - истинные фрагменты
  num_sample - количество фрагментов
  num_img - необходимое количество для отрисовки фрагментов
  '''
  fig = plt.figure(figsize = (25, 10))
  xx = fig.subplots(2, num_img)
  for i in range(0, num_img):
    n = np.random.randint(num_sample)
    xx[0,i].imshow(x[n])
    xx[0,i].set_title(n)
    xx[1,i].imshow(y[n])
    xx[1,i].set_title(n)
    
    for i in xx.flat:
      i.set_xticks([])
      i.set_yticks([])
  plt.show()

# ******************************************************************************************
def processing(path):
    path_files = path # путь к файлам
    variants_sample = os.listdir(path_files)

    train_X1,train_X2 = [],[]
    train_Y1, train_Y2 = [],[]
    test1, test2    = [],[]
    # Предварительная обработка изображений
    # Преобразование в матрицу, Сбор изображений в массивы по соотвествующим группам
    for temp in variants_sample:
      path = os.path.join(path_files, temp) 
      for image_ in np.sort(os.listdir(path)):
        x = img2matrix(image_, path)
        if temp == 'train_X':
          if x.shape == (256, 544, 3):
            train_X1.append(x)
          else:
            train_X2.append(x)
        elif temp == 'train_Y':
          if x.shape == (256, 544, 3):
            train_Y1.append(x)
          else:
            train_Y2.append(x)
        elif temp == 'test':
          if x.shape == (256, 544, 3):
            test1.append(x)
          else:
            test2.append(x)

    # Картинки двух видов размера  (256, 544, 3), (416, 544, 3)  
    train_X1 = np.array(train_X1)
    train_X2 = np.array(train_X2)
    train_Y1 = np.array(train_Y1)
    train_Y2 = np.array(train_Y2)
    test1 = np.array(test1)
    test2 = np.array(test2)

    # Подготовка выборки для обучения (на фрагменты, объединение в массив)
    img_parts_x1, img_parts_y1 = division_img(train_X1[:40], train_Y1[:40])
    img_parts_x2, img_parts_y2 = division_img(train_X2[:90], train_Y2[:90])
    trainx = np.concatenate([img_parts_x1, img_parts_x2])
    trainy = np.concatenate([img_parts_y1, img_parts_y2])

    # Подготовка выборки для тестирования (на фрагменты, объединение в массив)
    img_parts_x11, img_parts_y11 = division_img(train_X1[40:], train_Y1[40:])
    img_parts_x22, img_parts_y22 = division_img(train_X2[90:], train_Y2[90:])
    trainx_test = np.concatenate([img_parts_x11, img_parts_x22])
    trainy_test = np.concatenate([img_parts_y11, img_parts_y22])
    return trainx, trainy, trainx_test, trainy_test, test1, test2