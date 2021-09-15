import numpy as np

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

def part_img_test(x):
  '''
  Деление тестовых картинок на фрагменты,
  с помощью ф-ии part_img()
  x - тестовая выборка
  '''
  trainx = np.array([])

  for i,img in enumerate(x):
    img_parts_x = part_img(img)                      # Деление на фрагменты размером (32,32)

    try:
      trainx = np.concatenate([trainx, img_parts_x]) # Объединение фрагментов в одну выборку
    except:
      trainx = img_parts_x
      print('!!!!!!!!!!! concatenate at the begining time', i, img_parts_x.shape)
  
  trainx = np.array(trainx)
  return trainx

# Деление тестовой выборки на фрагменты размером (32,32)
img_parts_test = part_img_test(test1)
img_parts_test2 = part_img_test(test2)

