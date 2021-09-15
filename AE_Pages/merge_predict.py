import os,sys
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt

def merge_part_img(img, n): 
  ''' 
  Объединение тестовой выборки после predict
  img - фрагменты после predict
  n - 
  '''
  k=0
  for j in range(0, n):
    for i in range(0, 16):
      if i == 0:
        x_test = np.concatenate([img[k], img[k+1]],axis=1)
      else:
        x_test = np.concatenate([x_test, img[k+1]],axis=1)
      k+=1
    k+=1

    if j == 0:
      xx_test = x_test
    else:
      xx_test = np.concatenate([xx_test, x_test],axis=0)
  return xx_test

def compare_part_img(img_parts_test, train_test):
  temp = 0
  n = int(train_test.shape[1]/32)
  step = int(n*(train_test.shape[2]/32))
  while temp < img_parts_test.shape[0]:
    tes = merge_part_img(img_parts_test[temp:temp+step], n)
    tes = np.expand_dims(tes, axis=0)
    if temp == 0:
      test_img = tes
    else:
      test_img = np.concatenate([test_img, tes])
    temp+=step
  return test_img


