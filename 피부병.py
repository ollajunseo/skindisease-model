#!/usr/bin/env python
# coding: utf-8

# In[10]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
np.random.seed(123)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import itertools

import keras
from keras.utils import to_categorical  
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K
from keras.layers import BatchNormalization

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split


# In[11]:


def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['acc']) + 1, len(model_history.history['acc']) / 10))
    axs[0].legend(['train', 'val'], loc='best')
    
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1, len(model_history.history['loss']) / 10))
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()


# In[50]:


base_skin_dir = os.path.join('..', 'input')
# a = './pic/ISIC_0024306.jpg'
# print(os.path.exists(a))
# test = glob(os.path.join('./pic', '*', '*.jpg'))
# test1 = glob(os.path.join('./pic', '*.jpg'))

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join('./pic', '*.jpg'))}



lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}


# In[51]:


skin_df = pd.read_csv('./csv/HAM10000_metadata.csv')


skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get) 
skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes


# In[52]:


skin_df.head()


# In[53]:


skin_df['age'].fillna((skin_df['age'].mean()), inplace=True)


# In[54]:


skin_df.isnull().sum()


# In[55]:


skin_df.info()


# In[56]:


fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))
skin_df['cell_type'].value_counts().plot(kind='bar', ax=ax1)


# In[57]:


skin_df['dx_type'].value_counts().plot(kind='bar')


# In[58]:


skin_df['localization'].value_counts().plot(kind='bar')


# In[59]:


skin_df['age'].hist(bins=40)


# In[60]:


skin_df['sex'].value_counts().plot(kind='bar')


# In[61]:


sns.scatterplot(x='age',y='cell_type_idx',data=skin_df)


# In[62]:


sns.catplot(x='sex',y='cell_type_idx',data=skin_df)


# In[65]:


skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))


# In[64]:


print(skin_df['path'])


# In[ ]:





# In[ ]:




