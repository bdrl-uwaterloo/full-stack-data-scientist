# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


# %%
import pathlib
os.chdir('../../Data')
data_dir = os.path.join('coil_100')
data_dir= pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.png'))) #data_dir.glob will scrap whatever it is in the data_dir.
class_name = os.listdir(data_dir)


# %%
image_data_generator = preprocessing.image.ImageDataGenerator(rescale=1./255)
train_data_gen = image_data_generator.flow_from_directory(directory=data_dir, target_size = (28,28), batch_size=100, shuffle=True, classes = list(class_name))


# %%
from tensorflow.keras import preprocessing
BATCH_SIZE =100
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
def show_batch(image_batch ):
  plt.figure(figsize=(8,8))
  for i in range(9):
      ax = plt.subplot(3,3,i+1)
      plt.imshow(image_batch[i])
      plt.axis('off')


# %%
image_batch, label_batch = next(iter(train_data_gen))
show_batch(image_batch )
#plt.savefig('coil100_2828.png', dpi=72, bbox_inches='tight')


# %%
train_data_gen = image_data_generator.flow_from_directory(directory=data_dir, target_size = (128,128),
                                                              batch_size=100, shuffle=True,
                                                              classes = list(class_name))
image_batch, label_batch = next(iter(train_data_gen))
show_batch(image_batch )


# %%
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Input, Reshape, Flatten, Conv2DTranspose, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential, Model
img_width, img_height = 28,28
input_depth =3
input_shape_val = (img_width, img_height,input_depth)

input_img = Input(shape=input_shape_val, name = 'input')
encoder = Conv2D(32, (3, 3), activation='relu', padding='same' )(input_img)
encoder = MaxPooling2D((2, 2), padding='same')(encoder)
encoder = Conv2D(16, (3, 3), activation='relu', padding='same' )(input_img)
encoder = MaxPooling2D((2, 2), padding='same')(encoder)
encoder = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder)
encoder = MaxPooling2D((2, 2), padding='same')(encoder )
encoder  = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder)

encoded = MaxPooling2D((2, 2), padding='same')(encoder )
#DECODER
decoder = Conv2D(8,(3, 3),  activation='relu', padding='same')(encoded)
decoder = UpSampling2D((2,2))(decoder)
decoder = Conv2D(8,(3, 3),  activation='relu', padding='same')(decoder)
decoder = UpSampling2D((2,2))(decoder)
decoder = Conv2D(16,(3, 3), activation='relu')(decoder)
decoder = UpSampling2D((2,2))(decoder)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(decoder)
auto_encoder = Model(input_img, decoded)
auto_encoder.summary()


# %%
def train_images ():
    
    image_data_generator = preprocessing.image.ImageDataGenerator(rescale=1./255,dtype='float32')
    train_data_gen = image_data_generator.flow_from_directory(directory=data_dir, target_size = (28,28),shuffle=True,
                                                              classes = list(class_name))
    return train_data_gen[0][0].astype('float32'), train_data_gen[0][1].astype('float32')

 


# %%
auto_encoder.compile(optimizer="adadelta", loss="mse")
auto_encoder.fit(next(iter( train_images())),next(iter( train_images())),steps_per_epoch=100, batch_size= 20, epochs =500)


# %%
prediction = auto_encoder.predict(train_images(), verbose=1, batch_size=150)
# you can now display an image to see it is reconstructed well
x =prediction[11]
plt.imshow(x)
#plt.savefig('AE_output1.png', dpi=72, bbox_inches='tight')

# %%
## Save model

# %%
autoencoder_save = auto_encoder.to_json()
with open ('autoencoder.json', 'w') as json_file:
    json_file.write(autoencoder_save)


# %%
## Load pretrained model


# %%
json_file = open('autoencoder.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

from tensorflow.keras.models import model_from_json
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_tex.h5")
print("Loaded model from disk")
y_pred = loaded_model.predict(train_images(), verbose=1, batch_size=10)
