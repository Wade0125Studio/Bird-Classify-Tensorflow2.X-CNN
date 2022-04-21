
#https://www.kaggle.com/gpiosenka/100-bird-species

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)

from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D


img = cv2.imread('C:/Users/GIGABYTE/Downloads/BIRD_CLASSIFICATION/images to test/3.jpg')
img.shape
plt.imshow(img)
plt.show()

train_dir = "C:/Users/GIGABYTE/Downloads/BIRD_CLASSIFICATION/train"
test_dir = "C:/Users/GIGABYTE/Downloads/BIRD_CLASSIFICATION/test"
val_dir = "C:/Users/GIGABYTE/Downloads/BIRD_CLASSIFICATION/valid"



train_data = tf.keras.preprocessing.image_dataset_from_directory(train_dir, label_mode='categorical',
                                                                 image_size=(224,224), batch_size=32)

test_data =  tf.keras.preprocessing.image_dataset_from_directory(test_dir, label_mode='categorical',
                                                                image_size=(224,224), batch_size=32)

val_data =  tf.keras.preprocessing.image_dataset_from_directory(val_dir, label_mode='categorical',
                                                                image_size=(224,224), batch_size=32)

len(train_data.class_names)
labels_names = train_data.class_names

plt.figure(figsize=(12,12))
for image, label in train_data.take(1):
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(image[i].numpy().astype('uint8'))
        plt.title(labels_names[tf.argmax(label[i])])
        plt.axis("off")
#---------------------------------------------------------
#Build EfficientNetB7 Model

base_model = tf.keras.applications.EfficientNetB7(include_top=False, weights='imagenet')
for layer in base_model.layers[:-5]:
    base_model.trainable = False
inputs = tf.keras.Input(shape=(224,224,3))
x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(325, activation='softmax')(x)
eff7_model = tf.keras.Model(inputs, outputs)
eff7_model.summary()
eff7_model.compile(
    loss = tf.keras.losses.categorical_crossentropy,
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics = ['accuracy'])


eff7_model.fit(train_data,
               epochs=20,
               steps_per_epoch = len(train_data),
               validation_data = val_data,
               validation_steps = len(val_data))

eff7_model_evaluation = eff7_model.evaluate(test_data)

print(f"Efficient Model Accuracy: {eff7_model_evaluation[1] * 100 : 0.2f}%")

test_labels = test_data.class_names
len(test_labels)

plt.figure(figsize=(16,16))
for image, label in test_data.take(1):
    model_prediction = eff7_model.predict(image)
    for i in range(18):
        plt.subplot(6,3,i+1)
        plt.imshow(image[i].numpy().astype('uint8'))
        plt.title(f"Model Prediction: {test_labels[tf.argmax(tf.round(model_prediction[i]))]}\nOriginal Labels: {test_labels[tf.argmax(label[i])]}")
        plt.axis("off")
#---------------------------------------------------------
#Build EfficientNetB6 Model        
        
base_model = tf.keras.applications.EfficientNetB6(include_top=False, weights='imagenet')
for layer in base_model.layers[:-5]:
    base_model.trainable = False
inputs = tf.keras.Input(shape=(224,224,3))
x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(325, activation='softmax')(x)
eff6_model = tf.keras.Model(inputs, outputs)
eff6_model.summary()
eff6_model.compile(
    loss = tf.keras.losses.categorical_crossentropy,
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics = ['accuracy'])
eff6_model.fit(train_data,
               epochs=20,
               steps_per_epoch = len(train_data),
               validation_data = val_data,
               validation_steps = len(val_data))

eff6_model_evaluation = eff6_model.evaluate(test_data)
print(f"Efficient Model Accuracy: {eff6_model_evaluation[1] * 100 : 0.2f}%")
test_labels = test_data.class_names
len(test_labels)
plt.figure(figsize=(16,16))
for image, label in test_data.take(1):
    model_prediction = eff6_model.predict(image)
    for i in range(18):
        plt.subplot(6,3,i+1)
        plt.imshow(image[i].numpy().astype('uint8'))
        plt.title(f"Model Prediction: {test_labels[tf.argmax(tf.round(model_prediction[i]))]}\nOriginal Labels: {test_labels[tf.argmax(label[i])]}")
        plt.axis("off")        
        
        
#---------------------------------------------------------
#Build EfficientNetB5 Model         
base_model = tf.keras.applications.EfficientNetB5(include_top=False, weights='imagenet')
for layer in base_model.layers[:-5]:
    base_model.trainable = False
inputs = tf.keras.Input(shape=(224,224,3))
x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(325, activation='softmax')(x)
eff5_model = tf.keras.Model(inputs, outputs)
eff5_model.summary()
eff5_model.compile(
    loss = tf.keras.losses.categorical_crossentropy,
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics = ['accuracy'])
eff5_model.fit(train_data,
               epochs=20,
               steps_per_epoch = len(train_data),
               validation_data = val_data,
               validation_steps = len(val_data))

eff5_model_evaluation = eff5_model.evaluate(test_data)
print(f"Efficient Model Accuracy: {eff5_model_evaluation[1] * 100 : 0.2f}%")
test_labels = test_data.class_names
len(test_labels)
plt.figure(figsize=(16,16))
for image, label in test_data.take(1):
    model_prediction = eff5_model.predict(image)
    for i in range(18):
        plt.subplot(6,3,i+1)
        plt.imshow(image[i].numpy().astype('uint8'))
        plt.title(f"Model Prediction: {test_labels[tf.argmax(tf.round(model_prediction[i]))]}\nOriginal Labels: {test_labels[tf.argmax(label[i])]}")
        plt.axis("off")        
        
#---------------------------------------------------------
#Build EfficientNetB4 Model         
base_model = tf.keras.applications.EfficientNetB4(include_top=False, weights='imagenet')
for layer in base_model.layers[:-5]:
    base_model.trainable = False
inputs = tf.keras.Input(shape=(224,224,3))
x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(325, activation='softmax')(x)
eff4_model = tf.keras.Model(inputs, outputs)
eff4_model.summary()
eff4_model.compile(
    loss = tf.keras.losses.categorical_crossentropy,
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics = ['accuracy'])
eff4_model.fit(train_data,
               epochs=20,
               steps_per_epoch = len(train_data),
               validation_data = val_data,
               validation_steps = len(val_data))
               epochs=20,
               steps_per_epoch = len(train_data),
               validation_data = val_data,
               validation_steps = len(val_data))

eff4_model_evaluation = eff4_model.evaluate(test_data)
print(f"Efficient Model Accuracy: {eff4_model_evaluation[1] * 100 : 0.2f}%")
test_labels = test_data.class_names
len(test_labels)
plt.figure(figsize=(16,16))
for image, label in test_data.take(1):
    model_prediction = eff4_model.predict(image)
    for i in range(18):
        plt.subplot(6,3,i+1)
        plt.imshow(image[i].numpy().astype('uint8'))
        plt.title(f"Model Prediction: {test_labels[tf.argmax(tf.round(model_prediction[i]))]}\nOriginal Labels: {test_labels[tf.argmax(label[i])]}")
        plt.axis("off")        
        
#---------------------------------------------------------
#Build EfficientNetB3 Model         
base_model = tf.keras.applications.EfficientNetB3(include_top=False, weights='imagenet')
for layer in base_model.layers[:-5]:
    base_model.trainable = False
inputs = tf.keras.Input(shape=(224,224,3))
x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(325, activation='softmax')(x)
eff3_model = tf.keras.Model(inputs, outputs)
eff3_model.summary()
eff3_model.compile(
    loss = tf.keras.losses.categorical_crossentropy,
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics = ['accuracy'])
eff3_model.fit(train_data,
               epochs=20,
               steps_per_epoch = len(train_data),
               validation_data = val_data,
               validation_steps = len(val_data))

eff3_model_evaluation = eff3_model.evaluate(test_data)
print(f"Efficient Model Accuracy: {eff3_model_evaluation[1] * 100 : 0.2f}%")
test_labels = test_data.class_names
len(test_labels)
plt.figure(figsize=(16,16))
for image, label in test_data.take(1):
    model_prediction = eff3_model.predict(image)
    for i in range(18):
        plt.subplot(6,3,i+1)
        plt.imshow(image[i].numpy().astype('uint8'))
        plt.title(f"Model Prediction: {test_labels[tf.argmax(tf.round(model_prediction[i]))]}\nOriginal Labels: {test_labels[tf.argmax(label[i])]}")
        plt.axis("off")         
        
#---------------------------------------------------------
#Build EfficientNetB2 Model         
base_model = tf.keras.applications.EfficientNetB2(include_top=False, weights='imagenet')
for layer in base_model.layers[:-5]:
    base_model.trainable = False
inputs = tf.keras.Input(shape=(224,224,3))
x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(325, activation='softmax')(x)
eff2_model = tf.keras.Model(inputs, outputs)
eff2_model.summary()
eff2_model.compile(
    loss = tf.keras.losses.categorical_crossentropy,
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics = ['accuracy'])
eff2_model.fit(train_data,
               epochs=20,
               steps_per_epoch = len(train_data),
               validation_data = val_data,
               validation_steps = len(val_data))

eff2_model_evaluation = eff2_model.evaluate(test_data)
print(f"Efficient Model Accuracy: {eff2_model_evaluation[1] * 100 : 0.2f}%")
test_labels = test_data.class_names
len(test_labels)
plt.figure(figsize=(16,16))
for image, label in test_data.take(1):
    model_prediction = eff2_model.predict(image)
    for i in range(18):
        plt.subplot(6,3,i+1)
        plt.imshow(image[i].numpy().astype('uint8'))
        plt.title(f"Model Prediction: {test_labels[tf.argmax(tf.round(model_prediction[i]))]}\nOriginal Labels: {test_labels[tf.argmax(label[i])]}")
        plt.axis("off")            
#---------------------------------------------------------
#Build EfficientNetB1 Model         
base_model = tf.keras.applications.EfficientNetB1(include_top=False, weights='imagenet')
for layer in base_model.layers[:-5]:
    base_model.trainable = False
inputs = tf.keras.Input(shape=(224,224,3))
x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(325, activation='softmax')(x)
eff1_model = tf.keras.Model(inputs, outputs)
eff1_model.summary()
eff1_model.compile(
    loss = tf.keras.losses.categorical_crossentropy,
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics = ['accuracy'])
eff1_model.fit(train_data,
               epochs=20,
               steps_per_epoch = len(train_data),
               validation_data = val_data,
               validation_steps = len(val_data))

eff1_model_evaluation = eff1_model.evaluate(test_data)
print(f"Efficient Model Accuracy: {eff1_model_evaluation[1] * 100 : 0.2f}%")
test_labels = test_data.class_names
len(test_labels)
plt.figure(figsize=(16,16))
for image, label in test_data.take(1):
    model_prediction = eff1_model.predict(image)
    for i in range(18):
        plt.subplot(6,3,i+1)
        plt.imshow(image[i].numpy().astype('uint8'))
        plt.title(f"Model Prediction: {test_labels[tf.argmax(tf.round(model_prediction[i]))]}\nOriginal Labels: {test_labels[tf.argmax(label[i])]}")
        plt.axis("off")         
        
#---------------------------------------------------------
#Build EfficientNetB0 Model         
base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet')
for layer in base_model.layers[:-5]:
    base_model.trainable = False
inputs = tf.keras.Input(shape=(224,224,3))
x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(325, activation='softmax')(x)
eff0_model = tf.keras.Model(inputs, outputs)
eff0_model.summary()
eff0_model.compile(
    loss = tf.keras.losses.categorical_crossentropy,
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics = ['accuracy'])
eff0_model.fit(train_data,
               epochs=20,
               steps_per_epoch = len(train_data),
               validation_data = val_data,
               validation_steps = len(val_data))

eff0_model_evaluation = eff0_model.evaluate(test_data)
print(f"Efficient Model Accuracy: {eff0_model_evaluation[1] * 100 : 0.2f}%")
test_labels = test_data.class_names
len(test_labels)
plt.figure(figsize=(16,16))
for image, label in test_data.take(1):
    model_prediction = eff0_model.predict(image)
    for i in range(18):
        plt.subplot(6,3,i+1)
        plt.imshow(image[i].numpy().astype('uint8'))
        plt.title(f"Model Prediction: {test_labels[tf.argmax(tf.round(model_prediction[i]))]}\nOriginal Labels: {test_labels[tf.argmax(label[i])]}")
        plt.axis("off")        
#---------------------------------------------------------
#Build VGG16_API Model  

set_trainable = False
base_model1 = tf.keras.applications.VGG16(include_top=False, weights='imagenet',input_shape=(224, 224, 3))
for layer in base_model1.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
base_model.Sequential()
base_model.add(base_model1)        
base_model.add(layers.Dense(2048, activation='relu'))
base_model.add(layers.Dense(2048, activation='relu'))
base_model.add(layers.Dense(3, activation='softmax'))
inputs = tf.keras.Input(shape=(224,224,3))
x = base_model1(inputs)
x =layers.Flatten()(x)
x = layers.Dense(2048, activation='relu')(x)
x = layers.Dense(2048, activation='relu')(x)
outputs = layers.Dense(325, activation='softmax')(x)
eff_model = tf.keras.Model(inputs, outputs)
eff_model.summary()
eff_model.compile(
    loss = tf.keras.losses.categorical_crossentropy,
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics = ['accuracy'])
eff_model.fit(train_data,
               epochs=20,
               steps_per_epoch = len(train_data),
               validation_data = val_data,
               validation_steps = len(val_data))
eff_model_evaluation = eff_model.evaluate(test_data)
print(f"Efficient Model Accuracy: {eff_model_evaluation[1] * 100 : 0.2f}%")
test_labels = test_data.class_names
len(test_labels)
plt.figure(figsize=(16,16))
for image, label in test_data.take(1):
    model_prediction = eff_model.predict(image)
    for i in range(18):
        plt.subplot(6,3,i+1)
        plt.imshow(image[i].numpy().astype('uint8'))
        plt.title(f"Model Prediction: {test_labels[tf.argmax(tf.round(model_prediction[i]))]}\nOriginal Labels: {test_labels[tf.argmax(label[i])]}")
        plt.axis("off")


#---------------------------------------------------------
#Build VGG16_KEYBOARD Model  

input_shape = (224, 224, 3)
vgg16_model = Sequential([
    Conv2D(64, (3, 3), input_shape=input_shape, padding='same',
           activation='relu'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dense(4096, activation='relu'),
    Dense(325, activation='softmax')])
vgg16_model.summary()
vgg16_model.compile(
    loss = tf.keras.losses.categorical_crossentropy,
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics = ['accuracy'])
vgg16_model.fit(train_data,
               epochs=20,
               steps_per_epoch = len(train_data),
               validation_data = val_data,
               validation_steps = len(val_data))
vgg16_model_evaluation = vgg16_model.evaluate(test_data)
print(f"Efficient Model Accuracy: {vgg16_model_evaluation * 100 : 0.2f}%")
test_labels = test_data.class_names
len(test_labels)
plt.figure(figsize=(16,16))
for image, label in test_data.take(1):
    model_prediction = vgg16_model.predict(image)
    for i in range(18):
        plt.subplot(6,3,i+1)
        plt.imshow(image[i].numpy().astype('uint8'))
        plt.title(f"Model Prediction: {test_labels[tf.argmax(tf.round(model_prediction[i]))]}\nOriginal Labels: {test_labels[tf.argmax(label[i])]}")
        plt.axis("off")

#---------------------------------------------------------
#Build VGG19_API Model  
set_trainable = False
base_model1 = tf.keras.applications.VGG19(include_top=False, weights='imagenet',input_shape=(224, 224, 3))
for layer in base_model1.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
base_model.Sequential()
base_model.add(base_model1)        
base_model.add(layers.Dense(2048, activation='relu'))
base_model.add(layers.Dense(2048, activation='relu'))
base_model.add(layers.Dense(3, activation='softmax'))
inputs = tf.keras.Input(shape=(224,224,3))
x = base_model1(inputs)
x =layers.Flatten()(x)
x = layers.Dense(2048, activation='relu')(x)
x = layers.Dense(2048, activation='relu')(x)
outputs = layers.Dense(325, activation='softmax')(x)
eff_model = tf.keras.Model(inputs, outputs)
eff_model.summary()
eff_model.compile(
    loss = tf.keras.losses.categorical_crossentropy,
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics = ['accuracy'])
eff_model.fit(train_data,
               epochs=20,
               steps_per_epoch = len(train_data),
               validation_data = val_data,
               validation_steps = len(val_data))
eff_model_evaluation = eff_model.evaluate(test_data)
print(f"Efficient Model Accuracy: {eff_model_evaluation[1] * 100 : 0.2f}%")
test_labels = test_data.class_names
len(test_labels)
plt.figure(figsize=(16,16))
for image, label in test_data.take(1):
    model_prediction = eff_model.predict(image)
    for i in range(18):
        plt.subplot(6,3,i+1)
        plt.imshow(image[i].numpy().astype('uint8'))
        plt.title(f"Model Prediction: {test_labels[tf.argmax(tf.round(model_prediction[i]))]}\nOriginal Labels: {test_labels[tf.argmax(label[i])]}")
        plt.axis("off")
#---------------------------------------------------------
#Build VGG16_KEYBOARD Model  
input_shape = (224, 224, 3)
vgg19_model = Sequential([
    Conv2D(64, (3, 3), input_shape=input_shape, padding='same',
           activation='relu'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dense(4096, activation='relu'),
    Dense(325, activation='softmax')])
vgg19_model.summary()
vgg19_model.compile(
    loss = tf.keras.losses.categorical_crossentropy,
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics = ['accuracy'])
vgg19_model.fit(train_data,
               epochs=20,
               steps_per_epoch = len(train_data),
               validation_data = val_data,
               validation_steps = len(val_data))
vgg19_model_evaluation = vgg19_model.evaluate(test_data)
print(f"Efficient Model Accuracy: {vgg19_model_evaluation * 100 : 0.2f}%")
test_labels = test_data.class_names
len(test_labels)
plt.figure(figsize=(16,16))
for image, label in test_data.take(1):
    model_prediction = vgg19_model.predict(image)
    for i in range(18):
        plt.subplot(6,3,i+1)
        plt.imshow(image[i].numpy().astype('uint8'))
        plt.title(f"Model Prediction: {test_labels[tf.argmax(tf.round(model_prediction[i]))]}\nOriginal Labels: {test_labels[tf.argmax(label[i])]}")
        plt.axis("off")







