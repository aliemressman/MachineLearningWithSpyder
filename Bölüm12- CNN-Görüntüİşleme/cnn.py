from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Model oluşturma
classifier = Sequential()

# 1. Convolution + ReLU
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# 2. Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# 2. Convolution Katmanı
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# 3. Flattening
classifier.add(Flatten())

# 4. Fully Connected Layer (YSA)
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Model Derleme
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Veri Ön İşleme
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('veriler/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,  # Batch size artırıldı
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('veriler/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

# Modeli eğitme
classifier.fit(training_set,
               steps_per_epoch=len(training_set),
               epochs=10,  # Epoch sayısı artırıldı
               validation_data=test_set,
               validation_steps=len(test_set))

# Tahmin Yapma
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

test_set.reset()
pred = classifier.predict(test_set, verbose=1)

# Eşik belirleme (0.5'ten büyükse 1, küçükse 0)
pred = (pred > 0.5).astype(int)

# Test etiketlerini alma
test_labels = []
for i in range(len(test_set)):
    test_labels.extend(test_set[i][1])

# Dosya isimleri
dosyaisimleri = test_set.filenames

# Sonuçları DataFrame olarak saklama
sonuc = pd.DataFrame({'dosyaisimleri': dosyaisimleri, 'tahminler': pred.flatten(), 'test': test_labels})

# Confusion Matrix Hesaplama
cm = confusion_matrix(test_labels, pred)
print(cm)
