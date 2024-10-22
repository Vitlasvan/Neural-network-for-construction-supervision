import os
import cv2
import numpy as np
import tensorflow as tf
import requests  #Для отправки данных на Flask-сервер
import tensorflow.keras.preprocessing.image as tf_image
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
import json  # Для загрузки JSON с координатами

1. #Путь к JSON-файлу с координатами 
coordinates_file = 'D:/neural network for construction supervision/data/inference/coordinates.json'

1.1 #Проверяем наличие файла с координатами
if not os.path.exists(coordinates_file):
    raise FileNotFoundError(f"JSON файл с координатами не найден: {coordinates_file}")

1.2 #Загрузка координат из JSON-файла
with open(coordinates_file, 'r') as f:
    coordinates = json.load(f)

2. #Функция для предобработки изображений
def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return img

def preprocess_images(input_dir, output_dir, target_size=(224, 224)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_file in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_file)
        img = preprocess_image(img_path, target_size)
        output_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_path, np.uint8(img * 255))

2.1 #Предобработка изображений для категорий
input_dir_defects = 'data/train/defects/'
output_dir_defects = 'data/preprocessed/defects/'
preprocess_images(input_dir_defects, output_dir_defects)

input_dir_normal = 'data/train/normal/'
output_dir_normal = 'data/preprocessed/normal/'
preprocess_images(input_dir_normal, output_dir_normal)

3.#Создание модели
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
base_model.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

4.#Аугментация данных
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'data/preprocessed',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'data/preprocessed',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

5.#Обучение модели
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[early_stopping]
)

6.#Построение графиков
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

7.#Сохранение модели
model.save('my_model.h5')

8.#Оценка модели
loss, accuracy = model.evaluate(validation_generator)
print(f'Test accuracy: {accuracy:.2f}')

8.1#Предсказания и отчёт о метриках
predictions = model.predict(validation_generator)
true_labels = validation_generator.classes  # Извлекаем истинные метки
predicted_classes = np.where(predictions > 0.5, 1, 0)  # Устанавливаем порог в 0.5
print(classification_report(true_labels, predicted_classes))

8.2#Сохранение и загрузка модели
model.save('construction_defect_model.keras')
loaded_model = tf.keras.models.load_model('construction_defect_model.keras')
loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

9.#Функция для отправки данных на Flask-сервер
def send_data_to_flask(result, coords, image_path):
    data = {
        "result": result,
        "x": coords["x"],
        "y": coords["y"],
        "z": coords["z"]
    }
    
    files = {
        'image': open(image_path, 'rb')
    }
    
    try:
        response = requests.post("http://127.0.0.1:5000/send_defect_data", data=data, files=files)
        if response.status_code == 200:
            print(f"Data successfully sent to Flask server: {response.json()}")
        else:
            print(f"Failed to send data to Flask server: {response.status_code}")
    except Exception as e:
        print(f"An error occurred while sending data to Flask server: {e}")

10.#Функция для обработки изображений и отправки данных на Flask-сервер
def process_drone_images(images_folder, coordinates):
    for img_file in os.listdir(images_folder):
        if img_file.endswith(".jpg") and img_file in coordinates:
            coords = coordinates[img_file]  # Получаем координаты из JSON
            img_path = os.path.join(images_folder, img_file)

            10.1#Предобработка изображений
            img = tf_image.load_img(img_path, target_size=(224, 224))
            img_array = tf_image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            10.2#Предсказание с помощью нейросети
            prediction = model.predict(img_array)

            10.3#Результат предсказания
            result = "Defect detected" if prediction[0] < 0.5 else "No defect detected"

            10.4#Отправка данных на Flask-сервер, если найден дефект (изображение и координаты)
            if result == "Defect detected":
                print(f"Defect detected in {img_file}. Sending to Flask server.")
                send_data_to_flask(result, coords, img_path)
            else:
                print(f"No defect detected in {img_file}. Not sending to Flask server.")

10.5#Пример использования функции для изображений и их координат (X, Y, Z)
drone_images_folder = 'data/inference/images'
process_drone_images(drone_images_folder, coordinates)

11.#Применение модели для предсказаний
def predict_images(folder_path, model):
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        try:
            img = tf_image.load_img(img_path, target_size=(224, 224))
            img_array = tf_image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)
            if prediction[0] > 0.5:
                print(f'{img_file}: No defect detected')
            else:
                print(f'{img_file}: Defect detected')
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")

predict_images('data/inference/images', loaded_model)

12.#Визуализация изображений и предсказаний
def plot_images_and_predictions(folder_path, model):
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        try:
            img = tf_image.load_img(img_path, target_size=(224, 224))
            img_array = tf_image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)
            plt.imshow(img)
            plt.axis('off')
            if prediction[0] > 0.5:
                plt.title(f'{img_file}: No defect detected')
            else:
                plt.title(f'{img_file}: Defect detected')
            plt.show()
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

plot_images_and_predictions('data/inference/images', loaded_model)
