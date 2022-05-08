import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.layers import Input, Lambda ,Dense,Flatten , Dropout , GlobalAveragePooling2D
classifier_vgg16 = VGG16(input_shape= (64,64,3),include_top=False,weights='imagenet')
classifier_vgg16.summary()
for layer in classifier_vgg16.layers:
    layer.trainable = False
    main_model = classifier_vgg16.output
    main_model = GlobalAveragePooling2D()(main_model)
    main_model = Dense(1024, activation='relu')(main_model)
    main_model = Dense(1024, activation='relu')(main_model)
    main_model = Dense(512, activation='relu')(main_model)
    main_model = Dropout(0.5)(main_model)
    main_model = Dense(5, activation='softmax')(main_model)
    model = Model(inputs=classifier_vgg16.input, outputs=main_model)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    training_set = train_datagen.flow_from_directory('C:\Users\senda\PycharmProjects\pythonProject\data',
                                                     target_size=(64, 64),
                                                     batch_size=15,
                                                     class_mode='categorical')

    test_set = test_datagen.flow_from_directory('C:\Users\senda\PycharmProjects\pythonProject\data',
                                                target_size=(64, 64),
                                                batch_size=10,
                                                class_mode='categorical',
                                                shuffle=False)
    nb_train_samples = 1190
    nb_validation_samples = 170
    batch_size = 5

    history = model.fit_generator(training_set,
                                  validation_data=test_set,
                                  epochs=5,
                                  steps_per_epoch=nb_train_samples // batch_size,
                                  validation_steps=nb_validation_samples // batch_size)
    nb_train_samples = 1190
    nb_validation_samples = 170
    batch_size = 5

    history = model.fit_generator(training_set,
                                  validation_data=test_set,
                                  epochs=5,
                                  steps_per_epoch=nb_train_samples // batch_size,
                                  validation_steps=nb_validation_samples // batch_size)
    import matplotlib.pyplot as plt

    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.show()
    import matplotlib.pyplot as plt

    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.show()
    from keras.models import load_model

