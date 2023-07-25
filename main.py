import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def viz_pic():
    train_cats_dir='C:\Progetti\Personali\MachineLearning\ImageClassification\Coursera\HorsesvsHumans\data\\training\horses'
    train_cat_fnames=os.listdir(train_cats_dir)
    train_dogs_dir='C:\Progetti\Personali\MachineLearning\ImageClassification\Coursera\HorsesvsHumans\data\\training\humans'
    train_dog_fnames=os.listdir(train_dogs_dir)

    # Parameters for our graph; we'll output images in a 4x4 configuration
    nrows = 4
    ncols = 4

    pic_index = 0  # Index for iterating over images
    # Set up matplotlib fig, and size it to fit 4x4 pics
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)

    pic_index += 8

    next_cat_pix = [os.path.join(train_cats_dir, fname)
                    for fname in train_cat_fnames[pic_index - 8:pic_index]
                    ]

    next_dog_pix = [os.path.join(train_dogs_dir, fname)
                    for fname in train_dog_fnames[pic_index - 8:pic_index]
                    ]

    for i, img_path in enumerate(next_cat_pix + next_dog_pix):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off')  # Don't show axes (or gridlines)

        img = mpimg.imread(img_path)
        print(img.shape)
        plt.imshow(img)

    plt.show()

def create_generator():
    train_generator= tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1/255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    validation_generator=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

    train_datagen=train_generator.flow_from_directory(
        'C:\Progetti\Personali\MachineLearning\ImageClassification\Coursera\HorsesvsHumans\data\\training',
        batch_size=32,
        shuffle=True, target_size=(300, 300), class_mode='binary'
    )

    val_datagen=validation_generator.flow_from_directory(
        'C:\Progetti\Personali\MachineLearning\ImageClassification\Coursera\HorsesvsHumans\data\\validation',
        batch_size=16,
        shuffle=True, target_size=(300, 300), class_mode='binary'
    )

    return train_datagen,val_datagen

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy')>0.99:
            print(f'accuracy over 99% so end training')
            self.model.stop_training=True


def create_model():
    model=tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(32,(3,3),activation=tf.nn.relu, input_shape=(300,300,3)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation=tf.nn.relu),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
    return model


def create_model_transfer():
    base_model = tf.keras.applications.InceptionV3(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=(300, 300, 3),
        include_top=False)
    base_model.trainable = False
    #creating top layer
    # Choose `mixed_7` as the last layer of your base model
    last_layer = base_model.get_layer('mixed7')
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output
    x=tf.keras.layers.Dense(256,activation=tf.nn.relu)(last_output)

    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x=tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(56, activation=tf.nn.relu)(x)
    # A Dense classifier with a single unit (binary classification)
    outputs = tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)(x)
    model = tf.keras.Model(base_model.inputs,outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
    return model

def print_acc_loss(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()

    plt.show()
if __name__=='__main__':
    viz_pic()
    train_datagen, val_datagen=create_generator()
    callback=MyCallback()
    #model=create_model()
    model=create_model_transfer()
    history=model.fit(x=train_datagen,validation_data=val_datagen,epochs=10, callbacks=[callback])
    print_acc_loss(history)
