import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np

# Function to create the model
def create_model():
    batch_size = 32
    image_size = (64, 64)
    NUM_CLASSES = 5

    model = tf.keras.Sequential([
        layers.Resizing(image_size[0], image_size[1]),
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(0.2, 0.2),
        layers.RandomZoom(0.2),
        layers.Conv2D(32, (3, 3), input_shape=(image_size[0], image_size[1], 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Function to train the model
def train_model(model, train_ds, validation_ds, epochs):
    
    for epoch in range(epochs):
        progress_text = st.sidebar.empty() 
        # Your existing training code...
        history = model.fit(train_ds, validation_data=validation_ds, epochs=epochs)

        # Callback to update the progress in the Streamlit sidebar
        progress_text.text(f"Training epoch: {epoch + 1}/{epochs}")

    return history

# Function to visualize EDA and sample images
def visualize_eda(train_ds):
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for images, labels in train_ds.take(1):
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i].numpy().astype("uint8"))
            ax.axis("off")

    # Instead of plt.show(), use st.pyplot(fig) to display the figure in Streamlit
    st.pyplot(fig)

# Function to plot training history
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the loss curves on the first subplot
    ax1.plot(history.history['loss'], label='training loss')
    ax1.plot(history.history['val_loss'], label='validation loss')
    ax1.set_title('Loss curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot the accuracy curves on the second subplot
    ax2.plot(history.history['accuracy'], label='training accuracy')
    ax2.plot(history.history['val_accuracy'], label='validation accuracy')
    ax2.set_title('Accuracy curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # Adjust the spacing between subplots
    fig.tight_layout()

    # Display the Matplotlib figure in Streamlit
    st.pyplot(fig)

# Streamlit app
def main():
    st.title("Image Classification Streamlit App")

    # Load and preprocess data
    batch_size = 32
    image_size = (64, 64)
    validation_split = 0.2

    train_ds = image_dataset_from_directory(
        directory='dataset/train',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset='training',
        seed=123,
    )

    validation_ds = image_dataset_from_directory(
        directory='dataset/train',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset='validation',
        seed=123,
    )

    # Sidebar controls
    st.sidebar.header("Model Training Controls")
    epochs = st.sidebar.slider("Number of Epochs", 1, 25, 13)
    train_button = st.sidebar.button("Train Model")

    # Create and train the model when the button is clicked
    if train_button:
        st.sidebar.text("Training in progress...")
        model = create_model()
        history = train_model(model, train_ds, validation_ds, epochs)
        
        # Visualize EDA and sample images
        visualize_eda(train_ds)

        # Plot training history
        plot_history(history)

        st.sidebar.text("Training complete!")

        test_ds = image_dataset_from_directory(
            directory='dataset/test',
            labels='inferred',
            label_mode='categorical',
            batch_size=batch_size,
            image_size=image_size,
        )
    
        test_loss, test_acc = model.evaluate(test_ds)
        st.write(f'Test Accuracy: {test_acc:.2%}')

if __name__ == "__main__":
    main()