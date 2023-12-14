import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.preprocessing import image
import io

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
        history = model.fit(train_ds, 
                            validation_data=validation_ds, 
                            steps_per_epoch = len(train_ds),
                            epochs=epochs
                            )

        # Callback to update the progress in the Streamlit sidebar
        progress_text.text(f"Training epoch: {epoch + 1}/{epochs}")

    
    return history

def get_class_counts(dataset):
    class_counts = dict()
    class_labels = ['Beaches', 'Oceans', 'Forests', 'Sunsets', 'Architecture']
    for _, labels in dataset:
        class_indices = np.argmax(labels, axis=1)

        for class_index in class_indices:
            class_name = class_labels[class_index]  # Replace with your class names
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    return class_counts

# Function to visualize EDA and sample images
def visualize_eda(train_ds):
    st.subheader("EDA")
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for images, labels in train_ds.take(1):
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i].numpy().astype("uint8"))
            ax.axis("off")

    # Instead of plt.show(), use st.pyplot(fig) to display the figure in Streamlit
    st.pyplot(fig)

    # Display class names and counts
    class_counts = get_class_counts(train_ds)
    st.subheader("Class Counts in Training Dataset:")
    for class_name, count in class_counts.items():
        st.write(f"Class: {class_name}, Count: {count}")

# Function to plot training history
def plot_history(history):
    st.subheader("Loss/accuracy curve")
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

def display_confusion_matrix(model, test_ds):
    true_labels = np.concatenate([y for x, y in test_ds], axis=0)
    predicted_probs = model.predict(test_ds)
    predicted_labels = np.argmax(predicted_probs, axis=1)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(np.argmax(true_labels, axis=1), predicted_labels)

    # Display confusion matrix using seaborn heatmap with colors
    st.subheader("Confusion Matrix:")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    st.pyplot(fig)

def image_upload_predict(model, file):

    if file is not None:
        # Read the uploaded image
        image_content = file.read()
        image_array = image.img_to_array(image.load_img(io.BytesIO(image_content), target_size=(64, 64)))
        image_array = np.expand_dims(image_array, axis=0)

        # Make predictions using your model
        result = model.predict(image_array)

        # Assuming you have defined NUM_CLASSES somewhere in your code
        NUM_CLASSES = 5  # Replace with the actual number of classes

        # Extract the predicted class index
        predicted_class = np.argmax(result, axis=1)

        # Map the predicted class index to the corresponding label/category
        class_labels = ['Beaches', 'Oceans', 'Forests', 'Sunsets', 'Architecture']

        # Ensure the predicted_class is within the valid range
        if 0 <= predicted_class < NUM_CLASSES:
            prediction = class_labels[predicted_class[0]]  # Extract the scalar value from the array
            st.write("Prediction:", prediction)
        else:
            st.write("Invalid predicted class index.")

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
    file = st.sidebar.file_uploader("Choose an image...", type="jpg")

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
        display_confusion_matrix(model, test_ds)

        test_loss, test_acc = model.evaluate(test_ds)
        st.write(f'Test Accuracy: {test_acc:.2%}')

        image_upload_predict(model, file)

if __name__ == "__main__":
    main()

