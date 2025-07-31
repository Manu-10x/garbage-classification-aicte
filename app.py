# %%
#Importing necessary libraries for the project
import numpy as np  # Importing NumPy for numerical operations and array manipulations
import matplotlib.pyplot as plt  # Importing Matplotlib for plotting graphs and visualizations
import seaborn as sns  # Importing Seaborn for statistical data visualization, built on top of Matplotlib
import tensorflow as tf  # Importing TensorFlow for building and training machine learning models
from tensorflow import keras  # Importing Keras, a high-level API for TensorFlow, to simplify model building
from tensorflow.keras import Layer  # Importing Layer class for creating custom layers in Keras
from tensorflow.keras.models import Sequential  # Importing Sequential model for building neural networks layer-by-layer
from tensorflow.keras.layers import Rescaling , GlobalAveragePooling2D
from tensorflow.keras import layers, optimizers, callbacks  # Importing various modules for layers, optimizers, and callbacks in Keras
from sklearn.utils.class_weight import compute_class_weight  # Importing function to compute class weights for imbalanced datasets
from tensorflow.keras.applications import EfficientNetV2B2  # Importing EfficientNetV2S model for transfer learning
from sklearn.metrics import confusion_matrix, classification_report  # Importing functions to evaluate model performance
import gradio as gr


# %%
from google.colab import files
uploaded = files.upload()  # Upload your dataset zip (e.g., trash_dataset.zip)
unzip archive\ \(3\).zip -d dataset/

# %%
import os
from PIL import Image
from IPython.display import display, Markdown

# Path to the dataset directory
dataset_dir = 'dataset/TrashType_Image_Dataset'

# Check if the directory exists
if not os.path.isdir(dataset_dir):
    print(f"Error: Directory '{dataset_dir}' not found. Please make sure you have unzipped the dataset correctly.")
else:
    # Get the class names from the subdirectories
    class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])

    if not class_names:
        print(f"No class subdirectories found in '{dataset_dir}'.")
    else:
        # Create a markdown table header
        markdown_table = "| Class Name | Number of Images | Image Format | Dimensions |\n"
        markdown_table += "|---|---|---|---|\n"

        # Populate the table with information for each class
        for name in class_names:
            class_path = os.path.join(dataset_dir, name)
            image_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            num_images = len(image_files)

            # Get image format and size from the first image
            if num_images > 0:
                first_image_path = os.path.join(class_path, image_files[0])
                try:
                    with Image.open(first_image_path) as img:
                        image_format = img.format
                        image_size = f"{img.width}x{img.height}"
                except Exception:
                    image_format = "N/A"
                    image_size = "N/A"
            else:
                image_format = "N/A"
                image_size = "N/A"

            markdown_table += f"| {name} | {num_images} | {image_format} | {image_size} |\n"

        # Display the table in the notebook
        display(Markdown(markdown_table))

# %%
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/TrashType_Image_Dataset",   # ✅ updated path
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(124, 124),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/TrashType_Image_Dataset",   # ✅ updated path
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(124, 124),
    batch_size=32
)



# %%
val_class=val_ds.class_names
print(val_class)

# %%
val_batches=tf.data.experimental.cardinality(val_ds)
print(val_batches)



# %%
test_ds=val_ds.take(val_batches //2)
val_dat=val_ds.skip(val_batches //2)
test_ds_eval=test_ds.cache().prefetch(tf.data.AUTOTUNE)
print(train_ds.class_names)
print(val_class)
print(len(train_ds.class_names))


# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
for images, labels in train_ds.take(1):
    for i in range(12):
        ax = plt.subplot(4, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(train_ds.class_names[labels[i]])
        plt.axis("off")

# %%
def count_distribution(dataset, class_names):
    total = 0
    counts = {name: 0 for name in class_names}

    for _, labels in dataset:
        for label in labels.numpy():
            class_name = class_names[label]
            counts[class_name] += 1
            total += 1

    for k in counts:
        counts[k] = round((counts[k] / total) * 100, 2)  # Convert to percentage
    return counts

# %%
def simple_bar_plot(dist, title):
    plt.bar(dist.keys(), dist.values(), color='cornflowerblue')
    plt.title(title)
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()

# %%
# Ensure datasets are defined
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/TrashType_Image_Dataset",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(124, 124),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/TrashType_Image_Dataset",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(124, 124),
    batch_size=32
)

# Recalculate val_batches and test_ds after re-creating val_ds
val_batches = tf.data.experimental.cardinality(val_ds)
test_ds_split = val_ds.take(val_batches // 2)
val_dat = val_ds.skip(val_batches // 2)

class_names=train_ds.class_names
train_dist=count_distribution(train_ds,class_names)
val_dist=count_distribution(val_ds,class_names)
test_dist = count_distribution(test_ds_split, class_names)

overall_dist={}
for k in class_names:
  overall_dist[k] = round((train_dist[k] + val_dist[k]) / 2, 2)

print("Training Distribution:", train_dist)
print("Validation Distribution:", val_dist)
print("Test Distribution:", test_dist)
print("Overall Distribution:", overall_dist)

# %%
class_counts={i:0 for i in range(len(class_names))}
all_labels=[]
for images,labels in train_ds:
  for label in labels.numpy():
    class_counts[label]+=1
    all_labels.append(label)

class_weights_array=compute_class_weight(class_weight='balanced',classes=np.arange(len(class_names)),y=all_labels)
class_weights={i: w for i,w in enumerate(class_weights_array)}
print(class_counts)
print(class_weights)

# %%
data_augmentation=Sequential([layers.RandomFlip("horizontal"),layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),])


# %% [markdown]
# 

# %%
base_model=EfficientNetV2B2(include_top=False,weights='imagenet',input_shape=(180,180,3),include_preprocessing=False)
base_model.trainable=True
for layer in base_model.layers[:100]:
  layer.trainable=False

# %%
model=Sequential([layers.Input(shape=(124,124,3)),data_augmentation,layers.Resizing(180, 180),base_model,GlobalAveragePooling2D(),layers.Dropout(0.2),layers.Dense(len(class_names),activation='softmax')])

# %%
model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# %%
early=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)
epochs=15
history=model.fit(train_ds,validation_data=val_ds,epochs=epochs,class_weight=class_weights,batch_size=32,callbacks=[early])

# %%
model.summary()

# %%
base_model.summary()

# %%
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs_range=range(len(acc))


# %%
plt.figure(figsize=(10,8))
plt.subplot(1,2,1)
plt.plot(epochs_range,acc,label='training accuracy')
plt.plot(epochs_range,val_acc,label='validation accuracy')
plt.legend(loc='lower right')
plt.title("Training vs Validation Loss")

plt.subplot(1,2,2)

plt.plot(epochs_range,loss, label='training Loss')
plt.plot(epochs_range,val_loss, label='validation loss')
plt.legend(loc='upper right')
plt.title("Training vs Validation Loss")
plt.show() #display the graphs

# %%
loss,accuracy=model.evaluate(test_ds_eval)
print(f'Test Accuracy :{accuracy:.4f}, Test Loss :{loss:.4f}')

# %%
# Extract true labels from all batches in the test dataset
y_true=np.concatenate([y.numpy() for x,y in test_ds_eval],axis=0) #Convert Tensor labels to NumPy array and concatenate them
#get predictions as probabilities from the model
y_pred_probs=model.predict(test_ds_eval)#predict class probabilities for each sample in dataset
#convert probabilities to predicted class indices
y_pred=np.argmax(y_pred_probs,axis=1)#select the class with the highest probability for each sample
cm = confusion_matrix(y_true, y_pred)
print(cm)  # Display confusion matrix
print(classification_report(y_true, y_pred))  # Print precision, recall, and F1-score for each class


# %%
#plotting the confusion matrix to visualize model performance
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('true')
plt.title('confusion matrix')
plt.show() #display the plot

# %%
class_names=train_ds.class_names #extract class names
for images,labels in test_ds_eval.take(1):
  predictions=model.predict(images)
  pred_labels=tf.argmax(predictions,axis=1)
  for i in range(8):
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(f"True:{class_names[labels[i]]},pred:{class_names[pred_labels[i]]}")
    plt.axis("off")
    plt.show()

# %%
model.save('Effiicientnetv2b2.keras') # saving the model

# %%
from google.colab import files
files.download('Effiicientnetv2b2.keras') #downloading the model file

# %%
model = tf.keras.models.load_model('Effiicientnetv2b2.keras') #load the model

# %%
print(accuracy)

# %%
!pip install gradio

# %%
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
def classify_image(img):
    # Resize image to 124x124 pixels (Note: Comment says 128x128, but code resizes to 124x124)
    img = img.resize((124, 124))

    # Convert image to a NumPy array with float32 dtype
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)

    # Expand dimensions to match model input shape (adds a batch dimension)
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction using the trained model
    prediction = model.predict(img_array)

    # Get the index of the highest predicted probability
    predicted_class_index = np.argmax(prediction)

    # Map the predicted index to its corresponding class name
    predicted_class_name = class_names[predicted_class_index]

    # Extract confidence score (probability of the predicted class)
    confidence = prediction[0][predicted_class_index]

    # Return formatted prediction result with confidence score
    return f"Predicted: {predicted_class_name} (Confidence: {confidence:.2f})"


# %%
import gradio as gr
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# A function to classify the image
def classify_image(img):
    # Resize image to 124x124 pixels
    img = img.resize((124, 124))

    # Convert image to a NumPy array with float32 dtype
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)

    # Expand dimensions to match model input shape (adds a batch dimension)
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction using the trained model
    prediction = model.predict(img_array)

    # Get the index of the highest predicted probability
    predicted_class_index = np.argmax(prediction)

    # Map the predicted index to its corresponding class name
    predicted_class_name = class_names[predicted_class_index]

    # Extract confidence score (probability of the predicted class)
    confidence = prediction[0][predicted_class_index]

    # Return formatted prediction result with confidence score
    return f"Predicted: {predicted_class_name} (Confidence: {confidence:.2f})"

# Create the Gradio interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload an image of trash"),
    outputs="text",
    title="Trash Classifier",
    description="This app classifies images of trash into one of six categories: cardboard, glass, metal, paper, plastic, or trash.",
    examples=[
        ["dataset/TrashType_Image_Dataset/cardboard/cardboard_001.jpg"],
        ["dataset/TrashType_Image_Dataset/glass/glass_001.jpg"],
        ["dataset/TrashType_Image_Dataset/metal/metal_001.jpg"],
    ],
    theme="huggingface"
)

# Launch the interface
iface.launch(share=True)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Create a bar plot of the class distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=list(image_counts.keys()), y=list(image_counts.values()))
plt.xlabel('Class Name')
plt.ylabel('Number of Images')
plt.title('Class Distribution of TrashNet Dataset')
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# ### Step 1: Create a Hugging Face Account
# 
# If you don't have one already, go to [huggingface.co/join](https://huggingface.co/join) and create a free account.
# 
# ### Step 2: Get Your Hugging Face API Token
# 
# You'll need a token with "write" permissions to upload your model.
# 1.  Go to your Hugging Face profile settings.
# 2.  Click on "Access Tokens" in the left menu.
# 3.  Click "New token", give it a name (e.g., "Colab Notebook"), and select the **write** role.
# 4.  Copy the generated token. You'll need it in the next step.

# %%
# Install the library needed to interact with the Hugging Face Hub
!pip install huggingface_hub -q

from huggingface_hub import HfApi, HfFolder, create_repo, notebook_login

# Login to Hugging Face
# You will be prompted to paste your token here.
notebook_login()

# --- IMPORTANT ---
# Define your Hugging Face username and a name for your new model repository.
# Replace "YOUR_USERNAME" with your actual Hugging Face username.
HF_USERNAME = "YOUR_USERNAME"
REPO_NAME = "trash-classifier-efficientnet"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

print(f"Your repository ID is: {REPO_ID}")

# Create the model repository on the Hub
try:
    create_repo(repo_id=REPO_ID, exist_ok=True)
    print(f"Repository '{REPO_ID}' created or already exists.")
except Exception as e:
    print(f"Error creating repository: {e}")

# Save your trained model to a file
model.save('trash_classifier_model.keras')

# Upload the model file to your new repository
api = HfApi()
try:
    api.upload_file(
        path_or_fileobj="trash_classifier_model.keras",
        path_in_repo="trash_classifier_model.keras", # The name of the file in the repo
        repo_id=REPO_ID,
        repo_type="model"
    )
    print(f"Model uploaded successfully to: https://huggingface.co/{REPO_ID}")
except Exception as e:
    print(f"Error uploading model: {e}")



# %% [markdown]
# ### Step 4: Create the Application Files
# 
# For deployment, you need two text files: `app.py` (your Gradio code) and `requirements.txt` (the libraries your app needs). You don't need to create these in Colab, but you will copy and paste the content below into your Hugging Face Space.
# 
# ---
# 
# **1. Content for `app.py`**
# 
# This code will run your Gradio app. It loads the model you just uploaded.
# 
# ```python
# import gradio as gr
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# from huggingface_hub import from_pretrained_keras
# 
# # --- IMPORTANT ---
# # Make sure this REPO_ID matches the one you created in Step 3
# REPO_ID = "YOUR_USERNAME/trash-classifier-efficientnet"
# 
# # Load the model from the Hugging Face Hub
# try:
#     model = from_pretrained_keras(REPO_ID, custom_objects={"TFSMLayer": tf.keras.layers.TFSMLayer})
#     print("Model loaded successfully.")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     model = None
# 
# class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
# 
# def classify_image(img):
#     if model is None:
#         return {"Error": "Model could not be loaded."}
#     
#     # Preprocess the image
#     img = img.resize((124, 124))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
#     img_array = np.expand_dims(img_array, axis=0)
#     
#     # Make prediction
#     prediction = model.predict(img_array)[0]
#     
#     # Format the output
#     confidences = {class_names[i]: float(prediction[i]) for i in range(len(class_names))}
#     return confidences
# 
# # Create the Gradio interface with gr.Blocks
# with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal")) as iface:
#     gr.Markdown("# ♻️ Trash Classifier App")
#     gr.Markdown("Upload an image of trash, and the model will classify it into one of six categories.")
# 
#     with gr.Row(variant="panel"):
#         with gr.Column(scale=1):
#             image_input = gr.Image(type="pil", label="Upload Image")
#             submit_button = gr.Button("Classify Image", variant="primary")
#         with gr.Column(scale=1):
#             gr.Markdown("## Prediction Results")
#             output_label = gr.Label(num_top_classes=3, label="Classification")
# 
#     submit_button.click(
#         fn=classify_image,
#         inputs=image_input,
#         outputs=output_label
#     )
#     
#     gr.Markdown("---")
#     gr.Markdown("### Example Images")
#     gr.Examples(
#         examples=[
#             "https://i.imgur.com/3qVjWd1.jpeg", # Cardboard
#             "https://i.imgur.com/3aV0T5A.jpeg", # Glass
#             "https://i.imgur.com/fplxTjD.jpeg"  # Metal
#         ],
#         inputs=image_input
#     )
# 
# iface.launch()
# ```
# 
# ---
# 
# **2. Content for `requirements.txt`**
# 
# This file lists the required Python libraries.
# 
# ```
# tensorflow
# gradio
# numpy
# Pillow
# huggingface_hub
# ```
# ---
# ### Step 5: Create and Deploy on Hugging Face Spaces
# 
# 1.  Go to [huggingface.co/new-space](https://huggingface.co/new-space).
# 2.  Give your Space a **name** (e.g., `trash-classifier`).
# 3.  Select a **license**, such as `MIT`.
# 4.  For the **Space SDK**, select **Gradio**.
# 5.  Choose **Public** visibility.
# 6.  Click **Create Space**.
# 7.  You'll be taken to your new Space. Click the **Files** tab.
# 8.  Click **Add file** and select **Create new file**.
# 9.  Name the file `requirements.txt`. Copy the text from `requirements.txt` above, paste it into the editor, and click **Commit new file**.
# 10. Click **Add file** again and **Create new file**.
# 11. Name this file `app.py`. Copy the Python code for `app.py` from above.
# 12. **Crucially, remember to replace `YOUR_USERNAME/trash-classifier-efficientnet` with your actual model repository ID in the `app.py` code.**
# 13. Click **Commit new file**.
# 
# That's it! Hugging Face will automatically build your application. After a minute or two, your permanent Gradio app will be live under the "App" tab. You can now share the URL of your Space with anyone.


