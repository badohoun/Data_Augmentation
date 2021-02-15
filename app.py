from PIL import Image

import streamlit as st
import matplotlib.pyplot as plt #For plotting our visualizations
from tensorflow.keras.preprocessing.image import ImageDataGenerator #Keras dataset generator class.
from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


from PIL import Image





icon = Image.open("Defend_Intelligence.png")

st.set_page_config(
   page_title="Image processing with ImageDataGenerator in tensorflow",
   page_icon = icon,
   layout = "centered",
   initial_sidebar_state ="auto"

)


image = Image.open("Augmented_images.png")

st.image(image)



audio_file = open("Rue Pierre Curie 89.m4a", "rb")
audio_bytes = audio_file.read()

image = Image.open("dog.jpg")
fig = plt.figure()

option = st.selectbox(
"Select an Option",
[
  "HomePage" ,
  "Show Image",
  "Rotate Image",
  "Vertical schift",
  "Horizontal schift",
  "Color spaces",
  "Zooming",
  "Vertical flipping",
  "Horizontal flipping"


]
)

if option == "HomePage":
    st.header("Image processing with ImageDataGenerator in tensorflow")
    """

    #
     Please select a page
    """



    st.image(image)
    st.audio(audio_bytes)

    st.balloons()
if option == "Show Initial Image":
    image = Image.open("dog.jpg")
    plt.imshow(image)
    plt.axis("off")
    st.pyplot(fig)
elif option == "Rotate Image":
    image = Image.open('dog.jpg')
    data = img_to_array(image)
    samples = expand_dims(data , 0)
    data_generated = ImageDataGenerator(rotation_range = 90)
    rotated_image = data_generated.flow(samples , batch_size = 1)
    for i in range(1):
        plt.subplot(330 + 1 + i)
        batch = rotated_image.next()
        result = batch[0].astype('uint8')
    plt.imshow(result)

    plt.axis("off")
    st.pyplot(fig)
elif option == "Vertical schift":
    image = Image.open('dog.jpg')
    data = img_to_array(image)
    samples = expand_dims(data,0)
    data_generated = ImageDataGenerator(height_shift_range = 0.3)
    vertical_schift = data_generated.flow(samples , batch_size = 1)
    for i in range(1):
        plt.subplot(330+1+i)
        batch = vertical_schift.next()
        result = batch[0].astype('uint8')
    plt.imshow(result)
    plt.axis("off")
    st.pyplot(fig)
elif  option == "Horizontal schift":
    image = Image.open('dog.jpg')
    data = img_to_array(image)
    samples = expand_dims(data,0)
    data_generated = ImageDataGenerator(width_shift_range = 0.3)
    horizontal_schift = data_generated.flow(samples , batch_size = 1)
    for i in range(1):
        plt.subplot(330+1+i)
        batch = horizontal_schift.next()
        result = batch[0].astype('uint8')
    plt.imshow(result)
    plt.axis("off")
    st.pyplot(fig)
elif option == "Color spaces":
    image = Image.open('dog.jpg')
    data = img_to_array(image)
    samples = expand_dims(data,0)
    data_generated = ImageDataGenerator(brightness_range = [0.2 , 1.0])
    color_spaces = data_generated.flow(samples , batch_size = 1)
    for i in range(1):
        plt.subplot(330+1+i)
        batch = color_spaces.next()
        result = batch[0].astype('uint8')
    plt.imshow(result)
    plt.axis("off")
    st.pyplot(fig)
elif option == "Zooming":
    image = Image.open('dog.jpg')
    data = img_to_array(image)
    samples = expand_dims(data,0)
    data_generated = ImageDataGenerator(zoom_range = [0.2 , 1.0])
    zooming = data_generated.flow(samples , batch_size = 1)
    for i in range(1):
        plt.subplot(330+1+i)
        batch = zooming.next()
        result = batch[0].astype('uint8')
    plt.imshow(result)
    plt.axis("off")
    st.pyplot(fig)
elif option == "Vertical flipping":
    image = Image.open('dog.jpg')
    data = img_to_array(image)
    samples = expand_dims(data,0)
    data_generated = ImageDataGenerator(vertical_flip=True)
    vertical_flipping = data_generated.flow(samples , batch_size = 1)
    for i in range(1):
        plt.subplot(330+1+i)
        batch = vertical_flipping.next()
        result = batch[0].astype('uint8')
    plt.imshow(result)
    plt.axis("off")
    st.pyplot(fig)
elif option == "Horizontal flipping":
    image = Image.open('dog.jpg')
    data = img_to_array(image)
    samples = expand_dims(data,0)
    data_generated = ImageDataGenerator(horizontal_flip=True)
    horizontal_flipping = data_generated.flow(samples , batch_size = 1)
    for i in range(1):
        plt.subplot(330+1+i)
        batch = horizontal_flipping.next()
        result = batch[0].astype('uint8')
    plt.imshow(result)
    plt.axis("off")
    st.pyplot(fig)
