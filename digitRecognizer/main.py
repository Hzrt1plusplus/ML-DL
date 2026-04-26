import streamlit as st
from streamlit_mnist_canvas import st_mnist_canvas
import numpy as np
from neuralnet import * 

st.subheader("Input")
result = st_mnist_canvas()
model = EnhancedNN([-1], leakyRelu, leakyReluDeriv, heinit) # [ RECOMMENDED TO USE NeuralNetwork CLASS FROM NN MODULE HERE] 
model.load_model("./params3")

"""

    Try model here, streamlit required, 
    install using 'pip install streamlit' and afterwards download streamlit_mnist_canvas using 'pip install streamlit_mnist_canvas'.
    I tried to write a canvas like object using only pygame for demonstration, but since MNIST images are ant-aliased(see https://en.wikipedia.org/wiki/Anti-aliasing)
    it is crucial to use a brush tool that anti-aliases canvas and transforms image. By the way, if you want to avoid this problem, 
    you can normalize input images, means you can filter values and set values high than a theresold(for example 150) to 255 and lowers to 0. It helps model to learn 
    from aliased versions of training images and do the same normalization for test images. So, if you can get a decent score after training, use model with an aliased
    canvas which can be implemented easily.

    Other enhancments include using CNN architecture rather than FNNs which is spesifically designed for image processing. The main reason of 
    my Feedforward choice is, implementing all the system using low level libs and wanting to abstract all process. 

"""


if result.is_submitted:

    st.write("Output: ")
    st.image(result.resized_grayscale_array, caption="Grayscale 28x28 Image")


    # Predict
    prediction = model.predict(result.resized_grayscale_array.reshape(28*28))
    st.write("Predicted Digit:", prediction)
