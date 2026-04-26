import streamlit as st
from streamlit_mnist_canvas import st_mnist_canvas
import numpy as np
from neuralnet import * 

st.subheader("Input")
result = st_mnist_canvas()
model = EnhancedNN([-1], leakyRelu, leakyReluDeriv, heinit)
model.load_model("./params3")


if result.is_submitted:

    st.write("Output: ")
    st.image(result.resized_grayscale_array, caption="Grayscale 28x28 Image")


    # Predict
    prediction = model.predict(result.resized_grayscale_array.reshape(28*28))
    st.write("Predicted Digit:", prediction)