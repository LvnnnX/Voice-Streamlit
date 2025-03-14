import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title='Voice Recognition Machine Learning', page_icon='🎤', initial_sidebar_state='auto')

def clear_background():
    st.markdown(
    """
<style>
[data-testid^="stAppViewContainer"]{
    background-color=black;

}
.sidebar .sidebar-content {
    background-image: linear-gradient(#2e7bcf,#2e7bcf);
    color: white;
}
[data-testid^="stFormSubmitButton"] > button:first-child {
    background-color: transparent;
    text-align: center;
    margin: 10;
    position: relative;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
}
[data-testid^="stFormSubmitButton"]:hover > button:first-child {
    border-color: green;
}

[class^="st-b"]  {
    color: white;
}
[data-testid^="stMarkdownContainer"]{
    background-color: transparent;
    size: 20px;
    color: white;
    weight: bold;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

[class^="main css-k1vhr4 egzxvld3"]{
    background-color:#0e1117;
}

</style>
""",
    unsafe_allow_html=True,
)

clear_background()

st.header('Voice Recognition Machine Learning')
st.write('Voice Recognition is meant to identify people by the unique characteristics of their voices.')

st.image('https://www.altexsoft.com/static/blog-post/2023/11/34ecda72-b5aa-46aa-af21-8a7a48ed3921.jpg', use_column_width=True, caption='Source: Altexsoft')

st.header('Data Collection')
st.write('Data collection is the process of gathering and measuring information on variables of interest, in an established systematic fashion that enables one to answer stated research questions, test hypotheses, and evaluate outcomes.')
st.write('The data that we will be using obtained manually from our own voice recordings.')
st.image('img/dataset.png', use_column_width=False, caption='Source: Author')

st.header('Data Preparation')
st.subheader('Data Labeling')
st.write('Data labeling or annotation invovles marking raw data with the correct answers to facilitate supervised machine learning. During training, your model will learn to identify patterns in new data and make accurate predictions based on these labels. Therefore, the quality and accuracy of the labels are crucial for the success of machine learning models.')

st.subheader('Feature Extraction - Time-frequency domain features')
st.write('This domain integrates both time and frequency elements, utilizing different types of spectograms to visually represent sound. A spectogram can be derived from a waveform by applying the short-time Fourier transform. Among the most widely used features in the time-frequency domain are mel-frequency cepstral coefficients (MFCCs). These features operate within the range of human hearing and are thus based on the mel scale and mel spectogram previously mentioned.')

st.image('https://www.altexsoft.com/media/2022/05/word-image.jpeg', use_column_width=True, caption='Source: Altexsoft')

st.header('Model Building')
st.subheader('Convolutional Neural Network (CNN)')
st.write('Convolutional neural networks are at the forefront of computer vision in healthcare and various other sectors. They are frequently considered the ideal option for image recognition tasks. The effectiveness of CNN architecture in processing spectrograms further reinforces this assertion.')
st.image('https://www.researchgate.net/publication/330106889/figure/fig1/AS:710963951063040@1546518423301/Architecture-of-a-Convolutional-Neural-Network-CNN-The-traditional-CNN-structure-is.png', use_container_width=True, caption='Source: ResearchGate')

# st.header('')