from tensorflow import keras
from PIL import Image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
#from keras.preprocessing import image
import keras.utils as image
import numpy as np

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title='Covid Disease Classification')
st.title('Covid 19 Classification')

@st.cache(allow_output_mutation=True)
def get_best_model():
    model = keras.models.load_model('Detection_Covid_19.h5',compile=False)
    model.make_predict_function()          # Necessary
    print('Model loaded. Start serving...')
    return model

st.subheader('Classify the image')
image_file = st.file_uploader('Choose the Image', ['jpg', 'png'])
print(image_file)
st.image(image_file, caption='Chest MRI Image', use_column_width=True)

if image_file is not None:

    import numpy as np
    import keras.utils as image
    # image = Image.open(image_file)
    #xtest_image = image.load_img('Dataset/Prediction/NORMAL2-IM-0354-0001.jpeg', target_size = (224, 224))
    #SARS-10.1148rg.242035193-g04mr34g05x-Fig5-day9
    xtest_image = image.load_img(image_file, target_size = (224, 224))
    xtest_image = image.img_to_array(xtest_image)
    xtest_image = np.expand_dims(xtest_image, axis = 0)
    model=get_best_model()
    #results = model.predict_classes(xtest_image)
    predict_x=model.predict(xtest_image) 
    results=predict_x
    if results == 0:
        prediction = 'Positive For Covid-19'
    else:
        prediction = 'Negative for Covid-19'
    print("Prediction Of Our Model : ",prediction)
    st.markdown(f'<h3>The image is predicted as {prediction}.</h3>', unsafe_allow_html=True)


    
    # image = Image.open(image_file)
    # st.image(image, caption='Input Image')

    # image = image.resize((224,224),Image.ANTIALIAS)
    # img_array = np.array(image)
    
    # x = np.expand_dims(img_array, axis=0)
    # images = np.vstack([x])
    # model=get_best_model()

    # # img = image.img_to_array(img)
    # # img = np.expand_dims(img,axis=0)
    # # predict_x=model.predict(img) 
    # # classes_x=np.argmax(predict_x,axis=1)

    
    # classes = model.predict(images, batch_size=10)
    # if classes>0.5:
    #     prediction = 'Normal'
    # else:
    #     prediction = 'Covid'
    # st.write(f'The image is predicted as {prediction}')
