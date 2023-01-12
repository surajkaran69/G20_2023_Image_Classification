
import numpy as np
import tensorflow as tf
import PIL
import streamlit as st

import cv2
from PIL import Image, ImageOps


html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">G20 Leaders </h2>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)
st.set_option('deprecation.showfileUploaderEncoding',False)
@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('G20.h5')
    return model

model=load_model()
st.write("""
         Upload Photo of a leader of G20 countries.
         """
         )
uploaded_file=st.file_uploader("Please upload an image of the above mentioned personalities.",type=["jpg","png"])
from PIL import Image,ImageOps


if uploaded_file is None:
    st.text("")
else:
    result=0
     # To read file as bytes:
    image = Image.open(uploaded_file)
    st.image(image, width=350)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if img is not None:
       gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       faces = face_cascade.detectMultiScale(gray, 1.3, 5)

       cropped_faces = []
       for (x,y,w,h) in faces:
               
               roi_color = img[y:y+h, x:x+w]
               cropped_faces.append(roi_color)
       
    
    for img in cropped_faces:
       scalled_raw_img = cv2.resize(img, (32, 32))
       
       scalled_raw_img = np.array(scalled_raw_img).reshape(1,32,32,3).astype(float)
       result=model.predict(scalled_raw_img)
       prediction=np.argmax(result,axis=1)
       output = prediction[0]
    class_names=["Narendra Damodardas Modi", "Vladimir Vladimirovich Putin",
              "Xi Jinping", "Recep Tayyip Erdogan",
              "Justin Pierre James Trudeau", "Ursula Gertrud von der Leyen",
              "Olaf Scholz MdB", "Rishi Sunak",
              "Luiz Inacio Lula da Silva", "Mohammed bin Salman Al Saud","Giorgia Meloni","Emmanuel Jean Michel Frederic Macron","Joko Widodo",
              "Yoon Suk Yeol","Cyril Ramaphosa","Anthony Norman Albanese","Andres Manuel Lopez Obrador","Alberto Angel Fernandez",
              "Fumio Kishida","Joseph Robinette Biden Jr"]
    try:
        st.subheader( "The image is of "+class_names[prediction[0]])
        # st.markdown(html_temp,unsafe_allow_html=True)
    except:
        st.subheader("Face not detectable")
