
import numpy as np
import tensorflow as tf
import PIL
import streamlit as st

import cv2
from PIL import Image, ImageOps


html_temp = """
    <div style="background-color:#221F1F;padding:7px">
    <h2 style="color:white;text-align:center;">G-20</h2>
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
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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
    class_names=["Prime minister of india :- Narendra Damodardas Modi ", "President Of Russian Federation :- Vladimir Vladimirovich Putin",
              "President of People's Republic Of China :- Xi Jinping", "President Of Turkey :- Recep Tayyip Erdogan",
              "Prime Minister of Canada :- Justin Pierre James Trudeau", "President Of The European Commission :- Ursula Gertrud von der Leyen",
              "Chancellor of Germany :- Olaf Scholz MdB", "Prime Minister of the United Kingdom :- Rishi Sunak",
              "President of Brazil :- Luiz Inacio Lula da Silva", " Crown Prince and Prime Minister the Kingdom of Saudi Arabia :- Mohammed bin Salman Al Saud","Prime Minister of Italy :- Giorgia Meloni","President Of French Republic :- Emmanuel Jean Michel Frederic Macron","President Of Republic Of Indonesia :- Joko Widodo",
              "President of Republic Of Korea( South Korea) :- Yoon Suk Yeol","President Of South Africa :- Cyril Ramaphosa","Prime Minister Of Australia :- Anthony Norman Albanese","President Of United Mexican States :- Andres Manuel Lopez Obrador","President Of Argentina :- Alberto Angel Fernandez",
              "Prime Minister Of Japan :- Fumio Kishida","President Of the United States of America :- Joseph Robinette Biden Jr"]
    try:
        st.subheader( "Person in the following Image is The "+class_names[prediction[0]])
        # st.markdown(html_temp,unsafe_allow_html=True)
    except:
        st.subheader("System not able to detect the following face of the person. Kindly choose another photo")
