import streamlit as st
import numpy as np
import pickle
from tensorflow import keras
from keras.models import load_model
from PIL import Image
from numpy import expand_dims
import os
from datetime import datetime
import pandas as pd
from skimage.segmentation import mark_boundaries
from lime import lime_image
import csv 

save_path = './model.h5'
model = keras.models.load_model(save_path)

html_temp = """
    <div style="background-color:#92a8d1;padding:0px">
    <h2 style="color:white;text-align:center;">Traffic Sign Recognition App </h2>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)

SignNames = { 0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)', 9:'No passing', 
            10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection', 12:'Priority road', 13:'Yield', 
            14:'Stop', 15:'No vehicles', 16:'Veh > 3.5 tons prohibited', 17:'No entry', 
            18:'General caution', 19:'Dangerous curve left', 20:'Dangerous curve right', 
            21:'Double curve', 22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right', 
            25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing', 
            29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing', 
            32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead', 
            35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left', 38:'Keep right', 
            39:'Keep left', 40:'Roundabout mandatory', 41:'End of no passing', 42:'End no passing veh > 3.5 tons',
            43:'Select Traffic Sign'}

def LoadTrainData():
    train_set = pickle.load(open('./train_set.pkl', 'rb'))
    y_train = pickle.load(open('./y_train.pkl', 'rb'))
    return train_set,y_train
           
def ImageProcessing(filename):
    #Read Image and Resize.
    picture=im=Image.open(filename)
    if(im.mode == "RGBA"):
       im = im.convert("RGB")
    im=im.resize((32,32))
    im =np.array(im)
    
    #Normalizing the iamges and bringing it to a scale between 0 to 1
    train_norm_output = im.astype('float32')
    train_norm_output /= 255

    #Expanding the dimension to (1,32,32,3)
    sample_images_output = expand_dims(train_norm_output, 0)
    return sample_images_output
    
          
list_filename = []     
def predict_uploaded_files_values(filename):
    
    picture=Image.open(filename)
    sample_images_output = ImageProcessing(filename)
    
    #Display details in a form
    form = st.form("my_form" + filename.name.upper())
    form.subheader('Input Image : ' + filename.name.upper())
    
    #predict the Input Image and get Confidence Score, Predicted Label.
    y_pred_prob =  model.predict(sample_images_output)
    value = int(np.argmax(y_pred_prob,axis=1))
    confidence_score = np.max(y_pred_prob[0])*100
    st.session_state[filename.name + "_old"] = str(SignNames[value]) + "|" + str(confidence_score) 
    
    cols = form.columns((3,5,8))
    cols[0].image(picture)
    cols[1].info("Predicted Traffic Sign :" + SignNames[value])
    #Chosen Label by user.
    result=cols[2].selectbox('Select Correct Traffic Sign if Predicted Sign is Wrong :', SignNames.values(), index=value)
    
    #Store the fianl value by user.
    st.session_state[filename.name] = str(result) 
    st.session_state["UploadStatus"] = 'True'
    
    colms = form.columns(2)
    #Model Interpretability
    if colms[0].form_submit_button("Interpret Model"):  
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(sample_images_output[0].astype('double'), model.predict,top_labels=3, hide_color=0, num_samples=1000)
        
        temp_1, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, negative_only=False, hide_rest=False, num_features=5)
        temp_2, mask_2 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False)
        form.subheader('Part of Images used to Interpret Model')
        cols = form.columns(4) 
        # first column of the ith row
        cols[0].image(temp_1, use_column_width=True,clamp=True)
        cols[1].image(temp_2, use_column_width=True,clamp=True)
        cols[2].image(mask_1, use_column_width=True,clamp=True) 
        cols[3].image(mask_2, use_column_width=True,clamp=True) 
    
    #Submit Button
    if colms[1].form_submit_button("Submit Selected Traffic Sign"):
        train_set,y_train = LoadTrainData()
        
        #To add new train values.
        train_set = np.append(train_set, sample_images_output, axis=0)
        
        #To add y_train values.
        index_value = [i for i in SignNames if SignNames[i]==str(result)][0]
        y_train_new = np.zeros((1, 43))
        y_train_new[0][index_value] = 1
        y_train = np.append(y_train, y_train_new, axis=0)
        pickle.dump(train_set, open('./train_set.pkl', 'wb'))
        pickle.dump(y_train, open('./y_train.pkl', 'wb'))
    
    #Submit Button
    #colms[2].form_submit_button("Submit New Corrected Value")
        
  
st.subheader('**Upload one or more Images.**')
uploaded_files = st.file_uploader("",accept_multiple_files=True)
if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        predict_uploaded_files_values(uploaded_file)
    
    list_a =[]
    for values in st.session_state:
        #check in order to prevent the display of Confidence Score table.
        if(values == "UploadStatus"):	
            if(st.session_state["UploadStatus"] == "True"):
                i=0
                st.subheader('Consolidated Data in Tabular Form')
                for value in st.session_state:
                  list_b =[]
                  if value.endswith('_old'):
                      pred_confiscore = st.session_state[value].split("|")
                      list_b = [value.replace('_old',''),pred_confiscore[0],float(pred_confiscore[1])]
                      list_a.append(list_b)
                      i=i+1
                df = pd.DataFrame(list_a,columns=('Image Name','Predicted Label','Confidence Score'))
                df = df.sort_values(by = 'Confidence Score',ascending = False)
                st.dataframe(df)
       

#This method will write the file details in a CSV file. In ordre to track what all items has been re-trained.
def get_session_values():
    now = datetime.now().strftime("%d%m%Y_%H%M%S")
    textfile = 'D:\Barnana\StreamLit\Output' + '\OutputFile_'+ now + '.csv'
    rows=[]
    fields = ['ImageName', 'SignName', 'ConfidenceScore'] 
    for value in st.session_state:
        if value.endswith('.png') or value.endswith('_old'):
            if "|" in st.session_state[value]:
                values=st.session_state[value].split('|')
                rows.append([value,values[0],values[1]])    
            else:
                rows.append([value,st.session_state[value],'NA'])
            
    # writing to csv file 
    with open(textfile, 'w') as csvfile: 
        csvwriter = csv.writer(csvfile)  
        csvwriter.writerow(fields) 
        csvwriter.writerows(rows)
    st.info("The output textfile is generated in the path: " + textfile)
    return textfile
            
        

if st.button("Re-Train Model"):  
    train_set,y_train = LoadTrainData()
    model.fit(train_set, y_train, epochs = 1)
    model.save('./model.h5')
    st.info('MODEL has been re-trained successfully. Re-Load the application and feed the required Image to view the correct output.')
    textfile = get_session_values()
    


	


	



