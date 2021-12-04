#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

train_dir = 'C:\\Users\\archive\\train'
val_dir = 'C:\\Users\\archive\\test'
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
# emotion_model.load_weights('emotion_model.h5')

cv2.ocl.setUseOpenCL(False)


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
emotion_model_info = emotion_model.fit_generator(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=7178 // 64)

emotion_model.save('emotion_model.h5')


# In[1]:


import tensorflow as tf

emotion_model = tf.keras.models.load_model('emotion_model.h5')
emotion_model.layers[0].input_shape

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
emoji_dist={0:"C:/Users/archive/emoji/angry.png",1:"C:/Users/archive/emoji/disgusted.png",2:"C:/Users/archive/emoji/fearful.png",3:"C:/Users/archive/emoji/happy.png",4:"C:/Users/archive/emoji/neutral.png",5:"C:/Users/archive/emoji/sad.png",6:"C:/Users/archive/emoji/surprise.png"}


# In[3]:


import PySimpleGUI as sg
import cv2
import numpy as np
import cv2
from PIL import Image
import io

show_text=[0]  
cap = cv2.VideoCapture(0)

def main():

    sg.theme('systemdefault')

    # define the window layout
    layout = [[sg.Text('Emojify', size=(40, 1), justification='center', font='Helvetica 20')],
              [sg.Image(filename='', key='image'),sg.Image(filename='',key="emoji")],
              [sg.Button('Start', size=(10, 1), font='Helvetica 14'),
               sg.Button('Exit', size=(10, 1), font='Helvetica 14'), ]]

    # create the window and show it without the plot
    window = sg.Window('Real time emoji',
                       layout, location=(800, 400))

    # ---===--- Event LOOP Read and display frames, operate the GUI --- #
    recording = False

    while True:
        event, values = window.read(timeout=40)
        
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break

        elif event == 'Start':
            recording = True
            
        if recording:
            flag1, frame1 = cap.read()
            frame1 = cv2.resize(frame1,(600,500))
            
            bounding_box = cv2.CascadeClassifier('C:/Users/archive/haarcascade_frontalface_default.xml')
            gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
            
            for (x, y, w, h) in num_faces:
                cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                roi_gray_frame = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
                
                prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                
                cv2.putText(frame1, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                show_text[0]=maxindex
    
                imgbytes = cv2.imencode('.png', frame1)[1].tobytes()
                window['image'].update(data=imgbytes)
                
                image = Image.open(emoji_dist[show_text[0]])
                image.thumbnail((400, 400))
                
                im = cv2.imread(emoji_dist[show_text[0]])
                cv2.imwrite("emo.png",im)
             
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["emoji"].update(data=bio.getvalue())
                
                      
    cap.release()
    window.close()
    
main()


# In[ ]:





# In[ ]:




