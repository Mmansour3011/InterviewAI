import subprocess
import os

import numpy as np
import cv2
from collections import Counter

from tensorflow.keras.preprocessing import image
import speech_recognition as sr
from tensorflow.keras.models import model_from_json

from datetime import datetime
from matplotlib import pyplot as plt
import tkinter as tk
os.chdir("Video-Interview-Analysis-master")
current_date = datetime.now().strftime('%Y-%m-%d')
# Define the base directory using the current date
base_directory = os.path.join(os.getcwd(), current_date)

# Create the base directory if it doesn't exist
if not os.path.exists(base_directory):
    os.makedirs(base_directory)

#--------------------------------------Loading Files---------------------------------#

# Enter the abolute video path here (make sure there isnt a corresponding audio file yet)
directory = input("Enter the path of your videos directory>>>")

# Define a function to check if a file is a video
def is_video(file_path):
    try:
        # Try to capture a frame from the file
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            cap.release()
            return True
    except:
        return False
    return False

# Crawl through the directory
def crawl_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if is_video(file_path):
                process_video(file_path)
            else:
                print(f"Skipping non-video file: {file_path}")

# Loading OpenCV Cascade Classifier for Face Detection        
face_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.abspath(cv2.__file__)), 'data/haarcascade_frontalface_default.xml'))

# Loading emotion recognition model and weights 
model = model_from_json(open('fer.json', 'r').read())
model.load_weights('fer.h5') #load weights

def write_text_to_file(directory, text):
    with open(directory, "w") as file:
        file.write(text)

def process_video(video_path):  
    
    
    #----------------------------------Speech Detection Part-----------------------------------#
    text = "default text"
    # Exracting the video base name (Example: interview.mp4)
    video_base_name = os.path.basename(video_path)

    # Extracting video name
    video_name = os.path.basename(video_base_name).split('.')[0]
    # Define the subdirectory for the video
    video_directory = os.path.join(base_directory, video_name)

    # Create the video directory if it doesn't exist
    if not os.path.exists(video_directory):
        os.makedirs(video_directory)
    # Getting video directory
    video_dir = os.path.dirname(video_path)

    # Creating output (audio) file path
    audio_path = video_directory + "\\" + video_name + '.wav'

    print('Input file: {}'.format(video_path))

    # Creating subprocess to convert video file to audio
    subprocess.call(['ffmpeg', '-i', video_path, '-codec:a', 'pcm_s16le', '-ac', '1', audio_path])

    print('Output file: {}'.format(audio_path))

    # Initialize recognizer class (for recognizing the speech)
    r = sr.Recognizer()

    with sr.AudioFile(audio_path) as source:
        audio = r.listen(source)
        try:
            # Google speech recognition (You can select from other options)
            text = r.recognize_google(audio)
            
            # Printing speech
            print('Speech Detected:')
            print(text)
        
        except:
            text = 'Could not hear anything!'

    #-------------------------------------------------------------------------------------------#
            
    #------------------------------------Emotion Detection Part---------------------------------#    
    # Capturing the video using from the path
    cap = cv2.VideoCapture(video_path)
    # Get the width and height of the video frames
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(video_directory + '\\output.avi', fourcc, 20.0, (frame_width, frame_height))

    # Emotion labels
    emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')

    # Creating empty list which will be used to store emotions
    emotion_list = []

    # Reading video frame by frame
    while(True):
    
        ret, img = cap.read()
        
        # Reading till the end of the video
        if ret:
            
            # Converting to greyscale
            #img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
            # Detecting faces
            faces = face_cascade.detectMultiScale(gray_img, 1.32, 5)
            
            """Show image using imshow function (removing this will only display images where a 
            face was captured thus cutting out on some of the frames) """
            img = cv2.resize(img, (1000, 800))
            
            # For every face detected
            for (x,y,w,h) in faces:
                
                # Drawing a rectangle 
                cv2.rectangle(img,(x-78,y),(x+w-180,y+h+50),(0,255,0), thickness = 2) 
                
                # Cropping the face
                face = gray_img[int(y):int(y+h), int(x):int(x+w)]
                
                # Resizing the cropped face
                face = cv2.resize(face, (48, 48))
                
                # Converted face image to pixels
                img_pixels = image.img_to_array(face)
                img_pixels = np.expand_dims(img_pixels, axis = 0)
                
                # Scalling image
                img_pixels = img_pixels/255
                
                # Using the model to predict the detected face
                predictions = model.predict(img_pixels)
                
                # Finding the index with most value
                max_index = np.argmax(predictions[0])
                
                # Finding corresponding emotion
                emotion = emotions[max_index]
                
                # Storing detected emotions in a list
                emotion_list.append(emotion)
                
                # Writing detected emotions on the rectangle
                cv2.putText(img, emotion, (int(x), int(y)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            color = (255,255,255),
                            thickness = 1)
                out.write(img)
                # Showing the frame with detected face
                cv2.imshow('Emotion Recognizer', img)
            
            # Press q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Closing OpevCV		
    cap.release()
    cv2.destroyAllWindows()

    # Counting emotions from the generated list
    c = Counter(emotion_list)
    #print([(i, c[i] / len(emotion_list) * 100.0) for i in c])

    #-------------------------------------------------------------------------------------------#

    #--------------------------------------Displaying Results-----------------------------------#

    # Function to write text to a file in the video directory

    write_text_to_file(video_directory + "\\" + video_name + ".txt", text)
    # Visualizing emotions using a Pie Chart
    plt.figure(num='Detected Emotions')
    plt.pie([float(v) for v in c.values()], labels=[k for k in c], autopct='%1.0f%%')
    plt.title("Emotions Detected throughout the Interview")
    pie_chart_path = os.path.join(video_directory, "emotions_pie_chart.png")
    plt.savefig(pie_chart_path)  # Save the pie chart to a file
    #-------------------------------------------------------------------------------------------#
crawl_directory(directory)