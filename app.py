from flask import Flask, request, render_template
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image

app = Flask(__name__)

IMG_SIZE = 256

model = tf.keras.models.load_model('0.89_Dense121_.h5')

def crop_image_from_gray(img, tol=7):
    """
    Applies masks to the orignal image and 
    returns the a preprocessed image with 
    3 channels
    
    :param img: A NumPy Array that will be cropped
    :param tol: The tolerance used for masking
    
    :return: A NumPy array containing the cropped image
    """
    # For Grayscale images
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(axis=1), mask.any(axis=0))]

    # If we have a normal RGB images
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # RGB image to grayscale
        mask = gray_img > tol  # creates a boolean matrix

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # Whole image is cropped as it was too dark
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(axis=1), mask.any(0))]  # applies mask to pixel 0
            img2 = img[:, :, 1][np.ix_(mask.any(axis=1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(axis=1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 10), -4, 128)
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/score', methods=['POST'])
def score():
    if request.method == "POST":
        image = request.files['u']
        image = Image.open(image)
        image = np.array(image)
        image = preprocess(image)
        image = np.expand_dims(image, axis=0)
        pred = model.predict(image)
        output = np.argmax(pred, axis=1)[0]  # Get the predicted stage

        # Define the messages based on the predicted stage
        stage_info = {
            0: ("0: No Diabetic Retinopathy", "You do not need any treatment. Regular eye check-ups are recommended."),
            1: ("1: Mild Diabetic Retinopathy", "You may need to monitor your condition. Consult your eye specialist."),
            2: ("2: Moderate Diabetic Retinopathy", "Follow up with your eye specialist for further assessment."),
            3: ("3: Severe Diabetic Retinopathy", "Immediate medical attention is advised. Please contact your doctor."),
            4: ("4: Proliferative Diabetic Retinopathy", "Urgent treatment is required. Please seek medical help right away.")
        }

        # Get the message for the predicted stage
        prediction_text, action_text = stage_info.get(output, ("Unknown Stage", "Consult your healthcare provider."))

    return render_template('index.html', 
                           prediction_text=f'{prediction_text}', 
                           action_text=f'Action Needed: {action_text}')

app.run(debug=True)
