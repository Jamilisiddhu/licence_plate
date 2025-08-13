import os
import io
import cv2 
import numpy as np
import easyocr
import imutils
import re
from flask import Flask, request, jsonify, render_template
from PIL import Image
import base64

# List of valid Indian State and Union Territory codes
INDIAN_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DH", "DL", "GA", "GJ", "HR", "HP", 
    "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", 
    "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UA", "UK", "UP", "WB", "AN", "DN", "BH"
}

# Initialize Flask app
app = Flask(__name__)

# Initialize EasyOCR reader once to avoid loading the model on every request
reader = easyocr.Reader(['en'])

def recognize_plate(image_data):
    """
    Core ANPR logic encapsulated in a function.
    Takes image data (bytes) as input and returns the recognized text and processed image.
    """
    # Convert image data to a NumPy array for OpenCV
    img = np.array(Image.open(io.BytesIO(image_data)))
    
    # Check if the image is valid
    if img is None:
        return None, "Error: Invalid image data."

    # Convert to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Image preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) # Noise reduction
    edged = cv2.Canny(bfilter, 30, 200) # Edge detection

    # Find contours and locate the license plate
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    final_text = "No Plate Detected"
    processed_img_base64 = None

    if location is not None:
        # Masking and cropping the license plate
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [location], 0, 255, -1) 
        
        (x, y) = np.where(mask == 255) 
        (x1, y1) = (np.min(x), np.min(y)) 
        (x2, y2) = (np.max(x), np.max(y)) 
        cropped_image = gray[x1:x2+1, y1:y2+1]

        # Use EasyOCR to read text
        result = reader.readtext(cropped_image)
        
        if result:
            raw_text = result[0][-2].upper().replace(' ', '')
            
            # 1. Character correction
            corrected_text = raw_text.replace('Z', '2').replace('G', '6').replace('I', '1').replace('O', '0')
            
            # 2. Check and correct the first two characters (State code)
            state_code_from_ocr = corrected_text[:2]
            
            final_state_code = state_code_from_ocr
            if state_code_from_ocr in INDIAN_STATE_CODES:
                final_state_code = state_code_from_ocr
            else:
                # Simple correction for common misreads
                if state_code_from_ocr == 'MW':
                    final_state_code = 'MH'
                elif state_code_from_ocr == 'HK':
                    final_state_code = 'HR'
                elif state_code_from_ocr == 'UR':
                    final_state_code = 'UP'
            
            plate_without_state_code = corrected_text[2:]
            combined_text = final_state_code + plate_without_state_code

            # 3. Format validation and correction using Regex
            pattern = re.compile(r'([A-Z]{2})([0-9]{1,2})([A-Z]{1,2})([0-9]{1,4})')
            match = pattern.search(combined_text)
            
            if match:
                state_code = match.group(1).strip()
                rto_code = match.group(2).strip().zfill(2)
                series_code = match.group(3).strip()
                unique_number = match.group(4).strip().zfill(4)
                
                final_text = f"{state_code} {rto_code} {series_code} {unique_number}"
            else:
                final_text = corrected_text
        
        # Draw the bounding box on the original image
        res_img = cv2.putText(img, text=final_text, org=(location[0][0][0], location[1][0][1] + 60), 
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        res_img = cv2.rectangle(img, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)

        # Convert the processed image to Base64 to send to the frontend
        _, buffer = cv2.imencode('.png', res_img)
        processed_img_base64 = base64.b64encode(buffer).decode('utf-8')
    else:
        # If no plate is found, just send back the original image
        _, buffer = cv2.imencode('.png', img)
        processed_img_base64 = base64.b64encode(buffer).decode('utf-8')
        
    return final_text, processed_img_base64


@app.route('/')
def home():
    """Renders the main page of the website."""
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    """
    API endpoint to receive the image and return the recognized text and processed image.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    image_data = file.read()
    
    recognized_text, processed_image = recognize_plate(image_data)
    
    return jsonify({
        "plate_number": recognized_text,
        "processed_image": processed_image
    })

if __name__ == '__main__':
    # When running locally, set debug=True for development
    app.run(debug=True, host='0.0.0.0', port=5000)
