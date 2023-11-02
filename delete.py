from PIL import Image, ImageDraw
from io import BytesIO
from ultralytics import YOLO
import json
from flask import request, Response, Flask , send_file
from waitress import serve

import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'gif', 'bmp'}

@app.route("/detect", methods=["POST"])
def detect():
    # Ensure that the uploaded file is an image
    if 'image_file' not in request.files:
        return {"error": "No file part"}

    file = request.files['image_file']

    # Ensure that the uploaded file is an image
    if file.filename == '':
        return {"error": "No selected file"}

    if not allowed_file(file.filename):
        return {"error": "Invalid image format"}

    # Read the uploaded image
    image = Image.open(file)

    # Perform object detection
    image_with_boxes, _ = detect_objects_on_image(image)

    # Convert the PIL image back to a NumPy array
    image_with_boxes = np.array(image_with_boxes)

    # Convert the NumPy array to bytes
    img_byte_array = BytesIO()
    Image.fromarray(image_with_boxes).save(img_byte_array, format="PNG")

    # Send the image with detected objects as a response
    response = Response(img_byte_array.getvalue())
    response.headers["Content-Type"] = "image/png"
    response.headers["Content-Disposition"] = 'inline; filename="detected_image.png"'

    return response





def detect_objects_on_image(image):
    model = YOLO("best.pt")
    results = model.predict(image)
    result = results[0]
    
    # Convert the image to a NumPy array
    image_with_boxes = np.array(image)

    # Convert the NumPy array back to a PIL Image
    image_with_boxes = Image.fromarray(image_with_boxes)

    draw = ImageDraw.Draw(image_with_boxes)
    
    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)

        # Draw the bounding box on the image
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), f"{result.names[class_id]}: {prob}", fill="red")

    return image_with_boxes, None






print("here")
if __name__ == '__main__':
    
     app.run(host='0.0.0.0', port=8000)