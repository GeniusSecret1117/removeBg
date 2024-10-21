from flask import Flask, request, send_file
from rembg import new_session, remove
from io import BytesIO
from flask_cors import CORS
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)
print("gggg")
#CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://picground.co.uk"]}})
@app.route('/rrr', methods=['POST'])
def rrr():
    return "dsfd"
# Route to remove background using the default rembg model
@app.route('/remove-background', methods=['POST'])
def remove_background():
    binary_data = request.data  # Receive the binary data
    
    # Call rembg to remove the background
    output = remove(binary_data)

    # Open the image using PIL and convert it to RGBA format
    img = Image.open(BytesIO(output)).convert("RGBA")

    # Save the image back to a BytesIO object to send as a response
    output_io = BytesIO()
    img.save(output_io, format='PNG')
    output_io.seek(0)  # Move to the start of the BytesIO buffer

    return send_file(output_io, mimetype='image/png')

# Route to remove background using a specific model
@app.route('/remove-background-isnet', methods=['POST'])
def remove_background_isnet():
    binary_data = request.data  # Receive the binary data

    model_name = "isnet-general-use"
    session = new_session(model_name)
    output = remove(binary_data, session=session)

    # Convert output to a PIL image and then to a BytesIO object
    img = Image.open(BytesIO(output)).convert("RGBA")
    output_io = BytesIO()
    img.save(output_io, format='PNG')
    output_io.seek(0)

    return send_file(output_io, mimetype='image/png')

# Route to remove background with alpha matting
@app.route('/remove-background-alpha', methods=['POST'])
def remove_background_alpha():
    binary_data = request.data

    output = remove(binary_data, alpha_matting=True, 
                    alpha_matting_foreground_threshold=270, 
                    alpha_matting_background_threshold=20, 
                    alpha_matting_erode_size=11)

    img = Image.open(BytesIO(output)).convert("RGBA")
    output_io = BytesIO()
    img.save(output_io, format='PNG')
    output_io.seek(0)

    return send_file(output_io, mimetype='image/png')

# Route to remove background with post-processing
@app.route('/remove-background-post-processing', methods=['POST'])
def remove_background_post_processing():
    binary_data = request.data

    output = remove(binary_data, post_process_mask=True)

    img = Image.open(BytesIO(output)).convert("RGBA")
    output_io = BytesIO()
    img.save(output_io, format='PNG')
    output_io.seek(0)

    return send_file(output_io, mimetype='image/png')

# Route to replace the background with a solid color
@app.route('/remove-background-replace-background', methods=['POST'])
def remove_background_replace_background():
    binary_data = request.data

    output = remove(binary_data, bgcolor=(255, 255, 255, 255))

    img = Image.open(BytesIO(output)).convert("RGBA")
    output_io = BytesIO()
    img.save(output_io, format='PNG')
    output_io.seek(0)

    return send_file(output_io, mimetype='image/png')

# Route to remove background with input points
@app.route('/remove-background-input-points', methods=['POST'])
def remove_background_input_points():
    binary_data = request.data

    input_points = np.array([[400, 350], [700, 400], [200, 400]])
    input_labels = np.array([1, 1, 2])
    model_name = "isnet-general-use"
    session = new_session(model_name)
    output = remove(binary_data, session=session, input_points=input_points, input_labels=input_labels)

    img = Image.open(BytesIO(output)).convert("RGBA")
    output_io = BytesIO()
    img.save(output_io, format='PNG')
    output_io.seek(0)

    return send_file(output_io, mimetype='image/png')

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=80)
    app.run(host='0.0.0.0', port=80, debug=True)
