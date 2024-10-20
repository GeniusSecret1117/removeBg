from rembg import new_session, remove
from PIL import Image, ImageFilter
from io import BytesIO
# For input points
import numpy as np

def remove_background(input_image_path, output_image_path):
    with open(input_image_path, 'rb') as input_file:
        input_image = input_file.read()
        
    output_image = remove(input_image)
    
    with open(output_image_path, 'wb') as output_file:
        output_file.write(output_image)

### With a specific model
def remove_background_isnet(input_image_path, output_image_path):
    with open(input_image_path, 'rb') as input_file:
        input_image = input_file.read()
    model_name = "isnet-general-use"
    session = new_session(model_name)
    output_image = remove(input_image, session=session)
    
    with open(output_image_path, 'wb') as output_file:
        output_file.write(output_image)

### With a Alpa matting
def remove_background_alpha(input_image_path, output_image_path):
    with open(input_image_path, 'rb') as input_file:
        input_image = input_file.read()
    output_image = remove(input_image, alpha_matting=True, alpha_matting_foreground_threshold=270,alpha_matting_background_threshold=20, alpha_matting_erode_size=11)
    
    with open(output_image_path, 'wb') as output_file:
        output_file.write(output_image)

### With a Only Mask
def remove_background_only_mask(input_image_path, output_image_path):
    with open(input_image_path, 'rb') as input_file:
        input_image = input_file.read()
    output_image = remove(input_image, only_mask=True)
    
    with open(output_image_path, 'wb') as output_file:
        output_file.write(output_image)

### With a post processing
def remove_background_post_processing(input_image_path, output_image_path):
    with open(input_image_path, 'rb') as input_file:
        input_image = input_file.read()
    output_image = remove(input_image, post_process_mask=True)
    
    with open(output_image_path, 'wb') as output_file:
        output_file.write(output_image)

### Replacing the background color
def remove_background_replace_background(input_image_path, output_image_path):
    with open(input_image_path, 'rb') as input_file:
        input_image = input_file.read()
    output_image = remove(input_image, bgcolor=(255, 255, 255, 255))
    
    with open(output_image_path, 'wb') as output_file:
        output_file.write(output_image)

### Using input points
def remove_background_input_points(input_image_path, output_image_path):
    with open(input_image_path, 'rb') as input_file:
        input_image = input_file.read()
    # Define the points and labels
    # The points are defined as [y, x]
    input_points = np.array([[400, 350], [700, 400], [200, 400]])
    input_labels = np.array([1, 1, 2])
    model_name = "isnet-general-use"
    session = new_session(model_name)
    output_image = remove(input_image,session=session, input_points=input_points, input_labels=input_labels)
    
    with open(output_image_path, 'wb') as output_file:
        output_file.write(output_image)

# Example usage
remove_background('examples/car-2.jpg', 'examples/car-ttttttttttt11.png')
remove_background_isnet('examples/car-2.jpg', 'examples/car-ttttttttttt22.png')
remove_background_alpha('examples/car-2.jpg', 'examples/car-ttttttttttt33.png')
remove_background_only_mask('examples/car-2.jpg', 'examples/car-ttttttttttt44.png')
remove_background_post_processing('examples/car-2.jpg', 'examples/car-ttttttttttt55.png')
remove_background_replace_background('examples/car-2.jpg', 'examples/car-ttttttttttt66.png')
remove_background_input_points('examples/car-2.jpg', 'examples/car-ttttttttttt7.png')

# Load the output image without background
output_image_path = 'examples/car-ttttttttttt1.png'
output_image = Image.open(output_image_path).convert("RGBA")

# Get dimensions
width, height = output_image.size

# Define adjustable shadow parameters based on image size
shadow_offset = (int(width * 0.1), int(height * 0.1))  # 10% offset
blur_radius = int(min(width, height) * 0.1)  # Blur radius is 10% of the smallest dimension
shadow_alpha = 100  # Shadow transparency (0-255)

# Create a shadow
shadow = output_image.copy()
shadow = shadow.filter(ImageFilter.GaussianBlur(blur_radius))  # Blur for soft shadow
shadow.putalpha(shadow_alpha)  # Set transparency for the shadow

# Create a new image with a transparent background
final_image = Image.new('RGBA', output_image.size)

# Position the shadow
final_image.paste(shadow, shadow_offset, shadow)  # Apply shadow with transparency

# Paste the original image on top of the shadow
final_image.paste(output_image, (0, 0), output_image)

# Save the final image
final_image.save('final_image_with_adjustable_shadow.png')

print("Final image with adjustable shadow saved as 'final_image_with_adjustable_shadow.png'.")