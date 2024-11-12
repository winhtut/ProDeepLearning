from PIL import Image
import numpy as np

# image အား ရယူခြင်း
image_path = 'img.png'
image = Image.open(image_path)

# image အား grayscale သို့ပြောင်းခြင်း
gray_image = image.convert('L')

# Get the grayscale data as a numpy array
gray_data = np.array(gray_image)

# Print grayscale values
print(gray_data)

# Save the grayscale image (optional)
gray_image.save('grayscale_image.png')

# Show the grayscale image (optional)
gray_image.show()
