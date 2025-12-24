

import imquality.brisque as brisque
import PIL.Image

img = PIL.Image.open("../../Image-Blur-detection-and-image-quality-check-python/Python/00005.jpg")
# Returns a score typically 0 (best) to 100 (worst)
score = brisque.score(img)
print(f"Badness Score: {score}")
