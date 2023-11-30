from keras.models import load_model
from PIL import Image
import numpy as np
model = load_model('C:/Users/202-5/Documents/my_model.h5')
image_path = './pic/ISIC_0024318.jpg'
img = Image.open(image_path)
img = img.resize((100, 75))
img = np.array(img)
img = img / 255.0
img = np.expand_dims(img, axis=0)
predictions = model.predict(img)
predicted_class = np.argmax(predictions, axis=1)
print("예측 클래스:", predicted_class)