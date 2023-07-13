from fastapi import FastAPI
from pydantic import BaseModel
import torch
from torchvision.transforms import ToTensor
import urllib.request
from PIL import Image
import requests
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

app = FastAPI()
class PredictionRequest(BaseModel):
    image_url: str

@app.get("/download_model")
async def download_model():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    file_id = '1WHESZSDimcJ9WMsYN7wTiZUuURncvJUt'
    file = drive.CreateFile({'id': file_id})
    file.GetContentFile('model.pth')

    return {"message": "Model downloaded successfully!"}

model_path = 'cropdiseasemodel.pth' 
model = torch.load(model_path)
model.eval()

@app.post('/predict')
async def predict(payload: PredictionRequest):
    image_url = payload.image_url
    image_tensor = load_and_preprocess_image(image_url)

    with torch.no_grad():
        prediction = model(image_tensor)
        class_labels = [
                        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                        'Tomato___Tomato_mosaic_virus',
                        'Tomato___Late_blight',
                        'Tomato___Septoria_leaf_spot',
                        'Strawberry___healthy',
                        'Tomato___Target_Spot',
                        'Tomato___Early_blight',
                        'Strawberry___Leaf_scorch',
                        'Tomato___healthy',
                        'Tomato___Bacterial_spot',
                        'Tomato___Leaf_Mold',
                        'Tomato___Spider_mites Two-spotted_spider_mite',
                        'Pepper,_bell___healthy',
                        'Potato___healthy',
                        'Squash___Powdery_mildew',
                        'Soybean___healthy',
                        'Potato___Late_blight',
                        'Potato___Early_blight',
                        'Pepper,_bell___Bacterial_spot',
                        'Peach___healthy',
                        'Raspberry___healthy',
                        'Peach___Bacterial_spot',
                        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                        'Orange___Haunglongbing_(Citrus_greening)',
                        'Grape___Esca_(Black_Measles)',
                        'Grape___healthy',
                        'Corn_(maize)___Common_rust_',
                        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                        'Blueberry___healthy',
                        'Apple___Cedar_apple_rust',
                        'Corn_(maize)___healthy',
                        'Cherry_(including_sour)___Powdery_mildew',
                        'Cherry_(including_sour)___healthy',
                        'Grape___Black_rot',
                        'Corn_(maize)___Northern_Leaf_Blight',
                        'Apple___healthy',
                        'Apple___Black_rot',
                        'Apple___Apple_scab'
                        ] 

        probabilities = torch.nn.functional.softmax(prediction, dim=1)[0]
        predicted_class_index = torch.argmax(probabilities).item()
        predicted_class_label = class_labels[predicted_class_index]
        confidence_score = probabilities[predicted_class_index].item()
    return {'predicted_class': predicted_class_label, 'confidence_score': confidence_score}

def load_and_preprocess_image(image_url):
    image_data = urllib.request.urlopen(image_url).read()
    image = Image.open(BytesIO(image_data))

    transform = ToTensor()
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)
    return image_tensor

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)