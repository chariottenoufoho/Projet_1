from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import face_recognition
import os
from PIL import Image
import io
import uvicorn

app = FastAPI()

# Allow all origins
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Charger les signatures des visages à partir de la base de données
signatures_class = np.load('FaceSignature_db.npy')
X = signatures_class[:, 0: -1].astype('float')
Y = signatures_class[:, -1]

@app.post("/RCW_API")
async def detect_similar_faces(file: UploadFile = File(...)):
    try:
        image_stream = io.BytesIO(await file.read())
        image = Image.open(image_stream)
        img = np.array(image)

        # Redimensionnement de l'image pour accélérer le traitement
        img_resize = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)

        # Détection des visages dans l'image uploadée
        faces_current = face_recognition.face_locations(img_resize)
        encodes_current = face_recognition.face_encodings(img_resize, faces_current)

        # Parcourir toutes les images PNG dans le dossier spécifié
        images_path = './images'
        similar_images = []
        for filename in os.listdir(images_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(images_path, filename)
                image_to_check = Image.open(path)
                image_to_check_array = np.array(image_to_check)
                
                # Convertir et redimensionner comme l'image uploadée pour comparaison
                image_to_check_array_resized = cv2.resize(image_to_check_array, (0, 0), None, 0.25, 0.25)
                image_to_check_array_resized = cv2.cvtColor(image_to_check_array_resized, cv2.COLOR_BGR2RGB)
                
                # Détection des visages et encodages dans l'image du dossier
                faces_to_check = face_recognition.face_locations(image_to_check_array_resized)
                encodes_to_check = face_recognition.face_encodings(image_to_check_array_resized, faces_to_check)
                
                for encode_check in encodes_to_check:
                    matches = face_recognition.compare_faces(encodes_current, encode_check)
                    if True in matches:
                        # Ajouter le nom du fichier similaire à la liste
                        similar_images.append(filename)
        
        return {"similar_images": similar_images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
