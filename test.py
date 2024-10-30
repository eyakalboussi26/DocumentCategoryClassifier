import os
import cv2
import numpy as np
import fitz  # PyMuPDF
from ultralytics import YOLO

# Chemins des fichiers
pdf_path = r'C:/Users/ekalboussi/OneDrive - ALTEN Group/Bureau/Project_NLP/OCR-and-Classifier-Doc/Classification/Test/Pdf.pdf'
output_image_dir = r'pdf_data'
output_folder = r'C:/Users/ekalboussi/OneDrive - ALTEN Group/Bureau/Project_NLP/OCR-and-Classifier-Doc/Classification/Test/Result_Classification'
output_log_path = r'C:/Users/ekalboussi/OneDrive - ALTEN Group/Bureau/Project_NLP/OCR-and-Classifier-Doc/Classification/Test/classification_log.txt'

# Créer les dossiers de sortie si non existants
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Étape 1 : Convertir le PDF en images avec PyMuPDF
print(f"Conversion du PDF {pdf_path} en images...")
pdf_document = fitz.open(pdf_path)
for page_num in range(pdf_document.page_count):
    page = pdf_document.load_page(page_num)  # Charger la page
    pix = page.get_pixmap()  # Récupérer une image de la page
    output_image_path = os.path.join(output_image_dir, f'page_{page_num + 1}.png')
    pix.save(output_image_path)  # Sauvegarder l'image de la page
pdf_document.close()

# Chemin vers le modèle YOLOv5 entraîné
model_path = r'C:/Users/ekalboussi/OneDrive - ALTEN Group/Bureau/Project_NLP/OCR-and-Classifier-Doc/Classification/Train/safe_directory/trained-model.pt'

# Charger le modèle YOLOv5 personnalisé
print(f"Chargement du modèle YOLOv5 depuis {model_path}...")
model = YOLO(model_path)

# Étape 2 : Appliquer la classification sur chaque image générée à partir du PDF
image_paths = [os.path.join(output_image_dir, f) for f in os.listdir(output_image_dir) if f.endswith('.png')]

# Ouvrir le fichier de log pour écrire les résultats
with open(output_log_path, 'w') as log_file:
    for image_path in image_paths:
        # Lire l'image originale
        img = cv2.imread(image_path)
        
        # Redimensionner l'image si nécessaire (par exemple, 640x640)
        img_resized = cv2.resize(img, (640, 640))
        
        # Effectuer une inférence sur l'image redimensionnée
        results = model(img_resized, conf=0.5, imgsz=640)  # Ajuster le seuil de confiance et la taille de l'image
        
        # Nom du fichier pour sauvegarder les résultats
        base_name = os.path.basename(image_path)
        output_image_path = os.path.join(output_folder, base_name)
        
        if not results:
            print(f"Aucune détection pour l'image {base_name}.")
            log_file.write(f"Aucune détection pour l'image {base_name}.\n")
        else:
            for result in results:
                # Log the results for debugging
                log_file.write(f"Détections pour {base_name}:\n")
                for i, box in enumerate(result.boxes):
                    xyxy = box.xyxy.cpu().numpy().flatten()  # Assurez-vous que `box.xyxy` est un tenseur PyTorch
                    conf = box.conf.cpu().numpy().item()  # Convertir la confiance en un nombre flottant
                    class_name = result.names[int(box.cls)]
                    
                    # Écrire les informations dans le fichier de log
                    log_file.write(f"Objet {i+1}:\n")
                    log_file.write(f" - Coordonnées brutes: {xyxy.tolist()}\n")
                    log_file.write(f" - Classe prédite: {class_name}\n")
                    log_file.write(f" - Confiance: {conf}\n")
                
                # Tracer les détections sur l'image redimensionnée
                img_result = result.plot()  # Obtenir l'image avec les détections tracées
                
                # Sauvegarder l'image avec les détections
                cv2.imwrite(output_image_path, img_result)
                print(f"Résultats pour l'image {base_name} sauvegardés.")

print("Traitement terminé.")
