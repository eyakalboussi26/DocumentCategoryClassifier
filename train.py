from ultralytics import YOLO
import os

def main():
    try:
        # Chemin vers le répertoire de sauvegarde
        save_dir = 'C:/Users/ekalboussi/OneDrive - ALTEN Group/Bureau/Project_NLP/OCR-and-Classifier-Doc/Classification/Train/safe_directory'
        
        # Créer le répertoire si nécessaire
        os.makedirs(save_dir, exist_ok=True)
        
        # Charger le modèle YOLOv5 
        model = YOLO('yolov5su.pt')
        
        # Entraînement du modèle avec les paramètres spécifiés
        model.train(data='C:/Users/ekalboussi/OneDrive - ALTEN Group/Bureau/Project_NLP/OCR-and-Classifier-Doc/Classification/Train/Dataset',
                   
                    epochs=170, 
                    imgsz=640, 
                    batch=8, 
                    project=save_dir)  # Répertoire de sauvegarde
        
        # Sauvegarde du modèle entraîné sous le nom 'trained-model.pt'
        model.save(os.path.join(save_dir, 'trained-model2.pt'))

    except PermissionError as e:
        print(f"Erreur de permission : {e}")
        print("Vérifiez les permissions d'accès au répertoire ou essayez de changer le répertoire de sauvegarde.")
    except RuntimeError as e:
        print(f"Erreur d'exécution : {e}")
        print("Essayez d'exécuter avec CUDA_LAUNCH_BLOCKING=1 ou vérifiez la configuration du jeu de données et du modèle.")
        
if __name__ == '__main__':
    main()
