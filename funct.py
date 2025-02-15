import os
import torch
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict
from torchvision import transforms, models
from ultralytics import YOLO

# 📍 CONFIGURATION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🏷️ Chargement du modèle YOLOv8
yolo_model = YOLO("yolov8n.pt")

# 🏷️ Chargement du modèle de classification (ex: ResNet18)
MODEL_PATH = "resnet18_classification.pth"
classification_model = models.resnet18(weights=True)

# ⚠️ Modifier selon le nombre de classes réelles
num_classes = 5
classification_model.fc = torch.nn.Linear(classification_model.fc.in_features, num_classes)
classification_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
classification_model.to(device)
classification_model.eval()

# 🔄 Prétraitement des images pour la classification
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 🏷️ Liste des classes (doit correspondre aux classes du modèle de classification)
class_names = ["autre produit", "bouteille_ramy", "bouteille_rouiba", "cannette_ramy", "pack_rouiba"]  # ⚠️ À personnaliser
allowed_classes = [39, 41] 
# 📌 Fonction principale
def count_products(image_path, detected_products_dir="detected_products"):
    """
    📌 Détecte et compte les bouteilles et canettes dans une image.
    ➡️ Renvoie un dictionnaire avec le nombre d'occurrences par classe.

    :param image_path: Chemin de l'image d'entrée
    :param detected_products_dir: Dossier où sauvegarder les crops des produits détectés
    :return: Dictionnaire {nom_classe: nombre_detecté}
    """
    # Assurer que le dossier de stockage des crops existe
    os.makedirs(detected_products_dir, exist_ok=True)

    # 📌 1. Détection des objets avec YOLO
    image = cv2.imread(image_path)
    results = yolo_model(image_path)

    # 📌 2. Initialiser un dictionnaire pour compter les produits
    product_counts = defaultdict(int)

    # 📌 3. Extraction et classification des produits détectés
    for i, box in enumerate(results[0].boxes.xyxy):
        class_id = int(results[0].boxes.cls[i])  # ID de la classe détectée

        # 🚨 Filtrer les objets non souhaités (seulement bouteilles et canettes)
        if class_id not in allowed_classes:
            print(f"❌ Produit {i} ignoré (Classe {class_id} non prise en charge)")
            continue

        # 📌 Si l'objet est valide, extraire et sauvegarder
        x1, y1, x2, y2 = map(int, box)
        cropped = image[y1:y2, x1:x2]

        # 📌 Sauvegarde du crop
        crop_path = os.path.join(detected_products_dir, f"product_{i}.jpg")
        cv2.imwrite(crop_path, cropped)

        # 📌 4. Classification du produit détecté
        image_pil = Image.open(crop_path).convert('RGB')
        image_tensor = transform(image_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = classification_model(image_tensor)
            _, predicted = output.max(1)

        predicted_class = class_names[predicted.item()]
        product_counts[predicted_class] += 1  # Incrémenter le compteur

        print(f"✅ Produit {i} détecté (Classe COCO {class_id}) et classé comme {predicted_class}")

    # 📊 5. Résumé du comptage des produits
    print("\n📊 RÉSUMÉ DU COMPTAGE :")
    for class_name, count in product_counts.items():
        print(f"🛒 {class_name} : {count} produits")

    return dict(product_counts)
if __name__ == "__main__":
    image_path = "ramyRayon.jpg"  # Remplace par ton chemin d'image
    results = count_products(image_path)
    print("\n📌 Résultat final :", results)
