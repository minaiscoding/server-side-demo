from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from datetime import datetime
import threading
import random

from funct import count_products  # Assurez-vous que la fonction est bien importée

app = Flask(__name__)
CORS(app)  # Active CORS pour Flutter

# 📌 Configuration de la base de données SQLite
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///products.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# 📌 Modèle de base de données
class ProductDetection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_name = db.Column(db.String(255), nullable=False)
    product_class = db.Column(db.String(255), nullable=False)
    product_count = db.Column(db.Integer, nullable=False)
    sale_point = db.Column(db.String(255), nullable=False)  
    user = db.Column(db.String(255), nullable=False)  
    received_at = db.Column(db.DateTime, default=datetime.utcnow)

# 📌 Création des tables au démarrage
with app.app_context():
    db.create_all()

# 📌 Fonction pour générer un point de vente et un utilisateur aléatoire
def generate_dummy_values():
    sale_points = ["Supermarché A", "Épicerie B", "Marché C", "Hyper U", "Shop D"]
    users = ["User1", "User2", "User3", "User4", "User5"]
    return random.choice(sale_points), random.choice(users)

# 📌 Fonction pour sauvegarder en base avec contexte Flask
def save_to_db(image_name, product_counts):
    with app.app_context():
        sale_point, user = generate_dummy_values()  
        for product_class, count in product_counts.items():
            entry = ProductDetection(
                image_name=image_name, 
                product_class=product_class, 
                product_count=count, 
                sale_point=sale_point, 
                user=user
            )
            db.session.add(entry)
        db.session.commit()

# 📌 Route principale (Test API)
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask API is running "}), 200

# 📌 Endpoint API pour le comptage des produits
@app.route("/count_products", methods=["POST"])
def upload_and_count():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    filename = secure_filename(image_file.filename)
    image_path = f"content/{filename}"
    image_file.save(image_path)

    # 📌 Appelle la fonction de comptage des produits
    result = count_products(image_path)  

    # 📌 Enregistrement en base dans un thread
    threading.Thread(target=save_to_db, args=(filename, result)).start()

    return jsonify(result)  

# 📌 Route du tableau de bord (Affichage des données en base)
@app.route("/dashboard")
def dashboard():
    detections = ProductDetection.query.order_by(ProductDetection.received_at.desc()).all()

    # 📊 Préparer les données pour le graphique en anneau
    product_data = {}
    for detection in detections:
        if detection.product_class in product_data:
            product_data[detection.product_class] += detection.product_count
        else:
            product_data[detection.product_class] = detection.product_count

    return render_template("dashboard.html", detections=detections, product_data=product_data)

# 📌 Lancer le serveur Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
