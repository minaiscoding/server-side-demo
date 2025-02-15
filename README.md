# **Product Counting API**  

## **Description**  
This API detects and counts products in an image using **YOLOv8** for object detection and **ResNet18** for classification.  
The results are stored in a database and can be accessed through an admin dashboard.  

---

## **Installation & Setup**  

### **1. Clone the repository**  
```bash
git clone <REPO_URL>
cd server-side-demo
```
## **2. Create a virtual environment & install dependencies**  
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

## **3. Configure the database**  
Ensure you have **PostgreSQL** or **SQLite** installed. Update `config.py` if needed:  

```python
SQLALCHEMY_DATABASE_URI = "sqlite:///database.db"  # Or a PostgreSQL/MySQL connection
```
Then initialize the database:
```bash
flask db init
flask db migrate -m "Initial migration"
flask db upgrade
```

## **4. Run the Flask server**  
```bash
python app.py
```
The API will be available at:
http://127.0.0.1:5000/
### **Available Routes**  


### **1. Check API status**  
```http
GET /
```
Response:
```json
{"message": "Flask API is running"}

```
### **2. Upload an image & count products**  
```http
POST /count_products
```
Parameters:

image: The image file to process.
Response:
```json
{
    "bottle_ramy": 3,
    "can_ramy": 2,
    "pack_rouiba": 1
}
```
The results are stored in the database with a timestamp, sales point, and user ID.
## **Admin Dashboard**  
A web-based dashboard allows users to view the results in tables and charts.  
Access it via:  
```http
GET /dashboard
```
## **Technologies Used**  
- **Flask** (Backend API)  
- **YOLOv8 & ResNet18** (Computer Vision)  
- **SQLite/PostgreSQL** (Database)  
- **Bootstrap & Chart.js** (Web Dashboard)  
- **Flutter** (Mobile App)  