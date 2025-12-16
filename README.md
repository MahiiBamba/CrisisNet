# CrisisNet

**CrisisNet: An AI-Powered Disaster Impact Prediction and Resource Allocation System**

---

## 🌟 Project Overview

CrisisNet is a **deep learning-based system** designed to **analyze post-disaster images** and predict both the **current and future disaster risk**. It provides **actionable recommendations** for disaster management authorities, NGOs, and emergency response teams.

The project combines:

- **CNN (Convolutional Neural Network)** to assess current disaster damage from images.
- **LSTM (Recurrent Neural Network)** to predict future risk based on historical patterns.
- A **decision policy** to suggest resource allocation and priority actions.

---

## 🧩 Features

- **Current Disaster Classification:** Determines if damage is High, Medium, or Low.  
- **Future Risk Prediction:** Estimates the risk in the coming hours/days.  
- **Decision Recommendations:** Suggests deployment of resources like medical teams, shelters, or monitoring.  
- **Interactive Web Interface:** Users can upload disaster images via a **Flask web application**.  
- **Professional, Dashboard-style UI:** Dynamic and responsive layout with disaster-themed backgrounds.  

---

## ⚙️ Installation

**Python version required:** 3.10  

1. Clone the repository:

```bash
git clone https://github.com/MahiiBamba/CrisisNet.git
cd CrisisNet
```
2. Create a virtual environment:
```bash
python3.10 -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Ensure the CNN and LSTM models are in the root directory.

🚀 Usage

Run the Flask application:
```bash
python app.py
```

Open your browser and navigate to:
```bash
http://127.0.0.1:5000
```

Upload a disaster-related image or data.

View current disaster severity, predicted future risk, and recommended action on the results page.

## 📊 AI Models
- CNN (ResNet50)	Disaster severity classification	Input: image → Output: High / Medium / Low
- LSTM	Future risk prediction	Input: sequence of risk scores → Output: predicted risk

#### Risk Mapping:

- High → 1.0

- Medium → 0.6

- Low → 0.2

