Here's a properly structured and professional version of your README:

---

# 🚢 Maritime Vessel Detection using Deep Learning & Selective Search

A deep learning-based system to detect maritime vessels in satellite imagery using **Convolutional Neural Networks (CNN)** and **Selective Search**. The model identifies potential ship regions and classifies them using a custom-trained neural network. Built with TensorFlow, OpenCV, and Streamlit.

---

## 🔍 Project Overview

Ship detection in satellite images is crucial for maritime security, navigation monitoring, and logistics. This project combines **region proposal** techniques with **deep learning** to perform accurate object detection without using complex architectures like Faster R-CNN or YOLO.

### ✨ Key Features
- 📷 Ship detection from aerial/satellite images  
- ⚙️ Region proposals using Selective Search  
- 🧠 Custom CNN to classify ship vs non-ship  
- 📊 Real-time visualization via Streamlit  

---

## 🧠 Core Concepts

| Module              | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| **Selective Search**| Extracts potential regions of interest from satellite imagery               |
| **CNN Classifier**  | Keras-based model trained to detect ships in proposed regions               |
| **OpenCV**          | Used for image preprocessing and bounding box visualization                |
| **Streamlit**       | Interactive web UI for uploading images and viewing detection results       |

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- Streamlit
- NumPy, Matplotlib, scikit-learn

---

## 📁 Project Structure

```
marine-detection/
├── app.py                 # Streamlit frontend
├── train_model.py         # CNN model training
├── utils/
│   ├── prediction.py      # Inference logic
│   └── selective_search.py# Region proposal logic
├── models/                # Saved trained model
├── Dataset/               # Satellite imagery dataset (not included)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/2D-array/marine-detection.git
cd marine-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🧪 Model Performance

> ⚠️ This model is trained on an older dataset and may not generalize well to all modern satellite imagery. This is meant for educational/demonstration purposes.

- **Validation Accuracy**: ~84%  
- **False Positives**: Low  
- **False Negatives**: Moderate  

---

## 📦 Dataset

The dataset used (`shipsnet.json`) is not included in this repo due to size limits.

**Download from:** [Kaggle - ShipsNet](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery)

**After downloading:**
- Place the dataset file inside the `Dataset/` directory.

---

## 📸 Demo

<p align="center">
  <img src="assets/demo1.png" width="600" alt="Detection Demo" />
  <br>
  <em>Ship detection with bounding boxes drawn using OpenCV</em>
</p>

---

## 🎯 Learnings & Challenges

- 📚 Learned how Selective Search extracts region proposals  
- 🤯 Tackled performance issues on large-scale satellite images  
- 🚀 Gained experience deploying deep learning models using Streamlit  

---

## 🙌 Acknowledgements

- Inspired by open-source contributions and Kaggle notebooks  
- Special thanks to GitHub repositories exploring CNN classification strategies  

---

## 📄 License

This project is licensed under the **MIT License**.  
Feel free to fork, use, or contribute for educational/research purposes.

---

## 🔗 Connect with Me

📧 Email: [adityaup0304@gmail.com](mailto:adityaup0304@gmail.com)

---

Happy shipping! 🚢😄

---

Let me know if you want help generating badges, deployment instructions, or adding a GIF demo!
