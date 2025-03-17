# 🚢 Maritime Vessel Detection using Deep Learning & Selective Search

A deep learning-based system to detect maritime vessels in satellite imagery using **Convolutional Neural Networks (CNN)** and **Selective Search**. The model identifies potential ship regions and classifies them with a custom-trained neural network. Built with TensorFlow, OpenCV, and Streamlit.

---

## 🔍 Project Overview

Ship detection in satellite images is crucial for maritime security, navigation monitoring, and logistics. This project combines **region proposal** techniques with **deep learning** to perform accurate object detection without relying on full-fledged object detection architectures like Faster R-CNN or YOLO.

**Key Features:**
- 📷 Ship detection from aerial/satellite images.
- ⚙️ Region proposals using Selective Search.
- 🧠 Custom CNN trained to classify ship vs. non-ship.
- 📊 Real-time inference and visualization using Streamlit.

---

## 🧠 Core Concepts

| Module              | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| **Selective Search**| Extracts potential regions of interest from satellite imagery.              |
| **CNN Classifier**  | A lightweight Keras-based model trained to detect ships in proposed regions.|
| **OpenCV**          | Used for image preprocessing and drawing bounding boxes.                    |
| **Streamlit**       | Interactive web app for uploading images and visualizing detections.        |

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **Streamlit**
- NumPy, Matplotlib, scikit-learn

---

## 📁 Project Structure

Maritime Vessel Detection/ ├── app.py # Streamlit frontend ├── train_model.py # CNN model training ├── predict.py # Prediction logic ├── models/ # Saved model files ├── utils/ # Helper scripts ├── Dataset/ # (Not included - contains large dataset) ├── .gitignore └── README.md

yaml
Copy
Edit

---

## 🚀 Quick Start

1. **Clone the Repository**
```bash
git clone https://github.com/2D-array/marine-detection.git
cd marine-detection
Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Run the Web App
bash
Copy
Edit
streamlit run app.py
🧪 Model Performance
⚠️ The current model is trained on an older dataset and may not generalize well to modern satellite images. It's built for educational/demonstration purposes.

Accuracy: ~84% (validation)
False Positives: Low
False Negatives: Moderate
📦 Dataset
shipsnet.json (Not included in the repository due to GitHub's 100MB file size limit)
Format: RGB images + ship/non-ship labels
You can download it separately from: ShipsNet Kaggle Dataset
Once downloaded, place the file in the Dataset/ directory.

📸 Screenshots
![Ship-Detector](https://github.com/user-attachments/assets/7b9df57f-eff0-487d-853a-09232301871a)


<p align="center"> <img src="assets/demo1.png" width="500" alt="Detection Example"> <br> <em>Ship detection with bounding boxes drawn using OpenCV</em> </p>
🧠 Learnings & Challenges
📚 Learned about region-based detection and how Selective Search works under the hood.
🤯 Faced challenges in managing large image datasets and performance optimization.
🎯 Gained experience deploying CV models in real-time using Streamlit.
🙌 Acknowledgements
Inspired by Kaggle Notebooks and open-source contributions.
Special thanks to various GitHub repositories for CNN classification strategies.
📄 License
This project is licensed under the MIT License.
Feel free to fork, use, or contribute for educational or research purposes.

🔗 Connect with Me
📧 adityaup0304@gmail.com
 

Happy shipping 🚢😄
