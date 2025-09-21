# 🏦 Customer Churn Prediction Web App

##  Problem Statement  
Predict whether a bank customer is likely to exit based on demographic and account activity data. This helps the bank proactively retain high-risk customers.

##  Project Overview  
An end-to-end machine learning pipeline using an Artificial Neural Network (ANN) to classify customer churn, integrated into a Streamlit web app and deployed on Streamlit Cloud.

##  Approach  
- **Data Preprocessing**  
  - Encode categorical features (Gender, Geography)  
  - Standardize numerical variables  
- **Modeling**  
  - ANN built with TensorFlow and Keras  
  - Dropout layers to prevent overfitting  
  - Optimized using binary crossentropy loss  
- **Serialization**  
  - Save model weights (`.h5`) and preprocessing objects (`.pkl`) for reuse  
- **Web Integration**  
  - Streamlit app for real-time predictions  
  - User-friendly UI with input widgets and result display  
- **Deployment**  
  - Hosted on Streamlit Cloud for public access  

## Project Structure  
```
churn-prediction/
├── app.py                 # Streamlit frontend
├── model.h5               # Trained ANN model
├── scaler.pkl             # StandardScaler object
├── encoder_gender.pkl     # LabelEncoder for gender
├── ohe_geo.pkl            # OneHotEncoder for geography
├── requirements.txt       # Dependencies
└── README.md              # Project overview
```

##  Features  
- Real-time churn prediction  
- Clean UI with intuitive inputs  
- Reusable model pipeline  
- Cloud-hosted for easy sharing  

## 🌐 Live Demo  
[Streamlit App Link](https://churn-modelling-using-ann-and-deploying-in-app-mbhf8jtgcmi2lx7.streamlit.app/) 

## 🛠 Tech Stack  
- Python, Pandas, NumPy  
- TensorFlow, Keras  
- Scikit-learn  
- Streamlit  

## 📬 Contact  
For questions or collaboration, reach out via [LinkedIn](https://www.linkedin.com/in/kavanakpadaki/) 
