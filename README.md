# Telecom Churn Prediction

---

###  A smart Streamlit app that helps predict whether a telecom customer is likely to stay or leave, explains the main reasons behind that prediction, and suggests practical retention actions.

This project was built to make customer risk easier to understand for non-technical users, so the app uses simple language like Will Stay and Will Leave instead of technical terms.

---

## What this project does?
* Predicts whether a customer will stay or leave.

* Shows the risk level in a simple visual format.

* Explains the top reasons behind the prediction.

* Suggests retention strategies based on the customer profile.

* Lets users load sample customer profiles quickly.

* Provides a downloadable prediction report.

---

## Why I built this?

* I wanted to build a project that feels more like a real product than a classroom model.

* __Most ML projects stop at prediction. I wanted to go one step further by:__

    * making the output easy to understand,

    * designing a polished UI,

    * and turning the model into something that could actually help a business team make decisions.

---

## Features
* Clean and modern Streamlit UI.

* Simple customer status prediction.

* Probability gauge chart.

* Sample customer buttons for quick testing.

* Downloadable CSV report.

---

## Explanation of important factors affecting the prediction.

### Retention strategy suggestions.

#### Tech stack :-

* __Frontend/UI:__ Streamlit

* __Language:__ Python

* __Data handling:__ Pandas, NumPy

* __Machine learning:__ scikit-learn

* __Model saving:__ Joblib

* __Charts:__ Plotly

---

## How it works:

* User enters customer details.

* The trained model predicts whether the customer will stay or leave.

* The app calculates probability and risk level.

* It shows the main reasons behind the result.

* It suggests actions to reduce customer loss.

---

### Sample use cases:-

__This app can be useful for:__

* telecom companies,

* customer success teams,

* business analysts,

* and recruiters evaluating applied ML projects.

---

## Project structure:

```bash
telecom-churn-project/
├── app.py
├── train_model.py
├── requirements.txt
├── data/
├── models/
│   ├── best_customer_leaving_model.joblib
│   └── model_metadata.joblib
└── README.md
```
---

## Installation:-
__Clone the repository:__

```bash
git clone https://github.com/lakshmidharn2007-code/Telecom-Churn-Project.git
cd Telecom-Churn-Project
```
__Create and activate a virtual environment:__

```bash
python -m venv venv
venv\Scripts\activate
```
__Install dependencies:__

```bash
pip install -r requirements.txt
Run the app
```
__Start the Streamlit app with:__

```bash
streamlit run app.py
Re-train the model
```
__If you want to train the model again:__

```bash
python train_model.py
```
---

## Screenshots

<img width="1912" height="831" alt="image" src="https://github.com/user-attachments/assets/7d372627-ade5-4780-bb55-2437ed4ff3c4" />

<img width="1908" height="855" alt="image-1" src="https://github.com/user-attachments/assets/20cdbcea-abd2-48a0-a84d-8e5c4ca68765" />

---

### Deployment & Documentation 
* This app is designed to run on Streamlit Community Cloud.

* [Documentation](https://www.linkedin.com/in/lakshmidharn-231107u/)

---

## Future improvements
* Add login for recruiters.

* Save prediction history.

* Add feature importance visualization.

* Add email alert for high-risk customers.

* Improve explanation with SHAP or LIME.

* Add dark/light theme toggle.

---

### My learning from this project
__This project helped me learn how to:__

* build an end-to-end ML product,

* make ML results understandable,

* design a better user experience,

* and deploy a real app for others to use.

---

### Contact:

Built by [LAKSHMIDHAR](https://www.linkedin.com/in/lakshmidharn-231107u/).

