# Diabetes Prediction MLOps Project

## Project Description
This project predicts diabetes using machine learning models and deploys the best model using FastAPI.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- FastAPI
- Matplotlib
- Seaborn

## Setup Instructions

### Create Virtual Environment
python -m venv venv

### Activate Environment
venv\Scripts\activate

### Install Dependencies
pip install -r requirements.txt

## Run API
uvicorn app:app --reload

## API Documentation
http://127.0.0.1:8000/docs

it show N if not diabetic. And Y if it diabetic.
## curl commands test
The folder of curl command is also added. To show the performance of the model.It peridicate accurately.