📊 Employee Salary Prediction using Machine Learning

🚀 Project Overview

This project is a Machine Learning-based web application that predicts whether an employee's annual salary is greater than 50K or less than/equal to 50K, based on features like age, workclass, education, occupation, marital status, hours worked per week, etc.

⚙️ Features

📊 Predicts salary based on employee profile inputs.

🤖 Uses a Machine Learning regression model trained on real-world-like employee datasets.

🖥️ Web-based interface built with Flask and HTML templates.

🔄 Model saved with joblib/pickle for easy reuse.

🌟 Users can interact with a web interface where they input employee details, and the application will predict the salary category (>50K or <=50K).



🛠️ Technologies & Libraries Used
Programming Language

Python 3.x

Machine Learning Libraries

Scikit-learn → For ML model training and evaluation

NumPy → For numerical operations

Pandas → For data manipulation and preprocessing

Joblib → For saving/loading trained ML models and encoders

Web Framework

Flask → For creating the web application and REST API

Flask-CORS → To handle Cross-Origin Resource Sharing

Frontend (inside templates/ folder)

HTML5 → Page structure

CSS3 → Styling

JavaScript → Input handling and sending requests



📊 Dataset

--> The dataset used contains employee-related features such as:

--> Experience (Years of Experience)

--> Education Level

--> Job Role / Position

--> Work Class (Private/Government/Self-employed, etc.)

--> Age

--> Gender

--> Other categorical/continuous variables

The target variable is the salary of the employee.



🧠 Machine Learning Model

Algorithm Used: Gradient Boosting ( Ensemble Algorithm )

Preprocessing: Label Encoding, One-Hot Encoding, and Standard Scaling

The trained model generalizes well and can make accurate salary predictions for unseen employee data.

🚀 Page Visiting link: https://employee-salary-prediction-x25w.onrender.com/

🤝 Contributing

Contributions are welcome!
Fork this repo, improve it, and submit a PR.


