# Heart Disease Prediction using Machine Learning

This project demonstrates a classic data science workflow to build and evaluate a predictive model for classifying the presence of heart disease. Using a clinical dataset from the UCI Machine Learning Repository, this project performs in-depth Exploratory Data Analysis (EDA), data preprocessing, and trains several classification models to identify the most effective one for this task.

The final model provides a functional proof-of-concept for a potential clinical decision support tool, achieving **98% accuracy** in its predictions.

---

## Project Workflow

The project is structured as follows:

1.  **Exploratory Data Analysis (EDA):** A thorough investigation of the dataset to understand feature distributions, correlations between variables, and initial insights into factors influencing heart disease.
2.  **Data Preprocessing:** Cleaning the data and scaling numerical features using `StandardScaler` to ensure all variables contribute equally to the model's performance.
3.  **Model Training & Comparison:** Several machine learning models were trained and evaluated, including:
    * K-Nearest Neighbors (KNN)
    * Random Forest
    * Logistic Regression
4.  **Hyperparameter Tuning:** The best-performing model, Logistic Regression, was further optimized using `GridSearchCV` to find the ideal hyperparameters, pushing its accuracy from 78% to 98%.
5.  **Performance Evaluation:** The final model was rigorously evaluated based on its accuracy, a detailed classification report (precision, recall, F1-score), and a confusion matrix.

---

## Tech Stack

* **Core Libraries:** Python 3
* **Data Analysis & Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn
* **Environment:** Jupyter Notebook
---
## How to Reproduce

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[your-username]/Heart-Disease-Prediction.git
    cd Heart-Disease-Prediction
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Launch the Jupyter Notebook:**
    ```bash
    jupyter notebook PredictingHeartDisease_ModelTrain.ipynb
    ```

4.  **Run the cells:**
    Execute the cells sequentially to perform the analysis and train the model. The dataset (`heart.csv`) is included in this repository.
