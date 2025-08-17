# Predicting Student At-Risk Status Using Machine Learning and Socioeconomic Factors

## Overview

This project aims to develop a predictive model for identifying students at risk of academic failure.  The model utilizes machine learning techniques and incorporates various factors, including academic performance data, socioeconomic indicators, and attendance records.  The analysis provides insights into the key factors contributing to academic risk and allows for the development of proactive intervention strategies to support at-risk students.

## Technologies Used

* Python 3.x
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn


## How to Run

1. **Install Dependencies:**  Ensure you have Python 3.x installed.  Then, install the required Python libraries using pip:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Script:** Execute the main script using:

   ```bash
   python main.py
   ```

## Example Output

The script will print a summary of the analysis to the console, including details about the chosen model, its performance metrics (e.g., accuracy, precision, recall, F1-score), and feature importance.  Additionally, the script will generate several visualizations, including plots that illustrate the relationships between various factors and the probability of at-risk status. These plots will be saved as PNG files in the `output` directory (you may need to create this directory if it doesn't exist).  The specific names of the output files will vary depending on the generated figures.  For example, you might find files like `feature_importance.png` or `model_performance.png`.


## Data

The project requires a dataset containing student information.  The dataset should include features such as GPA, test scores, attendance rate, socioeconomic indicators (e.g., family income, parental education), and a target variable indicating whether a student is at risk (e.g., binary classification: 0 for not at risk, 1 for at risk).  The specific format and location of the data file are assumed to be handled within the `main.py` script.  Consider adding a `data` folder to your repository to store the data (remember to avoid committing large datasets directly to GitHub).

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

[Specify your license here, e.g., MIT License]