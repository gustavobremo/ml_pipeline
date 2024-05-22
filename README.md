# Machine Learning Portfolio

This repository showcases various machine learning scripts, and resources that I have developed to demonstrate its application in the field of data science and machine learning.

## Overview

In this repository, you will find:

- **Data Analysis Scripts**: Python scripts for exploratory data analysis (EDA), feature engineering, and data preprocessing.
- **Machine Learning Models**: Implementation of various machine learning algorithms for classification, regression, and clustering tasks.
- **Projects**: End-to-end machine learning projects demonstrating the entire data science pipeline from data collection to model deployment.
- **Documentation**: Documentation and resources to help understand and use the scripts and projects effectively.

## Data Analysis Folder

The `data_analysis` folder contains Python scripts for performing exploratory data analysis (EDA) on datasets. Below is a brief overview of the functions available in the `data_analysis.py` script:

### Functions:

- **read_data(filename, separator="")**: Reads data from a CSV file.
- **describe_dataframe(data)**: Generates a textual summary of the dataset.
- **plot_numerical_distributions(data, save_path)**: Plots boxplots to visualize the distributions of numerical features.
- **plot_categorical(data, save_path)**: Creates pie charts to visualize the distribution of categorical data.
- **plot_correlation_matrix(data, save_path, threshold=False)**: Plots a heatmap of the correlation matrix for numerical data.
- **plot_feature_correlation(dataframe, target_variable, save_path)**: Plots pair scatter plots between numerical features and the target variable.
- **normality_check(data, target_variable, save_path)**: Visualizes the distribution of numerical variables and checks for normality.
- **plot_missing_values(data, data_filename)**: Plots the total count of missing values for each column in the dataset.
- **data_analysis_workflow(data_file_name, target_variable=False, sep=False)**: Performs an EDA workflow including data loading, statistics plotting, and correlation analysis.

Each function is documented with details about its purpose, parameters, and return values, making it easy to understand and use for exploratory data analysis tasks.
