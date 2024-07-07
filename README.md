# Best Feature Selection of Breast Cancer Diagnosis using Light Gradient Boost (LGBoost) Approach

### Under the guidance of MRS. VEDATRAYEE CHATTERJEE for the partial fulfillment of degree of Bachelor of Technology.
### ASANSOL ENGINEERING COLLEGE
### June 2024

## Abstract

Breast cancer, a life-threatening disease affecting millions worldwide, poses significant challenges due to its time-consuming manual determination process, potential risks, and human errors. It is a condition where cells of the breast develop unnaturally and uncontrollably, resulting in a mass called a tumor. If lumps in the breast are not addressed,they can spread to other regions of the body, including the bones, liver, and lungs. Early diagnosis is crucial for effective treatment and improved patient outcomes. We focused on employing machine learning models to achieve quick identification of breast cancer tumors as benign or malignant. The primary objective is to develop a decision-making visualization pattern using swarm plots and heat maps. To accomplish this, we have utilized the Light GBM (Gradient Boosting Machine) algorithm and evaluated the model performance.

## Introduction

Today, Breast cancer is affecting individuals, particularly women. According to the World Health Organization (WHO). It's a leading cause of female mortality. Around a million women succumb to breast cancer annually with India's fatality rate at 13.92%. The prevalence is higher in Australia, Europe and the US, while Malaysia observes later-stage presentations. Regular screening is vital due to asymptomatic cases. Early detection aids treatment and survival. Contributing factors include family history, obesity, radiation exposure, and genetics. Recently discovered, breast cancer is categorized as malignant or benign. Analyzing tumor characteristics helps differentiate them. Benign tumors are low-risk, while malignant ones spread to neighboring tissues and the body. Artificial Intelligence (AI) is being employed to classify breast cancer. AI algorithms train on datasets to label tumors as 1 for benign or 0 for malignant.

## Motivation

The initial aim of this study is to examine breast cancer data derived from a diagnostic dataset comprising 33 feature columns and approximately 569 rows. The primary goal is to identify common characteristics in these groups that distinguish benign cases from malignant ones effectively. Subsequently, we plan to generate a heatmap visualization to identify and eliminate redundant features from the dataset. Finally, our ultimate objective is to create a machine learning model that enables users to classify breast cancer cases as either benign or malignant accurately. By accomplishing these objectives, we hope to enhance the diagnostic process and contribute to more efficient and precise breast cancer classification.

Our project aims to address challenges and propose solutions to enhance accuracy in breast cancer classification. Accuracy is a critical factor, as an imprecise model can lead to suboptimal outcomes. The report primarily centers around improving the accuracy of various algorithms, namely Logistic Regression, Gradient Boosting Algorithm, Random Forest Algorithm (Octaviani and Z Rustam et al.) [4], XG Boost Algorithm, and Light GBM Algorithm. The objective is to achieve the highest possible accuracy for the model by Light GBM algorithms. By tackling accuracy-related issues, we aspire to provide more reliable and effective breast cancer classification results.

## Project Description

Currently, India reports approximately 178,000 cases of breast cancer. However, manually determining cancer in these cases is an arduous and time-consuming process, often leading to delays and the possibility of human errors. To address this issue, we aim to develop a predictive model that can efficiently classify breast tumors as either malignant or benign using Machine Learning techniques. Our approach involves analyzing the correlation between various features, eliminating redundant data, and ultimately creating a highly accurate model. By leveraging these advanced technologies, we strive to enhance the early detection and diagnosis of breast cancer, which can significantly improve patient outcomes.

## Proposed Model / Approach

The approach proposed is Light Gradient Boosting (Light GBM Approach).
 
A novel approach in breast cancer detection has been introduced utilizing the Light Gradient Boost machine learning technique. This innovative method aims to transform initially weak learners into robust ones, thereby achieving enhanced accuracy in breast cancer detection. Unlike the conventional employment of weak learners as standalone classifiers, this technique leverages a boosting ensemble to achieve heightened classification accuracy. In this approach, the weak learners are harnessed as classifiers, which alone may not yield optimal classification accuracy. However, the concept of a strong learner emerges through the ensemble of these weak classifiers. This ensemble-based boosting technique is rooted in tree-based classification.

<p align="center">
  <img src="https://github.com/arnab-maitra/Best-Feature-Selection-of-Breast-Cancer-Diagnosis-using-Light-GBM-Approach/assets/88264132/b043d9ca-c763-4a6a-bf97-865059d6a99f" alt="Flowchart of LightGBM Algorithm" />
  <br><b>Flowchart of LightGBM Algorithm.</b>

Notably, the Light Gradient Boost machine learning technique molds the decision tree classifier into a unique weak learner structure, characterized by a vertical orientation. This innovative design, termed the "Leaf-wise Decision Tree Algorithm," showcases its distinctiveness in minimizing training loss compared to alternative algorithms. Through these advancements, the Light Gradient Boost technique demonstrates its potential to significantly improve breast cancer detection accuracy, thus offering promising avenues for enhanced medical diagnostics.

## Steps of Execution

 1. Importing Required Libraries: NumPy for numerical operations, pandas for data manipulation, Matplotlib and Seaborn for data 
    visualization, and Light GBM for machine learning.
 2. Loading the Data: The dataset is loaded from a CSV file named 'data.csv' into a
    pandas DataFrame called data.
 3. Inspecting the Data:
 - Display the first few rows (head), last few rows (tail), total number of elements (size), and the shape of the DataFrame (shape).
 - The column names are extracted and printed.
 - The data types of each column are inspected.
 4. Separating Target Variable: Target Variable: The target variable y is extracted from the diagnosis column, indicating if the diagnosis 
    is Malignant (M) or Benign (B).
 5. Preparing Feature Variables: Columns Unnamed: 32, id, and diagnosis are dropped from the dataset, and the remaining columns are stored 
    in x.
 6. Visualizing the Class Labels:
 - A count plot is created to visualize the distribution of the diagnosis classes.
 - The counts of each class (Benign and Malignant) are printed.
 7. Plotting of Correlation Heatmap: A heatmap showing the correlation matrix of the features in x is plotted.
 8. Dropping Highly Correlated Features: Highly correlated features are identified and dropped from the dataset to reduce redundancy, 
    resulting in a new DataFrame x_1.
 9. Correlation Heatmap for Reduced Features: A heatmap for the reduced feature set x_1 is plotted.
 10. Splitting the Data:
 - Train-Test Split: The data is split into training and testing sets, with 30% of the data reserved for testing. The random_state ensures reproducibility.
 11. Initializing and Training the Model:
 - Model Initialization: A Light GBM classifier is initialized.
 - Model Training: The model is trained on the training data (x_train and y_train).
 12. Making Predictions: The trained model is used to predict the class labels for the test set (x_test).
 13. Evaluating the Model:
 - Accuracy Calculation: The accuracy of the model is calculated by comparing the predicted labels (y_pred) with the true labels (y_test).
 - Output Accuracy: The accuracy is printed.

## Experimental Setup

The project aims to discern the disparities between benign and malignant breast cancer cases. Initially, breast cancer data is gathered from a diagnostic dataset. The dataset is then preprocessed, and any missing values are handled by removal. Next, we utilize swarm plots to visualize and compare the features, assessing if there are distinct patterns between benign and malignant cases. Outliers in the features are identified and removed to ensure data integrity. Following the outlier removal, the preprocessed data is split into training and testing datasets. This methodology allows us to gain valuable insights into the characteristics that differentiate these types of cancer and create a robust predictive tool to aid in accurate diagnosis. For this project, we utilized a diagnostic dataset (The dataset has been collected from Kaggle repository) containing 569 rows and 33 columns. These 33 parameters were chosen as the basis for our analysis. These attributes play a vital role in producing visualization patterns, making it easier to generate heat maps for feature visualization.

<p align="center">
  <img src="https://github.com/arnab-maitra/Best-Feature-Selection-of-Breast-Cancer-Diagnosis-using-Light-GBM-Approach/assets/88264132/ef5ce34e-8c55-4ebf-b0d8-7234b539aa75" alt="Original Dataset collected from Kaggle Repository (generated in the form of a Heatmap)." />
  <br><b>Original Dataset collected from Kaggle Repository (generated in the form of a Heatmap).</b>

Once the dataset (from Kaggle’s repository) is imported using the Panda’s library, it becomes crucial to check for the presence of any missing values. The data cleaning process involves eliminating entire rows that contain any missing values. This step ensures that subsequent tasks, such as visualization, can be carried out effectively with high accuracy. Heat maps are then employed to identify and remove outliers.

<p align="center">
  <img src="https://github.com/arnab-maitra/Best-Feature-Selection-of-Breast-Cancer-Diagnosis-using-Light-GBM-Approach/assets/88264132/8f59607f-df26-4206-9868-a0885b1bf7ea" alt="Swarm plot graphs were plotted" />
  <br><b>Swarm plot graphs were plotted.</b>

The data is split into training and testing sets, with 30% of the data reserved for testing (and 70% for training), ensuring reproducibility by using a specified random_state. Following the split, a LightGBM classifier is initialized. The model is then trained on the training dataset, which includes both the feature set (x_train) and the corresponding labels (y_train). Once the model is trained, it is used to predict the class labels for the test dataset (x_test). To evaluate the model's performance, the accuracy is calculated by comparing the predicted labels (y_pred) to the true labels (y_test), and the resulting accuracy is printed to provide a measure of the model's predictive performance.

## Simulation Results

Feature selection was applied to the initial dataset, which originally comprised 33 features. Following the feature selection process and data cleaning, the refined dataset has been reduced to 8 features.

<p align="center">
  <img src="https://github.com/arnab-maitra/Best-Feature-Selection-of-Breast-Cancer-Diagnosis-using-Light-GBM-Approach/assets/88264132/80184978-34f4-4ef1-a48c-2d4a0680622a" alt="Refined Dataset after Removal of Outliers" />
  <br><b>Refined Dataset after Removal of Outliers.</b>

After completing the preprocessing steps and model training, the model achieved an accuracy of 97.07%.

```
Accuracy: 0.9707602339181286

```

 ## Discussion

In our dataset preprocessing on Google Colab, we began by importing a Kaggle dataset using Pandas. Initially, it had 569 rows and 33 columns, but after a careful feature selection process, we retained only 8 key features to streamline analysis and enhance model efficiency.

We addressed missing or inconsistent data using Pandas functions, applying imputation techniques or removing entries as necessary. Numerical features were normalized through Min-Max scaling or Z-score normalization to ensure comparability. Categorical variables were encoded using one-hot encoding for compatibility with machine learning models.

Swarm plots were utilized to visualize feature patterns and identify outliers, which were subsequently removed. We then split the preprocessed data into training and testing sets.

We trained the data using the Light GBM algorithm to classify breast cancer cases as benign or malignant. Our preprocessing and model training pipeline resulted in an impressive model accuracy of 97.07%, highlighting the effectiveness of our approach in achieving reliable breast cancer diagnosis.

 ## Future Work

We have achieved the highest possible accuracy of 97.07% in diagnosing breast cancer using the Light GBM algorithm. Consequently, there are currently no further improvements or future work planned for this project.

## Conclusion

The results of our study reveal that the Light GBM algorithm proves to be highly efficient and straightforward to implement when working with a diagnostic dataset. After removing the outliers, we found that 8 features remained, significantly contributing to the overall accuracy of the model. Light GBM achieved the accuracy of 97.07%. Hopefully, this will aid individuals in receiving early breast cancer treatment and proactively manage their lives.


## Contributors to this Project

1. Arnab Maitra
2. Subhadeep Puitandi
3. Ankita Mukherjee
4. Arijita Roy
