---
title: MLPF - Machine Learning Pipeline Framework
layout: home
nav_order: 3
parent: Projects
---

[mlpf github]: https://github.com/adamloec/mlpf

# MLPF - Machine Learning Pipeline Framework
{: .no_toc }
{: .fs-9 }
Machine Learning Pipeline Framework, created as a brief learning experience of applying basic ML algorithms to basic datasets.
{: .fs-6 .fw-300 }

[Github][mlpf github]{: .btn .fs-5 .mb-4 .mb-md-0 }

## Table of contents
{: .no_toc .text-delta }
1. TOC
{:toc}

---

## About

I created this project in the process of re-learning the basics to machine learning. This repository showcases the usage of a handful of common ML concepts and models.

### Readings, resources, and design inspirations

- Machine Learning Specialization coursework - Andrew Ng

---

## Goals

1. Further solidify my applicational knowledge of basic ML concepts.
2. Develop a pipeline with feature engineering.
3. Future: Add matplot visualization to view results comparing models?

---

## Machine Learning Concepts

### Supervised Learning

Supervised learning is a fundamental machine learning paradigm where algorithms learn from labeled training data to make predictions on new, unseen data. In this approach, the algorithm is trained on a dataset where the desired output (target variable) is known. Think of it like learning with a teacher - for each example, you know the correct answer. The algorithm learns to map input features to output labels by identifying patterns in the training data. Common applications include email spam detection, image classification, and price prediction.

#### Classification vs. Regression

While both classification and regression fall under supervised learning, they serve different purposes. Classification models predict discrete categories or classes - like determining whether an email is spam or not spam, or identifying the species of a flower. Regression models, on the other hand, predict continuous numerical values - such as house prices, temperature, or stock prices. The key difference lies in the output: classification provides categorical predictions (often expressed as probabilities for each class), while regression provides numerical predictions along a continuous spectrum.

### Feature Engineering

Feature engineering is the critical art and science of transforming raw data into meaningful features that better represent the underlying problem to predictive models. This process requires domain expertise and creative problem-solving. Common techniques include:

- Scaling numerical features to a standard range
- Encoding categorical variables into numerical formats
- Creating interaction terms between existing features
- Extracting meaningful components from complex data types (like text or dates)
- Handling missing values through imputation 

### Data Splitting

Data splitting is a crucial practice in machine learning where the available dataset is divided into separate sets for training, validation, and testing. The typical split is:

- Training set (60-80%): Used to train the model
- Validation set (10-20%): Used to tune hyperparameters and evaluate performance during development
- Test set (10-20%): Used only for final evaluation

### Model Accuracy

Model accuracy refers to how well a machine learning model performs its predictions, but it's just one of many important metrics. While accuracy measures the proportion of correct predictions, it can be misleading, especially with imbalanced datasets. Other crucial metrics include:

- Precision: Accuracy of positive predictions
- Recall: Ability to find all positive instances
- F1-Score: Harmonic mean of precision and recall
- ROC-AUC: Model's ability to distinguish between classes

### Logistic Regression

Despite its name, logistic regression is a classification algorithm that predicts the probability of an instance belonging to a particular class. It uses a logistic function (sigmoid) to transform linear predictions into probability scores between 0 and 1. Key advantages include:

- Simplicity and interpretability
- Fast training and prediction
- Probabilistic output
- Low computational requirements


### Decision Trees

Decision trees are intuitive models that make predictions by learning decision rules derived from data features. They split the data into smaller subsets based on the most important features until reaching leaf nodes with final predictions. Benefits include:

- Easy to understand and interpret
- Can handle both numerical and categorical data
- Requires little data preparation
- Automatically handles feature interactions

### Random Forest

Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Its advantages include:

- High accuracy and robust performance
- Resistance to overfitting
- Ability to handle large datasets with higher dimensionality
- Built-in feature importance estimation
- Minimal hyperparameter tuning required

---

## Development

The ML Pipeline implementation consists of a modular framework designed to streamline the machine learning workflow, focusing on model training, feature engineering, and data preprocessing. The core component is the MLPipeline class which orchestrates the entire machine learning process.

### Dependencies

- Python
- scikit-learn
- pandas
- mlflow

### Pipeline Architecture

{% highlight Python %}
MODELS = {
    'logistic': LogisticRegression(),
    'decision_tree': DecisionTreeClassifier(),
    'random_forest': RandomForestClassifier()
}
{% endhighlight %}

The Machine Learning Pipeline Framework (MLPF) serves as a streamlined interface for training and evaluating machine learning models. At its heart, the MLPipeline class orchestrates the entire workflow, from data preprocessing to model training. The pipeline is designed with flexibility in mind, supporting multiple model types including Logistic Regression, Decision Trees, and Random Forests through a simple registry system. This design makes it straightforward to switch between different models while maintaining a consistent interface for data processing and training.

The pipeline integrates feature engineering capabilities with model training, providing a cohesive workflow that handles both numerical and categorical data appropriately. By abstracting away the complexity of data preprocessing and model training, it allows users to focus on the higher-level aspects of their machine learning tasks rather than implementation details. The structure ensures that all data goes through proper preprocessing steps before being fed into any model, reducing the chance of errors and inconsistencies in the training process.

### Feature Engineer

{% highlight Python %}
class FeatureEngineer:
    def __init__(self):
        self.transformers = {}
        self.numeric_col = []
        self.categorical_columns = []

    def add_numeric_features(self, columns):
        self.numeric_columns = columns
        self.transformers['standard_scaler'] = StandardScaler()

    def add_categorical_features(self, columns):
        self.categorical_columns = columns

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        if self.numeric_columns:
            df[self.numeric_columns] = self.transformers['standard_scaler'].fit_transform(df[self.numeric_columns])

        return df
{% endhighlight %}

The FeatureEngineer class represents a sophisticated approach to data transformation and preparation. It implements essential preprocessing steps including standardization for numerical features and encoding for categorical variables. The system maintains separate handling for different types of features, allowing for appropriate transformations to be applied to each data type. The standardization process uses sklearn's StandardScaler, ensuring that numerical features are properly scaled to have zero mean and unit variance, which is crucial for many machine learning algorithms to perform optimally.

What makes this feature engineering system particularly useful is its stateful nature - it remembers the transformations applied during training and can consistently apply the same transformations to new data during prediction. This is essential for maintaining the integrity of the machine learning pipeline in production environments. The system is also extensible, designed to accommodate additional transformation types and feature engineering steps as needed.

### Running the Pipeline

The main execution script serves as the entry point for the entire framework, demonstrating its practical application using the Iris dataset as an example. It showcases how to properly initialize the pipeline, prepare data, train models, and evaluate their performance. The script handles command-line arguments to specify which model to use, making it easy to experiment with different algorithms. It implements a complete workflow from data loading to model evaluation, including proper train-test splitting to assess model generalization.

The runner provides clear feedback about the process, including model training and testing scores, making it easy to compare the performance of different models. It also includes error handling for cases where unsupported models are requested, ensuring graceful failure when invalid inputs are provided. The implementation demonstrates best practices for machine learning workflows, including proper data splitting, feature preprocessing, and model evaluation.

---