# Principal Component Analysis (PCA) with Python

### Dimensionality Reduction, Explained Variance, Data Visualization, and Machine Learning Classification

![PCA Projection of Handwritten Digits](assets/pca_notebook_figure_77.png)

*Figure: Two-dimensional PCA projection of handwritten digits, generated in the original project notebook.*

**Project Type:** Academic Data Science and Machine Learning Project  
**Field:** Data Science | Machine Learning | Linear Algebra | Dimensionality Reduction  
**Programming Language:** Python  
**Environment:** Google Colab | Jupyter Notebook  
**Author:** Data Engineer Luis Barahona

---

## 1. Project Overview

This project explores Principal Component Analysis (PCA), a statistical dimensionality reduction technique used to transform correlated numerical variables into a smaller set of orthogonal components.

The notebook combines mathematical explanations, manual PCA calculations, Scikit-learn implementations, data visualization, and classification experiments.

Three main practical areas are examined:

1. Admission examination scores.
2. Quality-of-life indicators across countries.
3. Handwritten digit recognition.

The final experiment compares Gradient Boosting classification performance using original image features and PCA-reduced representations.

The objective is to investigate how dimensionality reduction affects information preservation, visualization, and predictive performance.

---

## 2. Project Objectives

- Understand the mathematical foundations of PCA.
- Calculate covariance matrices.
- Obtain eigenvalues and eigenvectors.
- Measure explained variance.
- Determine the number of components needed to retain information.
- Construct projection matrices.
- Apply PCA using NumPy and Scikit-learn.
- Standardize variables with different numerical scales.
- Visualize high-dimensional datasets in reduced spaces.
- Reconstruct handwritten digit images using principal components.
- Compare machine learning classification performance before and after dimensionality reduction.

---

## 3. Principal Component Analysis

Principal Component Analysis transforms a dataset into new, mutually orthogonal variables called principal components.

Each component captures a portion of the variance in the original dataset.

The components are ordered by explained variance:

- The first principal component captures the largest possible variance.
- The second captures the largest remaining variance while being orthogonal to the first.
- Subsequent components capture progressively smaller amounts.

PCA can help reduce the number of dimensions while preserving a selected proportion of the dataset's variability.

### Mathematical Workflow

The notebook introduces the following procedure:

1. Calculate the covariance matrix.
2. Obtain its eigenvalues and eigenvectors.
3. Determine the cumulative explained variance.
4. Select the desired number of principal components.
5. Construct the projection matrix.
6. Transform the original observations into the reduced-dimensional space.

For a centered data matrix X and projection matrix W:

```text
Z = X_centered × W
```

Z represents the transformed dataset.

---

## 4. Experiment 1 — Admission Examination Scores

### Dataset

**File:** `examen_data.csv`

The dataset contains 1,000 observations with four numerical variables:

| Feature | Description |
|:---|:---|
| mat | Mathematics score |
| fis | Physics score |
| esp | Spanish score |
| lec | Reading score |

### Methodology

The notebook:

- Loads the data with Pandas.
- Calculates the mean of each variable.
- Constructs the covariance matrix.
- Calculates eigenvalues and eigenvectors.
- Determines cumulative explained variance.
- Selects the first two principal components.
- Projects the original four-dimensional dataset into two dimensions.
- Repeats the procedure using Scikit-learn.

### Experimental Results

| Principal Components | Cumulative Explained Variance |
|:---|---:|
| 1 | 49.81% |
| 2 | 88.62% |
| 3 | 95.80% |
| 4 | 100.00% |

### Key Finding

The first two principal components explain approximately 88.62% of the total variance.

The original dataset has the following dimensions:

```text
Original:  (1000, 4)
Projected: (1000, 2)
```

This example demonstrates how PCA can reduce dimensionality while retaining most of the dataset's variance.

---

## 5. Experiment 2 — Quality-of-Life Indicators

### Dataset

**File:** `quality_life_2020.csv`

**Source:** Countries Dataset 2020, available on Kaggle.

[Quality of Life Index by Countries 2020](https://www.kaggle.com/datasets/dumbgeek/countries-dataset-2020)

The analysis examines multiple numerical indicators describing quality-of-life conditions across countries.

Examples include:

- Quality of life.
- Purchasing power.
- Safety.
- Healthcare.
- Other socioeconomic indicators.

### Methodology

The notebook demonstrates why variable scaling matters before PCA.

The process includes:

1. Loading the dataset.
2. Inspecting numerical variables.
3. Visualizing feature distributions with box plots.
4. Applying StandardScaler.
5. Fitting PCA to the standardized data.
6. Calculating explained variance.
7. Projecting observations into three dimensions.

### Experimental Results

| Principal Components | Cumulative Explained Variance |
|:---|---:|
| 1 | 53.01% |
| 2 | 66.63% |
| 3 | 78.68% |
| 4 | 87.14% |
| 5 | 91.84% |

The first three components preserve approximately 78.68% of the variance in the standardized dataset.

### Key Finding

Standardization is important when original variables use different numerical scales.

Without appropriate preprocessing, variables with larger variances may dominate the principal components.

### Additional Dataset Exploration

The notebook also loads `communities.data` as an exploratory example.

This section is not developed into a complete, independently evaluated PCA experiment.

---

## 6. Experiment 3 — Handwritten Digit Recognition

### Dataset

The notebook uses the handwritten digits dataset included in Scikit-learn.

It contains:

- 1,797 digit images.
- 64 numerical features per image.
- Images represented as 8 × 8 pixel matrices.
- Digit labels from 0 through 9.

### Binary Classification Task

The experiment selects two digit classes:

- Digit 1.
- Digit 9.

This produces a subset containing 362 observations.

The objective is to examine whether PCA can reduce the number of image features while preserving useful information for distinguishing these two digits.

### Dimensionality Reduction

The notebook initially analyzes the explained variance of the 64 original features.

It identifies that nine principal components preserve approximately 80.63% of the variance in the selected digit subset.

The dimensionality is reduced as follows:

```text
Original representation: 64 features
Reduced representation:   9 components
Visualization space:      2 components
```

### Principal Component Visualization

Each principal component contains 64 values.

These values can be reshaped into 8 × 8 matrices and visualized as images.

This allows the notebook to explore which patterns are captured by the principal components.

### Image Reconstruction

The notebook also demonstrates approximate image reconstruction from the reduced-dimensional representation.

The experiment illustrates how dimensionality reduction can preserve recognizable structures while discarding some information.

### Two-Dimensional Projection

Images of digits 1 and 9 are transformed into a two-dimensional PCA space.

The resulting scatter plot illustrates how the selected classes are distributed in the reduced representation.

---

## 7. Machine Learning Classification Experiment

The final experiment compares Gradient Boosting classification using three feature representations.

### Model 1 — Original Features

A GradientBoostingClassifier is trained using the original 64 image features.

### Model 2 — PCA with 80% Explained Variance

PCA is fitted to the training data.

Nine principal components are selected to retain at least 80% of the training-data variance.

The transformed features are used to train a second Gradient Boosting classifier.

### Model 3 — PCA with Two Components

The original image features are reduced to two principal components.

A third Gradient Boosting classifier is trained using this two-dimensional representation.

### Train-Test Split

The experiment uses:

```python
train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)
```

The resulting datasets contain:

| Dataset | Observations |
|:---|---:|
| Training | 253 |
| Testing | 109 |
| Total | 362 |

PCA is fitted to the training subset for the comparative evaluation, and the same fitted transformation is applied to the test subset.

This avoids fitting the dimensionality reduction step directly on the testing observations.

---

## 8. Experimental Results

### Gradient Boosting Classification Accuracy

| Model | Features / Components | Training Accuracy | Testing Accuracy |
|:---|---:|---:|---:|
| Original features | 64 | 100% | 98% |
| PCA — 80% variance | 9 | 100% | 97% |
| PCA — 2D | 2 | 100% | 93% |

### Interpretation

The model trained on the original 64 features achieved the highest testing accuracy.

Reducing the features from 64 to nine principal components resulted in a small decrease in recorded test accuracy.

Reducing the data to only two principal components produced a larger reduction in performance.

### Main Finding

PCA can substantially reduce feature dimensionality while preserving useful information for classification.

However, preserving the variance of input features does not guarantee that all information relevant to predicting class labels will also be preserved.

The experiment demonstrates a trade-off between dimensionality reduction and predictive accuracy.

### Additional Training-Set Experiment

The notebook also fits Gradient Boosting classifiers using all available observations.

Both the original-feature and nine-component models achieve a training accuracy of 100%.

These results are not independent test-set evaluations and should not be interpreted as evidence of perfect generalization.

---

## 9. Technologies Used

| Technology | Purpose |
|:---|:---|
| Python | Programming language |
| NumPy | Linear algebra and numerical computations |
| Pandas | Data loading and manipulation |
| Scikit-learn | PCA, StandardScaler, and classification |
| GradientBoostingClassifier | Machine learning classification |
| Matplotlib | Statistical and image visualization |
| Seaborn | Exploratory data visualization |
| Google Colab | Cloud notebook environment |
| Jupyter Notebook | Interactive computational analysis |

---

## 10. How to Run

### Google Colab

[Open the Original PCA Notebook](https://colab.research.google.com/drive/1IvWj68HlZiAkqTM9ldArqy0eoeXAzr7y)

### Requirements

- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook or Google Colab

### Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn notebook
```

### Required External Files

The notebook references the following files:

```text
examen_data.csv
quality_life_2020.csv
communities.data
```

These files must be available in the working directory for the corresponding exercises.

The handwritten digits dataset is loaded directly from Scikit-learn.

### Execution Instructions

1. Open the notebook.
2. Upload the required external datasets.
3. Execute the import cells.
4. Review the covariance-based PCA calculations.
5. Examine the explained variance results.
6. Execute the quality-of-life analysis.
7. Explore the handwritten digit examples.
8. Run the Gradient Boosting classification comparison.
9. Review the accuracy metrics and PCA visualizations.

Some exploratory cells rely on earlier variable definitions, so they should be executed in notebook order.

---

## 11. Key Findings

### Dimensionality Reduction

The admission examination dataset was reduced from four to two dimensions while preserving approximately 88.62% of its variance.

### Explained Variance

The quality-of-life example demonstrated how a limited number of principal components can summarize variation across multiple standardized indicators.

### Computer Vision

The handwritten digit experiment showed how images represented by 64 features can be projected into smaller spaces.

### Machine Learning

The Gradient Boosting comparison showed that reducing 64 features to nine components retained high test accuracy in the recorded experiment.

### Model Evaluation

The results demonstrate that reducing dimensionality can improve compactness, but predictive performance must be measured separately.

---

## 12. Limitations

This notebook is an educational exploration of PCA and machine learning.

Important limitations include:

- Some exercises require external datasets.
- The notebook combines multiple examples with reused variable names.
- Several exploratory sections are not developed into full experiments.
- Some training-set results are presented separately from the holdout evaluation.
- The digit classification experiment considers only classes 1 and 9.
- A single train-test split does not establish general performance across different samples.
- PCA is sensitive to the scale and preprocessing of numerical variables.
- Explained variance does not directly measure classification accuracy.

The recorded results are specific to the datasets, model configurations, and saved notebook executions.

---

## 13. Future Improvements

Potential extensions include:

- Organizing each PCA experiment into independent functions.
- Adding automated dataset validation.
- Using Scikit-learn Pipelines.
- Comparing PCA with other dimensionality reduction methods.
- Evaluating different explained variance thresholds.
- Implementing cross-validation.
- Adding classification reports and confusion matrices.
- Comparing model training times.
- Evaluating memory consumption before and after PCA.
- Extending digit classification to all ten classes.
- Building an interactive dimensionality reduction dashboard.

---

## 14. Practical Applications

The techniques demonstrated in this project are relevant to:

- Feature reduction.
- Exploratory Data Analysis.
- Image processing.
- Machine learning preprocessing.
- High-dimensional data visualization.
- Pattern recognition.
- Correlated variable analysis.
- Statistical modeling.
- Data Science education.

---

## 15. Project Resources

### Google Colab

[Open PCA Notebook](https://colab.research.google.com/drive/1IvWj68HlZiAkqTM9ldArqy0eoeXAzr7y)

### Data Science Plus — PCA with Python

[Principal Component Analysis (PCA) with Python](https://datascienceplus.com/principal-component-analysis-pca-with-python/)

**Author:** Abhinav Choudhary  
**Published:** September 29, 2019  
**Publisher:** Data Science Plus

This external article provides an additional educational explanation of PCA, feature scaling, dimensionality reduction, and principal component interpretation.

The article is credited as a related learning resource. The figures generated in this notebook are separate from the article's illustrations.

### Scikit-learn Documentation

[Principal Component Analysis (PCA)](https://scikit-learn.org/stable/modules/decomposition.html#pca)

### Dataset Reference

[Quality of Life Index by Countries — Kaggle](https://www.kaggle.com/datasets/dumbgeek/countries-dataset-2020)

---

## 16. Conclusion

This project demonstrates the principles and practical applications of Principal Component Analysis using Python.

Through covariance matrices, eigenvalue decomposition, standardized data, image reconstruction, and classification experiments, the notebook explores how high-dimensional information can be transformed into compact numerical representations.

The final Gradient Boosting experiment achieved 98% testing accuracy using 64 original features, compared with 97% using nine principal components and 93% using two components.

These findings illustrate how PCA can reduce the number of input dimensions while retaining useful predictive information.

The notebook provides an educational foundation for further studies in Data Science, Machine Learning, Linear Algebra, and Computer Vision.

---

## Author

**Data Engineer Luis Barahona**

Mexican Data Engineer and 2026 graduate of Universidad del Caribe, interested in Data Engineering, Data Science, Artificial Intelligence, Machine Learning, and continuous professional development.

---

## Contact

| Platform | Contact |
|:---|:---|
| Email | [190300395@ucaribe.edu.mx](mailto:190300395@ucaribe.edu.mx) |
| GitHub | [github.com/LuisBarahonaMX](https://github.com/LuisBarahonaMX) |
| LinkedIn | [linkedin.com/in/luis-barahona-mx](https://www.linkedin.com/in/luis-barahona-mx/) |
| Instagram | [@luisbarahonamx](https://www.instagram.com/luisbarahonamx/) |
| X | [@luisbarahonaMX](https://x.com/luisbarahonaMX) |
| Website | [SIKO AI Platform](https://sikoai.web.app/) |
