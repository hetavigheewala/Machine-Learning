# Author: Hetavi Gheewala

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from itertools import cycle

# streamlit app title
st.title("Cardio Risk Level KNN Classifier")

# explanation about dataset
st.write("""
This project analyzes cardiovascular risk levels using a dataset containing medical and lifestyle features of patients.  
The goal is to classify patients into one of four risk levels — No Risk, Low Risk, Moderate Risk, and High Risk — based on their health data.
""")

# load and preprocess data 
cardioData = pd.read_csv("cardio_train.csv")

st.write("### Raw Data Overview")
st.write("Here is a snapshot of the dataset used for training and testing the model:")
st.dataframe(cardioData.head())

st.write("""
The dataset contains various health and lifestyle features for patients, such as age, height, weight, blood pressure (both systolic and diastolic), cholesterol levels, glucose levels, and habits like smoking and alcohol habits.  
For example, one row might show a 55-year-old male with a systolic blood pressure (`ap_hi`) of 140 and diastolic blood pressure (`ap_lo`) of 90.  
These features help the model learn patterns associated with cardiovascular risk.
""")

# filtering outliers (your existing logic)
cardioData = cardioData[
    (cardioData["ap_lo"] < 120) | (cardioData["ap_hi"] > 50) | (cardioData["ap_hi"] < 200)
]

st.write("""
I filtered out some outliers in blood pressure readings to improve the quality of the data.  
- `ap_hi`: Systolic blood pressure — the pressure when the heart beats  
- `ap_lo`: Diastolic blood pressure — the pressure when the heart rests between beats  

For example, a reading with `ap_hi` of 250 or `ap_lo` of 130 is unlikely and considered an outlier or measurement error. Removing such values helps reduce noise in the data, making the model more reliable.
""")

# define your risk level assignment function
def assign_risk_level(person):
    # explanation of how risk levels are assigned
    # Risk level 0 = No Risk: Healthy parameters, no smoking or alcohol
    # Risk level 1 = Low Risk: Slightly elevated parameters or mild lifestyle risk factors
    # Risk level 2 = Moderate Risk: Noticeable elevations in cholesterol, glucose, or blood pressure
    # Risk level 3 = High Risk: High cholesterol/glucose or blood pressure, or presence of alcohol/smoking
    
    if (
        person["cardio"] == 0
        and person["cholesterol"] <= 1
        and person["gluc"] <= 1
        and person["ap_hi"] <= 120
        and person["ap_lo"] <= 80
        and person["alco"] <= 0
        and person["smoke"] <= 0
    ):
        return 0  
    elif (
        person["cardio"] == 0
        and person["cholesterol"] <= 2
        and person["gluc"] <= 2
        and person["ap_hi"] <= 130
        and person["ap_lo"] <= 85
        and person["alco"] <= 1
        and person["smoke"] <= 1
    ):
        return 1  
    elif (
        person["cardio"] <= 1
        and person["cholesterol"] <= 3
        and person["gluc"] <= 3
        and 130 <= person["ap_hi"] <= 149
        and 85 <= person["ap_lo"] <= 89
        and person["alco"] <= 1
        and person["smoke"] <= 1
    ):
        return 2  
    elif (
        person["cardio"] <= 1
        and person["cholesterol"] <= 3
        and person["gluc"] <= 3
        and person["ap_hi"] <= 200
        and person["ap_lo"] <= 120
        and person["alco"] <= 1
        and person["smoke"] <= 1
    ):
        return 3  
    return -1  

cardioData["risk_level"] = cardioData.apply(assign_risk_level, axis=1)
cardioData = cardioData[cardioData["risk_level"] != -1]

st.write("""
Patients are categorized into risk levels using a set of clinical and lifestyle thresholds as described above.  
Entries that don't fit into these categories are removed for clearer modeling.
""")

# prepare features and labels
inputFeatures = cardioData[
    ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
]
outputRiskLevels = cardioData["risk_level"]

st.write("### Feature and Label Distribution")
st.write("The following features are used to predict risk level:")
st.write(inputFeatures.columns.tolist())
st.write("The target variable is the `risk_level` with 4 categories.")

# split dataset into training and testing sets
trainFeatures, testFeatures, trainRisk_level, testRisk_levels = train_test_split(
    inputFeatures, outputRiskLevels, test_size=0.2, random_state=42, stratify=outputRiskLevels
)

# normalize features to standard scale for knn
scaler = StandardScaler()
trainFeaturesNormalized = scaler.fit_transform(trainFeatures)
testFeaturesNormalized = scaler.transform(testFeatures)

st.write("""
Features are normalized to have zero mean and unit variance. This is important for K-Nearest Neighbors (KNN), which relies on distance calculations sensitive to scale.
""")

# apply pca for dimensionality reduction while preserving 95% variance
pca = PCA(n_components=0.95)
trainFeaturesPCA = pca.fit_transform(trainFeaturesNormalized)
testFeaturesPCA = pca.transform(testFeaturesNormalized)

st.write(f"Principal Component Analysis (PCA) reduces the feature space to {trainFeaturesPCA.shape[1]} dimensions, while retaining 95% of the original variance to improve model performance and reduce overfitting.")

# trying different distance metrics and k values for knn classifier
distance_metrics = ["manhattan", "euclidean", "minkowski"]
results = []

for metric in distance_metrics:
    bestK = 0
    bestAccuracy = 0
    for k in range(1, 21):
        knn_model = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn_model.fit(trainFeaturesPCA, trainRisk_level)
        predicted_risk_levels = knn_model.predict(testFeaturesPCA)
        modelAccuracy = accuracy_score(testRisk_levels, predicted_risk_levels) * 100
        if modelAccuracy > bestAccuracy:
            bestK = k
            bestAccuracy = modelAccuracy
    results.append((metric, bestK, bestAccuracy))

optimal_metric, optimal_k, optimal_accuracy = max(results, key=lambda x: x[2])

st.header("Model Tuning Results")
st.write("""
I evaluated KNN performance using different distance metrics (`manhattan`, `euclidean`, and `minkowski`)  
and varied the number of neighbors (`k`) from 1 to 20.  
The goal was to find the combination with the highest classification accuracy on the test set.
""")

for metric, k, accuracy in results:
    st.write(f"- Metric: **{metric}**, Optimal K: **{k}**, Accuracy: **{accuracy:.2f}%**")

st.write(f"### Optimal Model Selected")
st.write(f"- Distance Metric: **{optimal_metric}**")
st.write(f"- Number of Neighbors (K): **{optimal_k}**")
st.write(f"- Accuracy: **{optimal_accuracy:.2f}%**")

# train optimal model
knn_model = KNeighborsClassifier(n_neighbors=optimal_k, metric=optimal_metric)
knn_model.fit(trainFeaturesPCA, trainRisk_level)
predicted_risk_levels = knn_model.predict(testFeaturesPCA)
modelAccuracy = accuracy_score(testRisk_levels, predicted_risk_levels) * 100

st.write(f"Final model accuracy on test data: **{modelAccuracy:.2f}%**")

# confusion Matrix display
confMatrix = confusion_matrix(testRisk_levels, predicted_risk_levels)
riskLabels = ["No Risk", "Low Risk", "Moderate Risk", "High Risk"]

st.subheader("Confusion Matrix")
st.write("""
The confusion matrix below shows how well the model predicts each risk level category:  
- Rows represent the actual risk levels (ground truth)  
- Columns represent the predicted risk levels  
The closer the values are to the diagonal, the better the model is performing.
""")
st.dataframe(pd.DataFrame(confMatrix, columns=[f"Predicted-{label}" for label in riskLabels],
                          index=[f"Actual-{label}" for label in riskLabels]))

# classification Report
report_dict = classification_report(
    testRisk_levels,
    predicted_risk_levels,
    target_names=riskLabels,
    zero_division=0,
    output_dict=True
)
report_df = pd.DataFrame(report_dict).transpose()
report_df[['precision', 'recall', 'f1-score']] = report_df[['precision', 'recall', 'f1-score']].applymap(lambda x: f"{x:.2f}")

st.subheader("Classification Report")
st.write("""
This report provides key metrics for each risk class:  
- **Precision**: How many predicted positives were correct  
- **Recall**: How many actual positives were identified  
- **F1-score**: Harmonic mean of precision and recall, representing balance between the two  
High scores indicate a strong classifier.
""")
st.dataframe(report_df)

# plot confusion matrix heatmap
fig, ax = plt.subplots()
cax = ax.matshow(confMatrix, cmap="Blues")
fig.colorbar(cax)
ax.set_xticklabels([''] + [f"P-{label}" for label in riskLabels], rotation=45)
ax.set_yticklabels([''] + [f"A-{label}" for label in riskLabels])
ax.set_xlabel("Predicted Risk Level")
ax.set_ylabel("Actual Risk Level")
ax.set_title("Confusion Matrix Heatmap")
st.pyplot(fig)

st.write("""
The heatmap visually represents the confusion matrix, where darker colors indicate higher counts.  
The diagonal cells represent correct predictions, while off-diagonal cells indicate misclassifications.
""")
st.write("")  
st.write("")  

# ROC Curve and AUC for multi-class classification
yTestBinarized = label_binarize(testRisk_levels, classes=[0, 1, 2, 3])
yPredProba = knn_model.predict_proba(testFeaturesPCA)
nClasses = yTestBinarized.shape[1]

fpr, tpr, rocAuc = dict(), dict(), dict()
for i in range(nClasses):
    fpr[i], tpr[i], _ = roc_curve(yTestBinarized[:, i], yPredProba[:, i])
    rocAuc[i] = auc(fpr[i], tpr[i])

fig2, ax2 = plt.subplots()
colors = cycle(["blue", "green", "red", "orange"])
for i, color in zip(range(nClasses), colors):
    ax2.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f"ROC curve for {riskLabels[i]} (area = {rocAuc[i]:.2f})")

ax2.plot([0, 1], [0, 1], "k--", lw=2)
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve for Multi-Class Classification")
ax2.legend(loc="lower right")
st.pyplot(fig2)

st.write("""
The Receiver Operating Characteristic (ROC) curves evaluate the trade-off between sensitivity (True Positive Rate) and specificity (False Positive Rate) for each risk class.  
- The closer a curve is to the top-left corner, the better the classifier performance for that class.  
- The Area Under Curve (AUC) score quantifies this performance; higher AUC indicates better discrimination.
""")
