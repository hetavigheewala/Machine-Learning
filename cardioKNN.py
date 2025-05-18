# Author: Hetavi Gheewala


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, roc_curve, auc)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from itertools import cycle

# load cardioData
cardioData = pd.read_csv("cardio_train.csv")

# filter data of outliers
cardioData = cardioData[(cardioData["ap_lo"] < 120) | (cardioData["ap_hi"] > 50) | (cardioData["ap_hi"] < 200)]

# risk level sort function
def assign_risk_level(person):
    # No risk
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
    # Low risk
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
    # Moderate risk
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
    # High risk
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
    # invalid case
    return -1  

# apply the risk level sort function to all rows of data
cardioData["risk_level"] = cardioData.apply(assign_risk_level, axis=1)

# remove rows where the risk level is invalid AKA outliers
cardioData = cardioData[cardioData["risk_level"] != -1]

# select input and output featuer of model
inputFeatures = cardioData[
    ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
]
outputRiskLevels = cardioData["risk_level"]

# divide the cardoData into 80 percent of training and 20 percent of test sets
trainFeatures, testFeatures, trainRisk_level, testRisk_levels = train_test_split(
    inputFeatures, outputRiskLevels, test_size=0.2, random_state=42, stratify=outputRiskLevels
)

# normalize the input features to make them have zero mean and unit variance to make them easier for the model to handle
scaler = StandardScaler()
trainFeaturesNormalized = scaler.fit_transform(trainFeatures)
testFeaturesNormalized = scaler.transform(testFeatures)

# using PCA to reduce the number of features with keeping same variance
pca = PCA(n_components=0.95)
trainFeaturesPCA = pca.fit_transform(trainFeaturesNormalized)
testFeaturesPCA = pca.transform(testFeaturesNormalized)

# try multiple distance metrics
distance_metrics = ["manhattan", "euclidean", "minkowski"]
results = []

# get the best K value for K-Nearest Neighbors classifier by testing K value from 1-20
for metric in distance_metrics:
    bestK = 0
    bestAccuracy = 0
    for k in range(1, 21):
        # train KNN model with specific distance metric
        knn_model = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn_model.fit(trainFeaturesPCA, trainRisk_level)
        
        # predict and evaluate
        
        predicted_risk_levels = knn_model.predict(testFeaturesPCA)
        modelAccuracy = accuracy_score(testRisk_levels, predicted_risk_levels) * 100
        
        # track best K for this metric
        if modelAccuracy > bestAccuracy:
            bestK = k
            bestAccuracy = modelAccuracy
    
    # save results for this metric
    results.append((metric, bestK, bestAccuracy))

# get the overall best distance metric and K
optimal_metric, optimal_k, optimal_accuracy = max(results, key=lambda x: x[2])

# output results for each calculation method 
print("Results by Distance Metric:")
for metric, k, accuracy in results:
    print(f"Metric: {metric}, Optimal K: {k}, Accuracy: {accuracy:.2f}%")
print(f"\nOptimal Metric: {optimal_metric}, Optimal K: {optimal_k}, Accuracy: {optimal_accuracy:.2f}%")


# train the KNN model with the best K value
knn_model = KNeighborsClassifier(n_neighbors=optimal_k, metric=optimal_metric)
knn_model.fit(trainFeaturesPCA, trainRisk_level)

# check the trained KNNmodel and get its accuracy on the test data
predicted_risk_levels = knn_model.predict(testFeaturesPCA)
modelAccuracy = accuracy_score(testRisk_levels, predicted_risk_levels) * 100
print(f"Model Accuracy with Best K ({bestK}): {modelAccuracy:.2f}%")

# create confusion matrix to get the model's performance on the best K
confMatrix = confusion_matrix(testRisk_levels, predicted_risk_levels)
riskLabels = ["No Risk", "Low Risk", "Moderate Risk", "High Risk"]
print("\nConfusion Matrix:\n", pd.DataFrame(confMatrix, columns=["P-No Risk", "P-Low Risk", "P-Moderate Risk", "P-High Risk"],
                                             index=["A-No Risk", "A-Low Risk", "A-Moderate Risk", "A-High Risk"]))

# classification Report (Precision, Recall, F1-Score)
report = classification_report(testRisk_levels, predicted_risk_levels, target_names=riskLabels, zero_division=0)
print("\nClassification Report:\n", report)


# visualize the confusion matrix into a heatmap
plt.matshow(confMatrix, cmap="Blues")   
plt.colorbar()  
plt.xticks(range(len(riskLabels)), [f"P-{label}" for label in riskLabels], rotation=45)
plt.yticks(range(len(riskLabels)), [f"A-{label}" for label in riskLabels])
plt.xlabel("Predicted Risk Level")          #  x-axis
plt.ylabel("Actual Risk Level")             #  y-axis
plt.title("Confusion Matrix with Best K")   # title 
plt.show()                                  # display the map


# ROC Curve and AUC ---
# binarize the output for multi-class ROC
yTestBinarized = label_binarize(testRisk_levels, classes=[0, 1, 2, 3])
yPredProba = knn_model.predict_proba(testFeaturesPCA)
nClasses = yTestBinarized.shape[1]

# generate ROC curve and AUC for each class
fpr, tpr, rocAuc = dict(), dict(), dict()
for i in range(nClasses):
    fpr[i], tpr[i], _ = roc_curve(yTestBinarized[:, i], yPredProba[:, i])
    rocAuc[i] = auc(fpr[i], tpr[i])

# plot the ROC curve for each class
colors = cycle(["blue", "green", "red", "orange"])
plt.figure()
for i, color in zip(range(nClasses), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f"ROC curve for {riskLabels[i]} (area = {rocAuc[i]:.2f})")

plt.plot([0, 1], [0, 1], "k--", lw=2)  
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Multi-Class Classification")
plt.legend(loc="lower right")
plt.show()

