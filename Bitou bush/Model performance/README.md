# Accuracy Assessment

## Overview
This repository contains code for evaluating and comparing models based on their performance metrics in a classification scenario, particularly focusing on the classification of pandanus. The evaluation metrics used include accuracy, precision, recall, F1-score, and Intersection over Union (IoU).

## Evaluation Metrics
The models were assessed using the following metrics:

- **Overall Accuracy**: Measures the overall correctness of the classification.
- **Precision**: Measures the ratio of correctly predicted positive observations to the total predicted positives.
- **Recall**: Measures the ratio of correctly predicted positive observations to all actual positives.
- **F1-score**: The harmonic mean of precision and recall, providing a balance between the two metrics.
- **Intersection over Union (IoU)**: Measures the overlap between predicted and ground truth bounding boxes.

## Evaluation Descriptors
The evaluation descriptors used in determining the performance metrics are as follows:

- **True Positive (TP)**: Instances where the model correctly predicts positive cases.
- **False Positive (FP)**: Instances where the model incorrectly predicts positive cases.
- **True Negative (TN)**: Instances where the model correctly predicts negative cases.
- **False Negative (FN)**: Instances where the model incorrectly predicts negative cases.

These descriptors are utilized in the calculation of overall accuracy, precision, recall, F1-score, and IoU using the following equations:

1. **Overall Accuracy** = (TP + TN) / (TP + FP + TN + FN)

2. **Precision** = (TP) / (TP + FP)

3. **Recall** = (TP) / (TP + FN)

4. **F1-score** = 2 * ((Precision) * (Recall)) / ((Precision) + (Recall))

5. **Intersection over Union (IoU)** = (Area of Intersection) / (Area of Union)
