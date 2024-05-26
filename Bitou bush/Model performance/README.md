# Accuracy Assessment

The models were evaluated and compared based on their performance metrics and ability to handle different classes effectively in a classification scenario. Overall model accuracy, precision, recall, F1-score, and Intersection over Union (IoU) were used to evaluate the model’s performance for the classification of pandanus. Evaluation descriptors, including true positive (TP), false positive (FP), true negative (TN), and false negative (FN), were used to determine the overall accuracy (Equation (1)), precision (Equation (2)), recall (Equation (3)), F1-score (Equation (4)), and IoU (Equation (5)).

**Overall Accuracy**:
\[ \text{Overall Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} \] (1)

**Precision**:
\[ \text{Precision} = \frac{TP}{TP + FP} \] (2)

**Recall**:
\[ \text{Recall} = \frac{TP}{TP + FN} \] (3)

**F1-score**:
\[ \text{F1-score} = \frac{2TP}{FP + 2TP + FN} \] (4)

**Intersection over Union (IoU)**:
\[ \text{IoU} = \frac{\text{Area of intersection}}{\text{Area of Union}} \] (5)

These performance metrics provide insights into the model's ability to correctly classify pandanus instances and distinguish them from other classes, thus guiding the selection of the most suitable model for the task.
