import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = [0, 0, 1, 0, 2]
y_true = [0, 0, 1, 1, 2]

l = os.listdir('../../final_project_dataset_v0')
print(l)

conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)

disp.plot()
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()