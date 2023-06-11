import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

submit_data = pd.read_csv('D:\\ThesisProject\\FakeNewsDetection\\data\\submit.csv')
sns.set()
plt.figure(figsize=(15, 8))
plt.grid(None)
sns.countplot(x='label', data=submit_data)
plt.title("Distribution in submit.csv")
plt.savefig("distribution_submit.png")
plt.show()