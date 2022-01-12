import pandas as pd

class_df = pd.read_csv("D:\AI\kaggle-project\Bird-Species\datasets\class_dict.csv")
print(class_df)
class_count = len(class_df['class'].unique())
print(class_count)