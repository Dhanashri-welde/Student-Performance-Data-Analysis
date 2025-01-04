# Student-Performance-Data-Analysis

## Description:
This project uses a dataset containing information about 480 students. The goal is to analyze how various factors, such as class participation, parental involvement, and absences, affect students' academic performance. We use **Logistic Regression** to predict the performance category of the student.

## Columns:
- **`gender`**: Gender of the student (e.g., Male, Female)
- **`NationalITy`**: Nationality of the student
- **`PlaceofBirth`**: Where the student was born
- **`StageID`**: The stage or level of education the student is in
- **`GradeID`**: The grade the student is in
- **`SectionID`**: The section the student belongs to
- **`Topic`**: The subject the student is studying
- **`Semester`**: The semester during which data was recorded
- **`Relation`**: Parent’s relationship status
- **`raisedhands`**: Number of times the student raised their hand in class
- **`VisITedResources`**: Number of resources the student accessed for learning
- **`AnnouncementsView`**: Number of announcements the student viewed
- **`Discussion`**: Number of discussions the student took part in
- **`ParentAnsweringSurvey`**: Whether the student’s parent answered a survey (Yes/No)
- **`ParentschoolSatisfaction`**: How satisfied the parent is with the school
- **`StudentAbsenceDays`**: Number of days the student was absent
- **`Class`**: Performance classification (e.g., "Good", "Average", "Bad")

## Usage:
We can use this data to predict how students perform based on their engagement in class, parental involvement, and attendance.

## Machine Learning Model:

### Steps:
**Q.1 Visualize just the categorical features individually to see 
what options are included and how each option fares 
when it comes to count(how many times it appears) and 
see what can be deduce from that?** 
```python 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r"C:\Users\Asus\Desktop\ML Dataset\xAPI-Edu-Data.csv")

def EDA_info(df):
    print("Shape of dataframe")
    print(df.shape,'\n')

    print("First 5 rows of DataFrame")
    print(df.head(),'\n')

    print("Last 5 rows of DataFrame")
    print(df.tail(),'\n')

    print("DataFrame info")
    print(df.info(),'\n')

    print("Display all column in DataFrame")
    print(df.columns.tolist(),'\n')

    print("Statistical summary of numeric column")
    print(df.describe(),'\n')
```

![Screenshot 2025-01-04 154826](https://github.com/user-attachments/assets/217d0b50-6c18-4b57-acd9-71d0507fc79c)
![Screenshot 2025-01-04 162945](https://github.com/user-attachments/assets/de4b2b9e-7d0c-480f-ac6d-7bbb744da606)

**Q.2 To Check outlier and missing value using def function**
```python
def EDA_with_outlier(df):
    print("To Check Missing Values")
    print(df.isnull().sum(),'\n')

    print("To Check Outlier in Data")
    sns.boxplot(data = df)
```
![Missing values](https://github.com/user-attachments/assets/81198ad2-1828-4c04-a7cb-d5b66e7a7d98)
![outlier](https://github.com/user-attachments/assets/72a8ae24-53ed-4a4a-8240-731fd09e3554)

**Q.2 Look at some categorical features in relation to each 
other, to see what insights could be possibly read?**
```python
def Categorical_features(df):
    for col in df.columns:
        if df[col].dtype == 'object':  
            plt.figure(figsize=(8, 6))
            sns.countplot(data=df, x=col)
            plt.title(f"Distribution of {col}")  
            plt.xticks(rotation=45) 
            plt.show()
```
```python
def categorical_relationships_heatmap(data, feature_pairs):
    for feature1, feature2 in feature_pairs:
        crosstab = pd.crosstab(data[feature1], data[feature2])
        plt.figure(figsize=(6, 6))
        sns.heatmap(crosstab, annot=True, cmap="YlGnBu", fmt='d', cbar=True)
        plt.title(f"Heatmap of Relationship Between {feature1} and {feature2}", fontsize=14)
        plt.xlabel(feature2, fontsize=12)
        plt.ylabel(feature1, fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

feature_pairs = [
    ('gender', 'Class'), 
    ('StageID', 'GradeID'), 
    ('Relation', 'ParentschoolSatisfaction'),
    ('Semester', 'StudentAbsenceDays')
]
categorical_relationships_heatmap(df, feature_pairs)
```

![Heatmap](https://github.com/user-attachments/assets/7489044a-8921-4c57-8631-385c9198610e)
![Heatmap2](https://github.com/user-attachments/assets/3745d53e-2556-4fe0-95a7-bf290f1b5297)
`![Heatmap3](https://github.com/user-attachments/assets/b20f3d31-5a4a-4842-a862-401c0e985eb5)
![Heatmap4](https://github.com/user-attachments/assets/94a5b323-b449-42fc-a776-d574fc6a12e3)

**Q.3 Visualize categorical variables with numerical variables 
and give conclusions?**
```python
categorical_features = [
    'gender', 'NationalITy', 'StageID', 'ParentAnsweringSurvey',
    'ParentschoolSatisfaction', 'StudentAbsenceDays', 'Class'
]

numerical_features = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']

def visualize_cat_var_with_num_var():
    for cat in categorical_features:
        for num in numerical_features:
            plt.figure(figsize=(8, 4))
            sns.barplot(data=df, x=cat, y=num, palette='viridis')
            plt.title(f'{num} by {cat}')
            plt.xticks(rotation=45)
            plt.show()


visualize_cat_var_with_num_var()
```
![Categorial to num](https://github.com/user-attachments/assets/372341a2-d9e3-4d14-b01b-b467831a3a30)

**Q.4 From the above result, what are the factors that leads to 
get low grades of the students?**

![Conclusion 1](https://github.com/user-attachments/assets/487e1b27-ee5a-4bb3-b0cc-777c00a8cf86)
![Conclusion2](https://github.com/user-attachments/assets/b47b8634-3ec9-436d-88e1-51fe85ab5974)

**Q.5 Build classification model and present it's classification 
report?**
```python
from sklearn.preprocessing import LabelEncoder

def data_preprocess(df):
    numeric_data = df.select_dtypes(include=(np.number))
    category_data = df.select_dtypes(include=('object'))
    category_data = category_data.apply(LabelEncoder().fit_transform)
    combined_data = pd.concat([numeric_data,category_data], axis=1)
    return numeric_data, category_data, combined_data
numeric_data, category_data, combined_data = data_preprocess(df)
```
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.naive_bayes import GaussianNB

def train_model_with_sampling(data, target_column, n_features=5, test_size=0.20, random_state=36, sampling_method='both'):
    X = data.drop(target_column, axis=1)
    Y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    if sampling_method == 'oversample':
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    elif sampling_method == 'undersample':
        rus = RandomUnderSampler(random_state=random_state)
        X_train, y_train = rus.fit_resample(X_train, y_train)
    elif sampling_method == 'both':
        smote = SMOTE(random_state=random_state)
        rus = RandomUnderSampler(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        X_train, y_train = rus.fit_resample(X_train, y_train)
    logreg = LogisticRegression(random_state=random_state)
    selector = sfs(logreg, n_features_to_select=n_features, direction='forward', scoring='accuracy')
    selector.fit(X_train, y_train)

    selected_features = selector.get_support(indices=True)
    selected_feature_names = X_train.columns[selected_features]
    X_train_selected = X_train.iloc[:, selected_features]
    X_test_selected = X_test.iloc[:, selected_features]
    
    logreg_model = logreg.fit(X_train_selected, y_train)
    nb_model = GaussianNB()
    nb_model.fit(X_train_selected, y_train)

    return logreg_model, nb_model, selected_feature_names, X_train_selected, X_test_selected, y_train, y_test
logreg_model, nb_model, selected_features, X_train_selected, X_test_selected, y_train, y_test = train_model_with_sampling(
    combined_data, 'Class', sampling_method='both')

print("Selected Features:", selected_features)
```
**Prediction On train data**
```python
import numpy as np
from sklearn.metrics import classification_report

def train_prediction(df, Model, X_train_selected, y_train):
    training_data = pd.concat([X_train_selected, y_train], axis=1)
    training_data['Bad_Probability'] = Model.predict_proba(X_train_selected)[:, 1]
    training_data['Predicted'] = np.where(training_data['Bad_Probability'] >= 0.7, 1, 0)
    print(classification_report(training_data['Class'], training_data['Predicted']))
    return training_data
training_data_prediction = train_prediction(df, logreg_model, X_train_selected, y_train)
```
![accuracy on train data](https://github.com/user-attachments/assets/62c61a92-a07c-4481-ad43-467b077072ec)

**Prediction on test data**
```python
import numpy as np
from sklearn.metrics import classification_report

def test_prediction(df, Model, X_test_selected, y_test):
    testing_data = pd.concat([X_test_selected, y_test], axis=1)
    testing_data['Bad_Probability'] = Model.predict_proba(X_test_selected)[:, 1]
    testing_data['Predicted'] = np.where(testing_data['Bad_Probability'] >= 0.7, 1, 0)
    print(classification_report(testing_data['Class'], testing_data['Predicted']))
    return testing_data
testing_data_prediction = test_prediction(df, logreg_model, X_test_selected, y_test)
```
![accuracy on test data](https://github.com/user-attachments/assets/a95288b2-a726-4af3-9e4a-7c537c94b9e8)






