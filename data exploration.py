import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kruskal
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# these are variables for running ops further down
plots = 0
stat_anal = 1


# working directory
def directory(base_folder, subfolder=None):
    try:
        os.chdir(base_folder)
        if subfolder:
            full_path = os.path.join(base_folder, subfolder)
            os.chdir(full_path)
            print(f"working directory is: {full_path}")
        else:
            print("no sub folder. We're in data")

    except Exception as e:
        print(f"error setting working directory: {e}")


# set the folder and subfolder
folder = r'C:\Users\ander\Documents\.data'
dir = 'titanic'
directory(folder, dir)

plots_folder = os.path.join(folder, dir, 'plots')
os.makedirs(plots_folder, exist_ok=True)

# take a peek inside
print(os.listdir())

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

print(df.head().to_markdown())
print(df_test.head().to_markdown())

# grouping them by gender and survival
surv_count_gender = df.groupby(['Sex', 'Survived']).size().unstack()
surv_count_gender = surv_count_gender.div(surv_count_gender.sum(axis=1), axis=0)

if plots == 1:
    # plot and titles
    surv_count_gender.plot(kind='bar', stacked=True)
    plt.title('Survival Rate by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Rate')
    plt.xticks(rotation=0)
    plt.legend(['Died', 'Survived'])
    plot_path = os.path.join(plots_folder, 'survival_rate_by_gender.png')
    plt.savefig(plot_path)
    plt.show()

    # age histogram
    plt.figure()
    # set bins and PLOT ME
    age_bins = np.arange(0, df['Age'].max() + 5, 5)
    plt.hist(
        [df[df['Survived'] == 0]['Age'].dropna(),
         df[df['Survived'] == 1]['Age'].dropna()],
        bins=age_bins,
        stacked=True,
        label=['Died', 'Survived']
    )
    # titles
    plt.title('Survival by Age')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.legend()
    # show and save
    age_plot_path = os.path.join(plots_folder, 'survival_by_age_histogram.png')
    plt.savefig(age_plot_path)
    plt.show()

    # side by side histogram
    plt.figure()

    # bins and age groups again
    age_bins = np.arange(0, df['Age'].max() + 5, 5)
    died_counts, _ = np.histogram(df[df['Survived'] == 0]['Age'].dropna(), bins=age_bins)
    survived_counts, _ = np.histogram(df[df['Survived'] == 1]['Age'].dropna(), bins=age_bins)

    # calculate bin centers for positioning
    bin_centers = 0.5 * (age_bins[1:] + age_bins[:-1])
    bar_width = 2

    # plotting the bars side by side
    plt.bar(bin_centers - bar_width / 2, died_counts, width=bar_width, label='Died')
    plt.bar(bin_centers + bar_width / 2, survived_counts, width=bar_width, label='Survived')

    # adding titles and labels
    plt.title('Survival by Age - Side by Side Comparison')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.legend()

    # Save the plot
    side_by_side_plot_path = os.path.join(plots_folder, 'survival_side_by_side_by_age.png')
    plt.savefig(side_by_side_plot_path)

    # show the plot
    plt.show()


# engineering small features, family size
df['Familysize'] = df['SibSp'] + df['Parch']

# extract the title. This function is ghoulish, it needs to be an r function for escapes:
# , matches the comma after the last name
# \s* matches any whitespace following the comma
# ([^\.]+) is a capture group that matches and the character that are not a dot (this is the title)
# \. matches the dot after the title

df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')
df_test['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')

# print the titles for feature engineering
print(df['Title'].unique())
print(df['Title'].nunique())

# create a mapping that includes gender-specific categories
title_mapping = {
    'Mr': 'male_standard',
    'Mrs': 'female_standard',
    'Miss': 'female_standard',
    'Master': 'male_standard',
    'Ms': 'female_standard',
    'Mlle': 'female_standard',
    'Mme': 'female_standard',
    'Dona': 'female_standard',
    'Don': 'male_aristocracy',
    'Sir': 'male_aristocracy',
    'Lady': 'female_aristocracy',
    'The countess': 'female_aristocracy',
    'Jonkheer': 'male_aristocracy',
    'Rev': 'male_professional',
    'Dr': 'male_professional',
    'Major': 'male_professional',
    'Col': 'male_professional',
    'Capt': 'male_professional'
}

# apply the updated title mapping
df['TitleCat'] = df['Title'].map(title_mapping)
df_test['TitleCat'] = df_test['Title'].map(title_mapping)

# print to check the mapping
print(df[['Title', 'TitleCat']].sample(5))


# apply the feature engineering also to the training set
df_test['Familysize'] = df_test['SibSp'] + df_test['Parch']


# more general FE
# is alone?
df['IsAlone'] = (df['Familysize'] == 0).astype(int)
df_test['IsAlone'] = (df_test['Familysize'] == 0).astype(int)

# cabin deck (first letter of cabin might be deck)
df['CabinDeck'] = df['Cabin'].str[0]
df_test['CabinDeck'] = df_test['Cabin'].str[0]

# age bins
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100],
                        labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
df_test['AgeGroup'] = pd.cut(df_test['Age'], bins=[0, 12, 18, 35, 60, 100],
                             labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])

# we have the survival rate for each embarkation point, we can use the pre calculated number on the test set to see if it helps
embarked_survival_rate = df.groupby('Embarked')['Survived'].mean()
df['EmbarkedSurvivalRate'] = df['Embarked'].map(embarked_survival_rate)
df_test['EmbarkedSurvivalRate'] = df_test['Embarked'].map(embarked_survival_rate)


# spearman corr and kw test
if stat_anal == 1:
    columns_num = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch', 'Familysize',
                   'EmbarkedSurvivalRate']
    for column in columns_num:
        non_null_data = df[['Survived', column]].dropna()
        correlation, p_value = spearmanr(non_null_data['Survived'], non_null_data[column])
        print(f"Spearman correlation between 'Survived' and '{column}': {correlation:.3f} (p-value: {p_value:.3f})")

    columns_cat = ['Pclass', 'Sex', 'CabinDeck', 'Embarked', 'Title', 'TitleCat',
                   'IsAlone', 'AgeGroup']

    for column in columns_cat:
        non_null_data = df[['Survived', column]].dropna()
        groups = [non_null_data[non_null_data['Survived'] == status][column] for status in
                  non_null_data['Survived'].unique()]
        stat, p_value = kruskal(*groups)
        print(f"Kruskal-Wallis test for 'Survived' and '{column}': H-statistic = {stat:.3f}, p-value = {p_value:.3f}")



# imputation
columns_for_impute = df.columns.difference(['Age', 'Survived'])

age_imputer = IterativeImputer(max_iter=10, random_state=117)
mode_imputer = SimpleImputer(strategy='most_frequent')

# mode impute
df[columns_for_impute] = mode_imputer.fit_transform(df[columns_for_impute])
df_test[columns_for_impute] = mode_imputer.transform(df_test[columns_for_impute])

# iterative impute
df['Age'] = age_imputer.fit_transform(df[['Age']])
df_test['Age'] = age_imputer.transform(df_test[['Age']])



# print missings
print("Missing values in df (training set):")
print(df.isnull().sum())
print("\nMissing values in df_test (test set):")
print(df_test.isnull().sum())

# gradient boost tree model


# encode categorical variables
label_encoders = {}
for column in ['Sex', 'Embarked', 'TitleCat', 'CabinDeck', 'AgeGroup']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    df_test[column] = le.transform(df_test[column].astype(str))
    label_encoders[column] = le

# select features and target variable for training
features = ['Pclass', 'Sex', 'Fare', 'Familysize',
            'Embarked', 'TitleCat', 'AgeGroup']

X = df[features]
y = df['Survived']

# split into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=117)

# create the gradient boost model, and train it
model = GradientBoostingClassifier(n_estimators=30, learning_rate=0.1, max_depth=3, random_state=117)
model.fit(X_train, y_train)

# validate the model
y_pred = model.predict(X_val)
print(f"Validation Accuracy: {accuracy_score(y_val, y_pred)}")
print(classification_report(y_val, y_pred))
print(classification_report(y_val, y_pred))

# predict on the test set
X_test = df_test[features]
df_test['Survived'] = model.predict(X_test)

# show the first few predictions
print(df_test[['Survived']].head())



# create submission
submission = df_test[['PassengerId', 'Survived']]
submission.to_csv(os.path.join(folder, dir, 'predictions.csv'), index=False)
