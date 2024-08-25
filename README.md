#datascience 
# 1 - Examining the data
## 1.1 download
the first step in this project was to examine the data, easy peasy. We download the file, paste it somewhere in our computer, and we begin looking through it. I am using PyCharm IDE, but the principle would be the same regardless of how you're doing this. The competition gives us two datasets, train.csv and test.csv. these are practically the same, but train includes whether or not the passenger died, and test does not, as that is the objective of the challenge. 

## 1.2 data exploration
a good first step is data exploration. First thing I do is to print the head of the data. 
### Training set

|     | PassengerId | Survived | Pclass | Name                                                | Sex    | Age | SibSp | Parch | Ticket           |    Fare | Cabin | Embarked |
| --: | ----------: | -------: | -----: | :-------------------------------------------------- | :----- | --: | ----: | ----: | :--------------- | ------: | :---- | :------- |
|   0 |           1 |        0 |      3 | Braund, Mr. Owen Harris                             | male   |  22 |     1 |     0 | A/5 21171        |    7.25 | nan   | S        |
|   1 |           2 |        1 |      1 | Cumings, Mrs. John Bradley (Florence Briggs Thayer) | female |  38 |     1 |     0 | PC 17599         | 71.2833 | C85   | C        |
|   2 |           3 |        1 |      3 | Heikkinen, Miss. Laina                              | female |  26 |     0 |     0 | STON/O2. 3101282 |   7.925 | nan   | S        |
|   3 |           4 |        1 |      1 | Futrelle, Mrs. Jacques Heath (Lily May Peel)        | female |  35 |     1 |     0 | 113803           |    53.1 | C123  | S        |
|   4 |           5 |        0 |      3 | Allen, Mr. William Henry                            | male   |  35 |     0 |     0 | 373450           |    8.05 | nan   | S        |
### Testing set

|    |   PassengerId |   Pclass | Name                                         | Sex    |   Age |   SibSp |   Parch |   Ticket |    Fare |   Cabin | Embarked   |
|---:|--------------:|---------:|:---------------------------------------------|:-------|------:|--------:|--------:|---------:|--------:|--------:|:-----------|
|  0 |           892 |        3 | Kelly, Mr. James                             | male   |  34.5 |       0 |       0 |   330911 |  7.8292 |     nan | Q          |
|  1 |           893 |        3 | Wilkes, Mrs. James (Ellen Needs)             | female |  47   |       1 |       0 |   363272 |  7      |     nan | S          |
|  2 |           894 |        2 | Myles, Mr. Thomas Francis                    | male   |  62   |       0 |       0 |   240276 |  9.6875 |     nan | Q          |
|  3 |           895 |        3 | Wirz, Mr. Albert                             | male   |  27   |       0 |       0 |   315154 |  8.6625 |     nan | S          |
|  4 |           896 |        3 | Hirvonen, Mrs. Alexander (Helga E Lindqvist) | female |  22   |       1 |       1 |  3101298 | 12.2875 |     nan | S          |

So among the other information we have, we can see that the training set contains an extra column which is not present in the test set; Survived. This is the column we are predicting so this makes sense.  We also have the sex, age, SibSp (siblings or spouses) and Parch (parents or children) denoting how many people the passenger is travelling with. We also have Class, Cabin, Fare and Embarked. I suspect Ticket number will not correlate to survival chance, but we shall see. 

Another important part of EDA (exploratory data analysis) is to determine how much of the dataset is missing or null values. 
![[Pasted image 20240825212522.png]]
Most of the data is present, however cabin has a significant number of missing entries. This could be because Cabins were only assigned to the wealthiest passengers, or perhaps not recorded for many passengers. Age and embarkation also have a few missing values, but we can fix this later. 

## Visualising relationships
### Gender
We've all seen titanic, right? We know about women and children first, right? My first thought is, do women have a higher general survival rate than men?
![Figure_1](https://github.com/user-attachments/assets/000e11a8-2135-41d9-8508-7045cc16df58)

It looks like they do! This graph is unsatisfying though, as many more men travelled than women, by the looks of things. We want to turn this into a ratio, in order to clear up what's going on here. 
![Figure_2](https://github.com/user-attachments/assets/a7fa2359-7bc9-418e-a194-ffa499872ef4)

Perfect, we have rescaled this data to show exactly what we want to see. 

### Age
Next up, lets look at age. We start by looking at a histogram of the ages, by survived and died:
![survival_by_age](https://github.com/user-attachments/assets/c8a43567-f8ac-4903-89e2-738b460121b2)

wow, that's not as helpful as I thought, although it does appear that people at the ends of the spectrum are more likely to survive.

## Statistical Analysis
While these plots are helpful for visualising what is going on here, a more statistical approach is helpful, especially with some of the categorical columns having many different values. There are two tests we will do, which should cover all the columns. We will use the Spearman Correlation test on the numerical columns, and the Kruskal-Wallis test on the categorical ones. I like these tests because they're non parametric (do not assume a normally distributed linear spread of data), and we can use it on both numerical variables like age, and ordinal ones like ticket class.

The Spearman test provides us with a correlation coefficient, which gives us the strength of a monotonic relationship, as well as their direction. The null hypothesis is that there is no correlation. For the categorical variables (embarkation, sex as examples) we will use the Kruskal-Wallis test. This test doesn't give a correlation coefficient like the Spearman test, as it instead tests whether the values have different distributions. The test statistic denotes how different these distributions are, and the p value 

If the p-value of any of these tests are below 0.05, then we can say that the null hypothesis is disproved, and therefore, there is statistical significance. Basically, when it comes to p value, low number = good. 
![Pasted image 20240824211455](https://github.com/user-attachments/assets/f7133f15-a3ae-4296-92e3-aad8eb0eae4f)

Spearman Correlation shows Age is not correlative, however we know this is probably because it's not monotonic, i.e. it doesn't improve chances linearly, since low age and high age people are more likely to survive. The fact that the men and women have drastically different survival chances may also have something to do with this. From the categorical variables, only cabin seems to have a non significant effect. We have class in with the categorical variables because it's an ordinal variable, which the KW test can handle well. 

## Feature Engineering
our next step is going to be feature engineering. That is taking features we already have, and attempting to make new features, which we can then test with our statistical methods, to see if they are more indicative of survival chances. 

### Title Category
The first feature we create is title, a good way to separate more wealthy passengers from poorer ones (in theory). We use a regex to extract the titles:
![Pasted image 20240824211455](https://github.com/user-attachments/assets/797719a1-7937-478a-b9b0-bf87738e21b2)

giving us 17 unique titles. We are going to separate them into 6 groups, Standard, Professional and Aristocracy, each for men and women. The idea is to separate men and women into richer and poorer passengers, as we know that there is a difference between survival rates in the data.
![Pasted image 20240825205219](https://github.com/user-attachments/assets/73186dd6-78aa-4b70-90a7-b8d21ea99104)

We have made a single assumption here, that all doctors are male. This is probably not true at all, but there are only 10 doctors in the whole data, so it probably won't matter a lot. If we wanted to get into the nitty gritty we could separate them based on the recorded gender of the passenger, but that is probably not necessary.

### Other Features
Our goal here is to introduce a number of other features which can help with our prediction. We don't actually know what's going to work, so it's a case of throwing things at the wall, seeing what's significant, and taking those options. We can use 'domain knowledge' to try and guess what we can do for feature engineering, but this is an art, not a science. Some ideas I came up with are as follows:
- Family size: totals up the Parch and the SibSp columns to determine total family size
- IsAlone: Binary flag for if family size == 0
- age bins: we split the ages into bins, to hopefully simplify the model
- EmbarkedSurvivalRate: we can take the pre calculated survival rate from each embarkation point and use that as a feature. (we can do this for other variables, but I don't want to spend too much time on this project.)
- Cabin Deck: we take the first letter from each cabin, and assume it corresponds to a deck. Perhaps if you're closer to a lifeboat you're more likely to escape?

Afterwards, we do another round of Kruskal-Wallis and Spearman testing to determine if any of our new features are statistically significant. We can then discard any features that are not significant, and proceed with the next step.
![Pasted image 20240825213350](https://github.com/user-attachments/assets/14a1d451-76dc-4f6a-883b-a7aaa4972032)

We also have to remember to apply the feature engineering to the test set, not just the training set, in order to keep our variables consistent across the two datasets. 

# 2 - Pre Processing
A crucial part for any machine learning task is pre processing the data that we use. The first step in the pre-process pipeline is imputation. This is taking data which is missing, and replacing it with a value derived from the rest of the dataset. For us, there are very few missing values, as we can see from the previous analysis that 'cabin', the column with the most missing values, is not significant, even if we extract additional information, so we can simply drop the column. We only have 3 total columns to process, 'age' and 'embarked' from the training set, and 'age' and 'fare' from the test set. First, to tackle fare and embarked, since the amount of missing values is so low, we are just going to go with a mode impute. This is a clumsy and imprecise method, but as the missing data is so rare (3 total cases), this shouldn't be a big issue. 

The next step is to perform an iterative imputation on the age column. This is a more complex imputation method, which uses a Bayesian estimator to predict the missing value. This is ML² at this point! This should hopefully give us a more accurate prediction for the age, rather than using a mean/median imputation method which would be a bit more clumsy. This should allow the data to be more predictable. As this is a small dataset, it takes almost no time at all, although in prior projects I have done, this method can be very time consuming if performed on huge datasets. As before, these methods need to be repeated for the training and testing sets. 

The final step in our pre-process pipeline is to encode all the categorical variables as strings, so the machine doesn't try and predict them as numbers when they're not. We are using a simple label encoder for this task. I should have probably tried a one-hot encoding, but I did this whole challenge in one afternoon, so I decided to go with a label encoder for simplicity. 

# 3 - Machine Learning
The part we've been waiting for! We're going to use a gradient boosted trees setup for this for 3 reasons;
1. it's very easy to set up
2. it's fast
3. it's accurate

In past work I've done, the gradient boosted trees method has performed very well given it's ease of use and it's simplicity. I will be leaning on that here, since I want the method to be as simple and quick as possible.
## Grid search
one thing we will be doing in order to improve our accuracy, is a grid search over the hyperparameters. We will change the number of trees, the depth and the learning rate. We are choosing a wide range for these, as the whole learning process is completed in less than one second, so we can crunch out a heap of them, very quickly.

Once we've split the training data into further training and test sets, we run the model for all the parameters we've given, and let the machine do it's work. We find that 110 estimators, with 0.1 learning rate and a max depth of 4 gives us the best results. We apply that version of the model to the test set, and end up with our final prediction file, which we submit to kaggle. ![[Pasted image 20240825224711.png]]
This gives us a score of 0.79%, which for a single afternoons work, isn't as terrible as it could be. 

# 4 - Final thoughts
So this is a nice easy data science challenge, which proved to be quite simple. If I was to do it again, there are a number of ways we could refine it, such as using more sophisticated learning techniques, and improved feature engineering and data management. I'm not sure how much higher a score could be either, since there will be a degree of randomness involved with the data; some poor young men would survive, and some wealthy older women would die, it's simply in the nature of the data. I've had a look at the leader board, and there are a lot of 100% scores. I don't know if that's as a result of skulduggery or if there is a legitimate way to get 100% of the predictions correct. Anyway, I think next time I have a few days to spare with no data science projects running, I might try and improve my score in this, hopefully getting above 80%, although 85% would be a good target I think.
