# An Exploratory Analysis on the Correlation between Depression, Alcoholism and BMI

Try out the Depression Risk Assessment web app by clicking on the button below:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://depression-risk.herokuapp.com/)

https://depression-risk.herokuapp.com/

## ABSTRACT

We performed an exploratory analysis on the correlation between the mental and physical health of people. Specifically, our efforts were to identify the existence of any pattern in our 
data using machine learning techniques, to see if we can find any clusters that give us insights on how body mass index (BMI), alcohol consumption and depression can be related, 
and also to identify a risky group if possible.

## 1. INTRODUCTION

We often make a clear distinction when we talk about mind and body. But the fact is, we cannot talk
about one entity, ignoring the other. Poor physical health can lead to an increased risk of developing
mental health problems. Similarly, poor mental health can negatively impact on physical health, leading
to an increased risk of some conditions.

Depression has been found to be associated with an increased risk of coronary heart diseases. People
with the highest levels of self-rated distress were 32% more likely to have died from cancer. Recent
research conducted in the field of healthcare services shows that one reason for the increase in
respiratory disease, heart disease and cancer risk is that individuals with mental health conditions are
less likely to seek care for their physical health. The Mental Health Foundation reported that those who
take part in mental health services are statistically less likely to receive many routine checks, such as
weight, cholesterol and blood pressure, that could identify health concerns early on. Unhealthy habits,
including smoking, drug use, alcoholism and a lack of exercise, can also play a vital role, according to
an article from U.S. News and World Report [ 1 ].

We attempted to explore the correlation between alcohol consumption, body mass index and
depression. We considered the “National Health and Nutrition Examination Survey (NHANES),
2005 - 2006 ” [ 2 ] survey data for our analysis. The NHANES survey design is a stratified, multistage
probability sample of the civilian noninstitutionalized United States population. Out of the many tables
available as part of this survey, we are concentrating on 3 tables primarily for our purposes:

![image](https://user-images.githubusercontent.com/14270270/156234794-32790c89-142d-40d3-817f-640c848abae8.png)

## 2. MOTIVATION

There are a lot of studies and academic research going on, analyzing and addressing the growing
concerns of mental illnesses, alcoholism, drug abuse etc. These studies show that there’s a strong link
between serious alcohol use and depression. Alcohol is a depressant. That means any amount you
drink can make you more likely to get the blues. Drinking a lot can harm your brain and lead to
depression. People who are depressed and drink too much have more frequent and severe episodes of
depression, and are more likely to think about suicide.

It is truly remarkable, that each of these problems do get a lot of individual attention from society and
the government, which is absolutely required. But rather than addressing each of these problems
individually, would a holistic approach to understand the mindset of people, who are driven towards
alcoholism or leading an unhealthy lifestyle serve the community better? The promotion of positive
mental health can often be overlooked when treating a physical condition. Can we identify a pattern
and would it aid us to target our efforts and focus towards helping the population falling under the
risk group? Our aim is to present this correlation pattern, based on statistical evidence obtained by
employing machine learning techniques.


## 3. APPROACH & IMPLEMENTATION

### Data Cleaning:

The survey data was available as tab separated values. After identifying the needed tables, we loaded
the tables to Pandas data frames. Here, we present a brief description of our original data frames,
some visualizations and statistics to show the distribution of the data and lists some of the key data
cleanup activities we have performed.

- **Dropping redundant survey metadata columns from the data frames** : All the three
    tables chosen for the project contain the survey metadata columns. We will remove all the
    metadata columns from alcohol and depression tables and keep only the relevant metadata
    columns in the third table (weight), so that we'll just have one copy of these relevant columns
    in the final merged data frame.
    
- **Renaming data frame columns** : As the data frames loaded from the survey data have so
    many columns and each of these columns are named based on the survey questionnaire
    numbers, we are replacing the data frame column names to more meaningful names, which
    can be easily interpreted.
    
- **Dropping rows with missing data for all relevant columns** : In this step, we are
    removing the rows with missing data for all the relevant columns, for each of our data frames.
    We are setting the threshold limit to 2, which means all the rows which have values for at least
    2 columns (SEQUENCE_NUMBER and any other column) would be retained, because
    SEQUENCE_NUMBER will definitely have a value for all the rows.
    ![image](https://user-images.githubusercontent.com/14270270/156234903-e41e727f-3d6c-491c-9142-9e226cc413f2.png)

- **Standardizing feature values based on the unit of measurement** : Some of the
    columns in the Alcohol tables have data of varying units of measurements. For example, for
    some rows, the number under a certain column may depict the number of days per week when
    the participant had a drink, whereas for another row the same column could depict the number
    of days per month when the participant had a drink. In this step we are recalculating the
    feature values of such columns based on the unit of measurements (depicted in a separate
    column) and standardizing the feature values to number of days per year.

    ![image](https://user-images.githubusercontent.com/14270270/156235024-193e3430-bf77-42c9-88b7-021fc3e75e13.png)

- **Calculating missing data in Alcohol data frame based on the related features** : In
    this step, we are further removing the missing values, based on deductions from related
    features. For example, if the person had filled "yes" for having at least 12 drinks in the previous
    year, would imply that he/she has had at least the same number of drinks in his/her lifetime.
    
- **Dropping columns with more than 30% missing data** : In this step, we are removing
    columns with more than 30% missing data, as per the common norms, without letting it
    impact our overall data quality. Before handling the missing values, it is very important to
    explore them in the context of our data frame. We are using Missingno library to provide us
    with the distribution of missing values in our data frame by informative visualizations. Using
    the plots of Missingno, we are able to see where the missing values are located in each column.
    Then we calculate the percentage of missing data in each column, identify the columns above
    our threshold and drop them.
    ![image](https://user-images.githubusercontent.com/14270270/156235078-d46d62fe-d4e6-490a-8b2a-ae1c17b5b670.png)

- **Calculate BMI and replace height** : We have identified that the body mass index of a
    person is a clear indicator of whether a person is obese or not, hence it should be calculated
    and included as a relevant feature. In this step, we calculate the BMI of the participant using
    the recorded height and weight of the participant, and removing the height column later on, as
    the BMI feature comes in to picture and height column becomes redundant.
    ![image](https://user-images.githubusercontent.com/14270270/156235121-a57bd373-4419-477f-ae9b-9329ce9bcbe5.png)


- **Getting Weight data frame info and dropping non-relevant features** : After doing the
    preliminary cleanup tasks, we are getting the weight data frame info to check the extent of
    missing data and to see if we can further eliminate non-relevant features. There are columns
    depicting weight loss methods tried by the participant, the largest weight he has gained etc.
    most of which are redundant and non-relevant information.
    
- **MICE Imputation** : Finally, we are using MICE imputation to replace missing values, which
    models each feature with missing values as a function of other features, and uses that estimate
    for imputation.

### Exploratory Data Analysis:

- **Finding Covariance between each feature** :

This was done to identify how variables vary with each other.

![image](https://user-images.githubusercontent.com/14270270/156235164-6152aecc-a2ba-4729-aef9-7df542963824.png)


**Outcome** :
We could understand the dependency directions between the features. Some features are found to
have positive dependencies and few have slightly negative dependency. There are also features which
have almost zero dependency shared between them.

- **Plotting bar graph for exploring if there's a relationship between the annual**
    **family income and depressing thoughts** :

We hypothesize that lesser annual family income would add on to the stress/depression of a person.
We have plotted this visualization to test our hypothesis.

![image](https://user-images.githubusercontent.com/14270270/156235277-02becb24-0635-4dcc-b558-671d25820a7c.png)

**Outcome** :
As expected, the surveyees falling in the lesser risk group have a higher mean annual family income,
substantiating our hypothesis. Value 1 for the RISKY_GROUP attribute => having frequent/intense
dying/depressing thoughts. Lower income would likely add on to the depressing thoughts a person is
having, and could be one of the factors contributing towards his/her state of mind.

- **Finding correlation between each feature:**

Our aim is to find the impact of change on each variable with each other.

![image](https://user-images.githubusercontent.com/14270270/156235331-ec43854c-2708-4eaa-b690-eaaf14b8bc35.png)


**Outcome** :
We have plotted the correlation matrix to measure the strength of dependencies between the
variables. Identified the strongly correlated and weakly correlated features.

- **Plotting bar graph for exploring if there's a relationship between the Body Mass**
    **Index of a person and the frequency/intensity of depressing thoughts:**

Our aim is to find the correlation between obesity and depressing thoughts. We have already created
an attribute (RISKY_GROUP) to categorize the surveyees into 2 groups. Value 1 for this attribute
implies the person is having frequent/intense dying/depressing thoughts. Hence, we are plotting this
attribute against BMI. We are plotting this visualization to test our hypothesis.

![image](https://user-images.githubusercontent.com/14270270/156235424-850b332e-436b-4e94-b93c-a0b56886f09f.png)


**Outcome** :
As expected, the group of people identified as riskier (value 1 => having frequent/intense
dying/depressing thoughts), have a higher average body mass index, substantiating our hypothesis.

- **Calculating the weight change in the past 1 year and its impact on the depressing**
    **thoughts** :

We hypothesize that a person who has moved on towards a healthier lifestyle will be having fewer
suicidal/depressing thoughts. We are plotting this visualization to test our hypothesis.

![image](https://user-images.githubusercontent.com/14270270/156235460-a0df028c-da6b-49c5-bb8c-493414995527.png)


**Outcome** :
As expected, the riskier group (value 1 => having frequent/intense dying/depressing thoughts) has
had a positive increase in their body weight on an average in the past 1 year, whereas the less risky
group has, on an average, a decrease in their body weight.

- **Plotting bar graph for exploring if there's a relationship between the number of**
    **family members and drinking habits** :

We hypothesize that a person living alone is much more prone to alcohol addiction, when compared
to a person who is part of a larger family. We are plotting this visualization to test our hypothesis.

![image](https://user-images.githubusercontent.com/14270270/156235491-c8460fda-0348-4bc7-b670-826a971a9941.png)

**Outcome** :
As expected, the surveyees with greater number of family members had remarkably fewer drinks in
the past year, substantiating our hypothesis. The insight would probably come in handy when
identifying the risky group.

### Applying ML models and Evaluation:

**Experiment 1: Checking for clusters**

Our initial hypothesis was to identify the existence of any pattern in our data using unsupervised
machine learning, to see if we can find any clusters that give us insights on how weight, depression and
alcohol consumption can be related, and also to identify a risky group if possible.

For this, we employed KMeans Clustering Algorithm. KMeans can be used to confirm assumptions
about what types of groups exist or to identify unknown groups in complex data sets.

![image](https://user-images.githubusercontent.com/14270270/156235519-92b5c1c4-6302-46f1-8368-064e3ab0f8bc.png)


Applying KMeans Clustering algorithm did not help identify any significant cluster formation with the
chosen parameters. The cluster formed through plotting wasn’t very explanatory.
Hence, we shifted our focus to predicting a Risky Group through Classification algorithms.

**Experiment 2: To predict Risky Group**

Risky group is a categorical column and it has only 2 possible values, either 1 or 2. Having 2 value there
indicates that the person is highly risky. 1 value indicates that the person is not risky. This column was
derived from columns belonging to the Depression table in the initial data set.

![image](https://user-images.githubusercontent.com/14270270/156235547-1f0aec40-c6f1-4c29-9e23-30583cac4079.png)

Various Classification models were employed in predicting if a person is risky or not based on the final
clean dataset. Below is a sneak peek on the final dataset.

![image](https://user-images.githubusercontent.com/14270270/156235569-10fcde40-6ff7-4d97-a601-c3cc2bd2e4d0.png)

- **Naive Bayes Classification Model**

Risky group is a categorical variable predicted based on given Body Mass Index value, age, drinking
habits etc. Naive Bayes Regression is a great model to predict categorical value under given
circumstances. The dataset was split into training data (75%) and test data (25%).

![image](https://user-images.githubusercontent.com/14270270/156235610-2208ea33-6ecf-4876-b79f-34a2d8d9fdd4.png)

![image](https://user-images.githubusercontent.com/14270270/156235638-e52f8575-ea94-49f9-864a-4a3160897221.png)


Modeling data with Naive Bayes has produced an accuracy of 84.64% compared to 84.34%. This shows
the possibility of a risky group forming from the sample set. The correlation between alcoholism,
obesity and depression is very strong.

![image](https://user-images.githubusercontent.com/14270270/156235712-9ab37b20-56f6-43a3-8a54-3174eed7659f.png)

![image](https://user-images.githubusercontent.com/14270270/156235725-c452e762-fd74-4559-a677-4904a4811f18.png)

The figures above show the confusion metrics and accuracy of model.


- **K-NN Classification Model**

The dataset was split into training data (75%) and test data (25%). Risky group can be identified well
using the Logistic Regression machine learning algorithm with an accuracy of 84 percent given the age,
body mass index, gender and drinking habits of an individual. This shows the correlation between these
factors is really strong and a risky group is indeed formed with greater the impact of the factors
considered.

![image](https://user-images.githubusercontent.com/14270270/156235759-ec01c22f-0176-4426-bb2a-bf601dd817a1.png)

![image](https://user-images.githubusercontent.com/14270270/156235769-ab2a643f-6545-4c6d-aa5a-63d115db14e6.png)

- **Logistic Regression Classification Model**

The k-nearest neighbor classification (k-NN) is one of the most popular distance-based algorithms. This
classification is based on measuring the distances between the test sample and the training samples to
determine the final classification output. The k-NN classifier works naturally with numerical data. Our
dataset is numerical and the feature we are trying to predict is Risky Group, a categorical column.

The dataset was split into training data (75%) and test data (25%). The model could predict Risky Group
but accuracy rate is lower compared to previous models, 82.36%.

![image](https://user-images.githubusercontent.com/14270270/156235810-23bce151-76ae-41c6-8fcf-010c6a391a80.png)

![image](https://user-images.githubusercontent.com/14270270/156235827-25e3933c-4eea-4277-9c32-220cd7d050cd.png)


- **SVM Classification Model**

SVM looks at extreme cases very close to boundary and uses that to construct analysis. SVM tries to
find the “best” margin (distance between the line and the support vectors) that separates the classes
and this reduces the risk of error on the data.


The dataset was split into training data (75%) and test data (25%). The model could predict Risky Group
but accuracy rate is lower compared to previous models, 84.83%.

![image](https://user-images.githubusercontent.com/14270270/156235844-fa0d323b-2823-42eb-a8a9-1eedbc08e9f5.png)

![image](https://user-images.githubusercontent.com/14270270/156235860-a06e85c9-0c38-479f-8e8d-7f593929b0c9.png)


**Analysis**

All of the above experiment result shows that a person can be classified as risky or non-risky given their
age, Body Mass Index, drinking pattern, their family income, number of people they stay with etc. Below
are some plots that show the output of Classification Models. Only selected models are chosen as some
didn’t have enough output to give a good plot.

**Naive Bayes Modeling Output**

![image](https://user-images.githubusercontent.com/14270270/156235894-69688126-f6cd-4f22-968a-e5ad65d97844.png)

![image](https://user-images.githubusercontent.com/14270270/156235918-4080c12b-2f4f-4191-8699-b9374c672867.png)


The higher the BMI and drinking habits of a person irrespective of age, the more riskier they are. It is
shown in the plots above. This is as we hypothesized, that the lifestyle of a person indeed has a huge
impact on their mental health.

**KNN Classification Modeling Output**

![image](https://user-images.githubusercontent.com/14270270/156235951-4788b62a-046a-4a4a-bcde-698fa8802acc.png)

![image](https://user-images.githubusercontent.com/14270270/156235967-4b180c4e-b34c-4d03-ac69-1951ebe022b7.png)


**SVM Classification Modelling Output**

![image](https://user-images.githubusercontent.com/14270270/156235992-ea829f58-d021-49eb-8d56-1aeed0b1d822.png)

![image](https://user-images.githubusercontent.com/14270270/156236012-116900b1-891c-4be0-b5e1-10e1a6f37f73.png)


Another exploration we tried is applying SVM modeling to observe the impact on a person’s mental
health, with a stronger family and steady annual family income. The below graph shows how they are
correlated.

![image](https://user-images.githubusercontent.com/14270270/156236045-19c7db18-2cb6-4c65-b101-a9f98a4d3684.png)

This was one of our hypotheses, that, a greater number of people someone stay with, especially your
family, the stronger he/she will be. The same is observed as per the last plot (from above figure).

## 4. CONCLUSION

We examined the survey data to identify the existence of any pattern in our data using machine
learning techniques, to see if we can find any clusters that give us insights on how body mass index
(BMI), alcohol consumption and depression can be related, and also to identify a risky group if
possible. We applied several models to see which fits best and Naïve Bayes classifier model suits best
and has the best accuracy. The fundamental Naive Bayes assumption is that each attribute makes an
independent equal contribution to the outcome. Naive Bayes classifiers tend to perform especially
well in one of the following situations: when the naive assumptions actually match the data (in our
case it did) and for very well-separated categories, when model complexity is less important. Our
model is capable of classifying an individual given his drinking habits and physical attributes like
height and weight. Hence, we could validate our hypothesis that your lifestyle has a significant impact
on your mental health.

## 5. REFERENCES

[ 1 ] US News - https://health.usnews.com/health-news/articles/2012/04/27/can-your-mental-health-affect-your-longevity

[ 2 ] Inter-university Consortium for Political and Social Research - https://www.icpsr.umich.edu/web/ICPSR/studies/25504/datadocumentation

[ 3 ] scikit-learn - https://scikit-learn.org/stable/

[ 4 ] https://towardsdatascience.com/sentiment-analysis-introduction-to-naive-bayes-algorithm-96831d77ac


