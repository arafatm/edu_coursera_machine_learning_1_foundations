# MACHINE LEARNING FOUNDATIONS

https://www.coursera.org/specializations/machine-learning

## WEEK 1 INTRODUCTION

### THE MACHINE LEARNING PIPELINE

    Data -> ML Method -> Intelligence

### CASE STUDY APPROACH

Use various ML Methods in different case studies

- **Regression** Case Study 1: Predicting house prices

- **Classification** Case Study 2: Sentiment Analysis

- **Clustering** Case Study 3: Document retrieval

- **Matrix Factorization** Case Study 4: Product recommendation

- **Deep Learning** Cases Study 5: Visual product recommender

### Overview

Wk 2 Regression
- Case Study: Prediction house prices
- Models
  - linear regression
  - Regularization: Ridge (L2), Lasso (L1)
- Algorithms
  - Gradient descent
  - Coordinate descent
- Concepts
  - Loss functions
  - bias-variance tradeoff
  - cross-validation
  - sparsity
  - overfitting
  - model selection

Wk 3 Classification
- Case study: Analyzing sentiment 
- Models:
  - Linear classifiers (logistic regression, SVMs, perceptron)
  - Kernels
  - Decision trees
- Algorithms
  - Stochastic gradient descent
  - Boosting 
- Concepts
  - Decision boundaries
  - MLE ensemble methods
  - random forests
  - CART
  - online learning 

Wk 4 Clustering & Retrieval
- Case study: Finding documents
- Models
  - Nearest neighbors 
  - Clustering, mixtures of Gaussians 
  - Latent Dirichlet allocation (LDA)
- Algorithms
  - KD-trees, locality-sensitive hashing (LSH)
  - K-means
  - Expectation-maximization (EM)
- Concepts
  - Distance metrics
  - approximation algorithms
  - hashing
  - sampling algorithms
  - scaling up with map-reduce 

Wk 5 Matrix Factorization & Dimensionality Reduction
- Case study: Recommending Products 
- Models:
  - Collaborative filtering
  - Matrix factorization
  - PCA
- Algorithms
  - Coordinate descent
  - Eigen decomposition
  - SVD Algorithms
- Concepts
  - Matrix completion
  - eigenvalues
  - random projections
  - cold-start problem
  - diversity
  - scaling up 

Wk 6 Capstone: An intelligent application using deep learning

### GETTING STARTED WITH PYTHON

[Install anaconda, GraphLab, ipython 
notebook](https://turi.com/download/install-graphlab-create.html)

To run notebooks once above is installed
```
source activate gl-env
conda update pip
jupyter-notebook --no-browser
```

[First Notebook](week_1/Getting started with iPython Notebook.html)
- [source](week_1/Getting started with iPython Notebook.ipynb)

Basic types
```python
i = 4 # int
f = 4.1
b = True

type(i)
```

Advanced types
```python
l = [3, 1, 2] # list

d = {'foo':1, 'bar':2.3, 's':'string'} # dictionary

print d['foo']

n = None # null
type(n) # NoneType
```

Advanced printing
```python
print "Our float value is %s. Our int value is %s." % (f,i)
```

Conditional statements
```python
if i == 1 and f > 4:
  print "i == 1 and f > 4"
elif i > 4 or f > 4:
  print "(i or f) > 4"
else:
  print "(i and f) <= 4"
```

Loops
```python
print l

for e in l:
  print e

counter = 6
while counter < 10:
  print counter
  counter += 1
```

functions
```python
def add2(x):
  return x + 2
```

lambdas
```python
square = lambda x: x*x
square(3)
```

### GETTING STARTED WITH SFRAME AND GRAPHLAB CREATE

[Full iPython Notebook Source](https://github.com/arafatm/edu_coursera_machine_learning_1_foundations/blob/master/code/01.02.getting.started.with.sframes.ipynb)

Machine learning library [scikit-learn](http://scikit-learn.org/stable/)

Data manipulation tool [Pandas](http://pandas.pydata.org/)

Tools above require a learning curve. This course uses [GraphLab 
Create](https://turi.com) that includes 
[SFrame](https://github.com/turi-code/SFrame)


Load a tabular data set `sf = graphlab.SFrame('people-example.csv')`

view end of the table `sf.tail()`

visualizes any data structure in GraphLab Create `sf.show()`

Categorical view `sf['age'].show(view='Categorical')`

Some simple columnar operations
```
sf['age'].mean()
sf['age'].max()
```

Create new columns in our SFrame
`sf['Full Name'] = sf['First Name'] + ' ' + sf['Last Name']`

Use the apply function to do a advance transformation of our data
```
def transform_country(country):

  if country == 'USA':
    return 'United States'
  else:
  return country

transform_country('USA')

sf['Country'].apply(transform_country)
```

## WEEK 2 Regression: Predicting House Prices

### Introduction

### Linear regression modeling

This week you will build your first intelligent application that makes
predictions of house prices from data.

Create models that predict a **continuous value** (price) from **input 
features** (square footage, number of bedrooms and bathrooms,...).

#### Predicting house prices: A case study in regression

Want to list my house
- Compare houses in neighborhood
- Focus on similar houses: sq ft, bedrooms, etc

Plot a graph
- X = sq. ft.
- Y = Price

Terminology:
- x = **feature covariate** or **predictor**
- y = **observation** or **response**

**Note** no house on graph will have same sq ft as yours. Also, if you only 
include similar houses you're discarding the rest of the data on the graph

Fit a line through the data = `f(x) = w0 + w1 x`
- `w0` = intercept
- `w1` = slope

![Linear Regression 
Model](https://drive.google.com/uc?id=0BwjXv3TJiWYEOE1tWjFib2xVOWs)

`f(x)` is **parameterized** by `(w0, w1)`

**RSS** := Residaul sum of squares
- draw a line and sum the distance of plots from line

    RSS(w0,w1) = ($house1 - [w0 + w1(sq ft house1])^2
               + ($house2 - [w0 + w1(sq ft house2])^2
               + ($house3 - [w0 + w1(sq ft house3])^2
               + ...

![RSS](https://drive.google.com/uc?id=0BwjXv3TJiWYERUZfX29HWGZJMkU)

**best line** is the one that minimizes the cost over all possible w0,w1

`Ŵ = (ŵ0, ŵ1)` **W hat** 

`Fŵ(x) = ŵ0 + ŵ1 x`

Best guess of your house price `ŷ = ŵ0 + ŵ1 (sq ft your house)`

#### Adding higher order effects

But what if it's not a linear relationship. It could be quadratic.

`Fw(x) = w0 + w1 x + w2 x^2`

**note** we square x, but not w

We can apply even **higher order polynomials** to reduce RSS further

![Quadratic](https://drive.google.com/uc?id=0BwjXv3TJiWYEXzIySWUxX2t0dUU)

An example of an even higher order polynomial that may not be what you want :)

![Higher order polynomial](https://drive.google.com/uc?id=0BwjXv3TJiWYEYWViYzE0Tkhvdk0)

### Evaluating regression models

#### Evaluating overfitting via training/test split

Based on the last example, we can **overfit** to the point that it's not 
generalizable to new data

Want good predictions but can't observe future. We can **simulate prediction**
- **test set**: remove some houses
- **training set**: fit model on remaining houses
- predict test set

    Training error (w) = ($train1 - fw(sq.ft. train1))^2
                       + ($train2 - fw(sq.ft. train2))^2
                       + ($train3 - fw(sq.ft. train3))^2
                       + ...

![Training error](https://drive.google.com/uc?id=0BwjXv3TJiWYEbGZSdjgzR2xrTWs)


    Test error (ŵ) = ($test1 - fw(sq.ft. test1))^2
                   + ($test2 - fw(sq.ft. test2))^2
                   + ($test3 - fw(sq.ft. test3))^2
                   + ...

![Test error](https://drive.google.com/uc?id=0BwjXv3TJiWYETF9xREphRG1WanM)

#### Training/test curves

Training error `ŵ` decresases with increasing model order

Test error decreases *up to a point* but then starts increasing

![Training/Test curves](https://drive.google.com/uc?id=0BwjXv3TJiWYEa1ZBNTQ4MldTTjg)

#### Adding other features

What if we need to add additional variables e.g. *# bathrooms*

Each new variable is a new dimension. so adding bathroom is a 3d graph
- calculate **hyperplane** on the cube

![More features](https://drive.google.com/open?id=0BwjXv3TJiWYEU3VRS2QzdnN2TVk)


#### Other regression examples

- Salary after ML specialization `ŷ = ŵ0 + ŵ1 performance + ŵ capstone + ŵ forum`
- Stock prediction depends on recent prices, news event, related commodities
- tweet poplarity: # followers, # followers of followers, popularity of hashtag

### Summary of regression

![Regression Summary](https://drive.google.com/open?id=0BwjXv3TJiWYEUlVHT3NQX1pMVzA)

- Describe the input **features** and output **real-valued predictions** of a 
  **regression model**
- Calculate a goodness-of-fit metric (e.g., **RSS**) 
- Estimate model parameters by **minimizing RSS**  (algorithms to come...) 
- Exploit the estimated model to form **predictions** 
- Perform a **training/test split** of the data 
- Analyze performance of various regression models in terms of **test error** 
- Use test error to **avoid overfitting** when selecting amongst candidate 
  models 
- Describe a regression model using **multiple features** 

### Quiz: Regression

1. 1
2. 2
3. 3
4. 3
5. x 2,3 : 4
6. 4 
7. x 2,3 : 4
8. 3
9. 2

See [Explore the Quadratic 
Equation](https://www.mathsisfun.com/algebra/quadratic-equation-graph.html) to 
see the effect of the coefficients

### Predicting house prices: IPython Notebook

[Full iPython Notebook Source](https://github.com/arafatm/edu_coursera_machine_learning_1_foundations/blob/master/code/02.01.predicting.house.prices.ipynb)

#### Loading & exploring house sale data

`sales = graphlab.SFrame('home_data.gl/')`

View the data
`sales`

Generate a scatter plot
`sales.show(view="Scatter Plot", x="sqft_living", y="price")`
- can hover over individual points to explore further

#### Splitting the data into training and test sets

`train_data,test_data = sales.random_split(.8,seed=0)`
- Use `random_split` to split training and test data
- `0.8` => 80% training and 20% test
- set `seed` to 0 in this case. we should use a random seed or let GL pick it 
  for you

#### Learning a simple regression model to predict house prices from house size

`sqft_model = graphlab.linear_regression.create(train_data, target='price', 
features=['sqft_living'], validation_set=None)`
- `linear_regression.create`
- note the default algorithm used is **Newton's Method**

#### Evaluating error (RMSE) of the simple model

`print sqft_model.evaluate(test_data)`
- `max_error` is the outlier
- Also shows **RMSE**

#### Visualizing predictions of simple model with Matplotlib

```
import matplotlib.pyplot as plt
%matplotlib inline
```

```
plt.plot(test_data['sqft_living'], test_data['price'], '.',
         test_data['sqft_living'], sqft_model.predict(test_data), '-')
```
- blue `.` is the actual data scatter plot
- green `-` as a line is the predicted value based on the test_data

#### Inspecting the model coefficients learned

`sqft_model.get('coefficients')`
- (intercept) =  where the line crosses the y axis
- sqft_living ~= the average cost of a house per sq ft according to this 
  regression model

#### Exploring other features of the data

View other features of a house we might be interested in
```
my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
sales[my_features].show()

sales.show(view='BoxWhisker Plot', x='zipcode', y='price')
```
- `BoxWhisker Plot` to view the set split by feature (zipcode in this case)

#### Learning a model to predict house prices from more features

Based on the data we see other features such as zip code, and # bedrooms makes 
a difference in the estimated price of a home

```
my_features_model = graphlab.linear_regression.create(train_data,
                                                      target='price',
                                                      features=my_features,
                                                      validation_set=None)

```

`print my_features` to view what features are includedd

Compare the original model to the expanded features model
```
print sqft_model.evaluate(test_data)
print my_features_model.evaluate(test_data)
```
- Note that the **rmse** has lowered by adding additional features

#### Applying learned models to predict price of an average house

```
house1 = sales[sales['id']=='5309101200'] # find a particular house by id

house1 # to view the data for this house
```

`print house1['price']` to view the actual price based on the data

`print sqft_model.predict(house1)` to check the price our model predicted

`print my_features_model.predict(house1)` to predict price based on expanded 
features

:caution: the prediction model based on sqft was more accurate than the 
expanded feature model in **this case**

#### Applying learned models to predict price of two fancy houses

`house2 = sales[sales['id']=='1925069082']`
- this is an example of a house where due to an uncaptured feature, "on the 
  waterfront", was not predicted very well by our model
- expanded feature prediction is _closer_ than the original sqft model

Final example is Bill Gates' house. We don't have data on what the actual price 
is but the prediction gives us a price, that is probably on the low end.

### Quiz: Programming assignment

1. Selection and summary statistics: In the notebook we covered in the module, 
   we discovered which neighborhood (zip code) of Seattle had the highest 
   average house sale price. Now, take the sales data, select only the houses 
   with this zip code, and compute the average price.

Ans:
```python
houses = sales[sales['zipcode']=='98039']
houses['price'].mean()
```

2160606.5999999996

2. Filtering data: One of the key features we used in our model was the number 
   of square feet of living space (sqft_living) in the house. For this part, we 
   are going to use the idea of filtering (selecting) data.

   In particular, we are going to use logical filters to select rows of an 
   SFrame. You can find more info in the Logical Filter section of this 
   documentation. Using such filters, first select the houses that have 
   sqft_living higher than 2000 sqft but no larger than 4000 sqft. What 
   fraction of the all houses have sqft_living in this range? Save this result 
   to answer the quiz at the end.

Ans:
```python
100.0 * sales[(sales['sqft_living'] > 2000) & (sales['sqft_living'] <= 4000)].num_rows() / sales.num_rows()
```

42.187572294452416


3. Building a regression model with several more features: In the sample 
   notebook, we built two regression models to predict house prices, one using 
   just sqft_living and the other one using a few more features, we called this 
   set

 `my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']`

Now, going back to the original dataset, you will build a model using the
following features:

```
advanced_features = 
[
'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house       
'grade', # measure of quality of construction       
'waterfront', # waterfront property       
'view', # type of view        
'sqft_above', # square feet above ground        
'sqft_basement', # square feet in basement        
'yr_built', # the year built        
'yr_renovated', # the year renovated        
'lat', 'long', # the lat-long of the parcel       
'sqft_living15', # average sq.ft. of 15 nearest neighbors         
'sqft_lot15', # average lot size of 15 nearest neighbors 
]
```

Compute the RMSE (root mean squared error) on the test_data for the model using
just my_features, and for the one using advanced_features.

Ans:
```python
my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']

advanced_features = 
[
'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house       
'grade', # measure of quality of construction       
'waterfront', # waterfront property       
'view', # type of view        
'sqft_above', # square feet above ground        
'sqft_basement', # square feet in basement        
'yr_built', # the year built        
'yr_renovated', # the year renovated        
'lat', 'long', # the lat-long of the parcel       
'sqft_living15', # average sq.ft. of 15 nearest neighbors         
'sqft_lot15', # average lot size of 15 nearest neighbors 
]

train_data,test_data = sales.random_split(.8,seed=0)

f_model = graphlab.linear_regression.create(train_data, target='price', features=my_features, validation_set=None)
af_model = graphlab.linear_regression.create(train_data, target='price', features=advanced_features, validation_set=None)

print "RMSE diff = " + str(f_model.evaluate(test_data)['rmse'] - af_model.evaluate(test_data)['rmse'])
```

RMSE diff = 22711.3165108

## WEEK 3 Classification: Analyzing Sentiment

### CLASSIFICATION MODELING

#### Analyzing the sentiment of reviews

Rating with stars is too simple

Understand aspects of restaurant review
- build a restaurant review app
- categories for review e.g. experience, ramen, sushi
- Break all reviews into sentences in a **Sentence Sentiment Classifier**
- Average the predictions
- Display the most positive or negative reviews

#### Classification Aplications

    Input (sentence) -> Classifier -> Output (predicted rating)


Examples
- Webpage classification by category e.g. education, finance, technology
- Spam filtering: checks sender, text, ipaddress, etc
- Image classification
- Personalized medical diagnosis
- Reading your mind by FMRI


**simple threshold classifier**
- list of +ve words: great, awesome, etc
- list of -ve words: bad, terrible, etc
- `if # +ve words > # -ve words => ŷ = +ve else ŷ = -ve`
- e.g. "Great sushi, awesome food, but terrible service" => +2, -1

Problems with threshold classifier
- populating initial +ve and -ve word lists
- words have degrees of sentiment: e.g. "great" > "good" 
- single words are not enough: "good" vs "not good"

The first two can be address by learning a classifier

The 3rd issue can be address by more elaborate features


| Word                            | Weight |
| --------------------------------| -----: |
| great                           |  1.5   |
| awesome                         |  1.2   |
| bad                             | -1.0   |
| terrible                        | -2.1   |
| awful                           | -3.3   |
| restaurant, the, we, where, ... |  0.0   |

e.g. "Sushi was great, the food was awesome, but the service was terrible"
- `score = 1.2 (great) + 1.7 (awesome) - 2.1 (terrible) = 0.8`

####  Decision Boundaries

Can graph scores. e.g. awesome = 1.0 and awful = -1.5

    awful
     5|
     4|
     3|
     2| * (this ambiance is awesome but service is awful and food is awful)
     1|
     0|-------- awesome
      0 1 2 3 4

For linear classifiers
- when 2 weights are non-zero: **line**
- when 3 weights are non-zero: **plane**
- when 2 weights are non-zero: **hyperplane**

### Evaluating classification models

#### Training and Evaluating a Classifier

Training a classifier = Learning the weights
- split data into *training* and *test* sets
- training set is passed to *learned classifier* to learn weights of words
- test set is *evaluated* by **error** & **accuracy**

Test example "Sushi was great"
- Feed the sentence ^ to the learned classifier
- predict `ŷ` is +ve
- pass multiple test cases and compare *correct* vs *mistakes*

**Classification error & accuracy**
- `Error = (# mistakes) / (total # sentences)`
  - best possible value = 0.0
- `Accuracy = (# correct) / (total # sentences)`
  - best possible value = 1.0

`error = 1 - accuracy`

#### What's a good accuracy?

Purely random guessing on a binary classification = `0.5 accuracy`

For *k classes*, `accuracy = 1/k`

**we should at least beat random guessing**

This can be counterintuitive e.g.
- "90% email is spam"
- if we predict 100% email is spam we get 90% accuracy

Questions to ask:
- Is there class imbalance? 
- How does it compare to a simple,  baseline approach? 
  - Random guessing 
  - Majority class 
  - ... 
- Most importantly:  what accuracy does my application need? 
  - What is good enough for my users experience? 
  - What is the impact of the mistakes we make? 

#### False positives, false negatives, and confusion matrices

A **confusion matrix**

                   Predicted Label
                 |   +ve     |  -ve      |
                 |-----------|-----------|
     True    +ve | True +ve  | False -ve |
     Label   -ve | False +ve | True -ve  |

- True +ve & True -ve is good, we got it right
- **FN** (False -ve) & **FP** have different impacts

Example of domains

               | Spam filtering | Medical Diagnosis  |
               | -------------- | ------------------ |
     False -ve | Annyoing       | Disease untreated  |
     False +ve | Email lost     | Wasteful treatment |


Given 100 test examples, a possible confusion matrix for spam filtering

                Predicted Label
                 | +ve  | -ve  |
     True    +ve | (50) |  10  |
     Label   -ve |   5  | (35) |

Accuracy = 85/100 = 0.85 ie higher false +ve than false -ve

Multiclass classification example with 100 test examples

              Predicted Label
          |            | Healthy    | Cold | Flu |
          | ---------- | ---------- | ---- | --- |
    True  | Healthy 70 | (60)       | 8    |  2  |
    Label |    Cold 20 |   4        | (12) |  4  |
          |     Flu 10 |   2        |      | (8) |

Accuracy = (60 + 12 + 8)/100 = 80/100 = 0.8

#### Learning Curves: How much data do I need?

More data is good but *data quality* is more important

Theoretical techniques sometimes bound how much data is needed
- provide guidance but not as practical

In practice
- more complex models require more data
- empirical analysis can provide guidance

Learning curves
- Generally, the more data we have the fewer test errors we find

    Test error

          ^
          |*
          | *
          |   *
          |      *
          |            *
          |                        *
          |                                                 *
          |----------------------------------> Amount of training data

**Bias of model** even with infinite data, the curve never reaches 0. 

More complex models tend to have less bias
- sentiment classifier on single words does ok
- But some are just too hard e.g. "The sushi was *not good*"
- we can then score on pairs of words **bigram model**

Even bigram models have bias. The graph looks better (approaches 0 faster) but
never hits 0

#### Class Probabilities

Classifier provide a confidence level `P(y|x)`
- e.g. "the sushi & everything else were awesome" `P(y=+|x) = 0.99`
- "The sushi was good, the service was OK" `P(y=+|x) = 0.55`
  - less confident that this is a +ve review

### Summary of classification

#### Classification ML block diagram

![Classification ML Block Diagram](https://drive.google.com/uc?id=0BwjXv3TJiWYEOHY0MTRVUl80SUU)
- `y` = Training data
- `x` = word counts
- `ŷ` = predicted sentiment
- `ŵ` = weights for each word
- `y` = sentiment label
- compare `y` with `ŷ` to get a quality metric that is fed into ML Algorithm
- ML algorithm updates `ŵ`

#### Summary

- Identify a classification problem and some common applications 
- Describe decision boundaries and linear classifiers 
- Train a classifier  
- Measure its error 
  - Some rules of thumb for good accuracy 
- Interpret the types of error associated with classification 
- Describe the tradeoffs between model bias and data set size 
- Use class probability to express degree of confidence in prediction  

#### Quiz: Classification

1 The simple threshold classifier for sentiment analysis described in the video 
(check all that apply):
- Must have pre-defined positive and negative attributes
- Must either count attributes equally or pre-define weights on attributes
- Defines a possibly non-linear decision boundary

1

2 For a linear classifier classifying between positive and negative sentiment 
in a review x, Score(x) = 0 implies (check all that apply):
- The review is very clearly negative
- We are uncertain whether the review is positive or negative
- We need to retrain our classifier because an error has occurred

2

3 For which of the following datasets would a linear classifier perform 
perfectly?

x 1,2,3

4 True or false: High classification accuracy always indicates a good 
classifier.

false

5 True or false: For a classifier classifying between 5 classes, there always 
exists a classifier with accuracy greater than 0.18.

true

6 True or false: A false negative is always worse than a false positive.

false

7 Which of the following statements are true? (Check all that apply)
- Test error tends to decrease with more training data until a point, and then 
  does not change (i.e., curve flattens out)
- Test error always goes to 0 with an unboundedly large training dataset
- Test error is never a function of the amount of training data

1

### Analyzing sentiment: IPython Notebook

#### Open the iPython Notebook used in this lesson to follow along
#### Loading & exploring product review data
#### Creating the word count vector
#### Exploring the most popular product
#### Defining which reviews have positive or negative sentiment
#### Training a sentiment classifier
#### Evaluating a classifier & the ROC curve
#### Applying model to find most positive & negative reviews for a product
#### Exploring the most positive & negative aspects of a product

### Programming assignment
#### Analyzing product sentiment assignment
#### Quiz: Analyzing product sentiment
