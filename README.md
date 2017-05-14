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

Install anaconda, GraphLab, ipython notebook

`$ jupyter notebook` to start ipython notebook server

`jupyter nbconvert --execute <notebook>.ipynb` to create an html for your 
notebook
- notebooks can include code and markdown

`ipython <file>.py` to run the python script

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

#### Learning a simple regression model to predict house prices from house size

`sqft_model = graphlab.linear_regression.create(train_data, target='price', 
features=['sqft_living'], validation_set=None)`
- `linear_regression.create`

#### Evaluating error (RMSE) of the simple model

#### Visualizing predictions of simple model with Matplotlib

#### Inspecting the model coefficients learned

#### Exploring other features of the data

#### Learning a model to predict house prices from more features

#### Applying learned models to predict price of an average house

#### Applying learned models to predict price of two fancy houses

### Programming assignment

#### Predicting house prices assignment

#### Quiz: Predicting house prices3 questions

