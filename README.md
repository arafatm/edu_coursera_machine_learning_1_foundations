# MACHINE LEARNING FOUNDATIONS

https://www.coursera.org/specializations/machine-learning

## WK 1 INTRODUCTION

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

Machine learning library [scikit-learn](http://scikit-learn.org/stable/)

Data manipulation tool [Pandas](http://pandas.pydata.org/)

Tools above require a learning curve. This course uses [GraphLab 
Create](https://turi.com) that includes 
[SFrame](https://github.com/turi-code/SFrame)

To get the code below to run

```
pip install virtualenvwrapper
```

```python
import graphlab

graphlab.product_key.set_product_key('your product key here')
graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)
graphlab.product_key.get_product_key()

sf = graphlab.SFrame('people-example.csv') # Load a tabular data set

sf # we can view first few lines of table

sf.tail() # view end of the table
          # .show() visualizes any data structure in GraphLab Create

```

graphlab.canvas.set_target('ipynb')

sf['age'].show(view='Categorical')

Inspect columns of dataset

sf['Country']

sf['age']

Some simple columnar operations

sf['age'].mean()

sf['age'].max()

Create new columns in our SFrame

sf

sf['Full Name'] = sf['First Name'] + ' ' + sf['Last Name']

sf

sf['age'] * sf['age']

Use the apply function to do a advance transformation of our data

sf['Country']

sf['Country'].show()

def transform_country(country):

    if country == 'USA':

              return 'United States'

                  else:

                            return country

                            transform_country('Brazil')

                            transform_country('Brasil')

                            transform_country('USA')

                            sf['Country'].apply(transform_country)

                            sf['Country'] = 
                            sf['Country'].apply(transform_country)

                            sf


