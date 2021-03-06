# Newly Proposed Technique for Autism Spectrum Disorder based Machine Learning  

## Table of Contents
- Introduction.
- Technologies.
- Get Started.
  - Prerequisites.
  - Installation.
  - Approach.
  - Conclusion.

## Introduction
Machine Learning has become one of the important fields and fastest-growing areas of computer science and technology, it is a subfield of Artificial Intelligence (AI). Nowadays, machines are capable of learning like humans, and they are designed to perform tasks. Generally, Machine Learning used to understand the structure of data and fit that data into models that can be understood and utilized by people.

The field of ML can be implemented in healthcare to make better decision making about patients diagnoses and providing necessary needs for treatments. It aims to decrease the time needed to detect diseases, minimize physicians and health care professionals efforts in the detection of diseases and reducing progression through early discover. Predictive analysis by using ML algorithms help to predict the disease more correctly through processing a huge medical datasets.

Many diseases became easy to detect and recognize by using ML such as Heart Diseases Diagnosis and Autism Spectrum Disorder (ASD). ASD is known as a developmental disorder that affects behavior and communication, this autism can be diagnosed at any age. By using ML we can recognize these behaviors that are related to ASD. Therefore, it contributes to the provision of health services and the necessary needs early for infected toddlers, and early intervention can improve a child’s overall development, improve and enhance communication skills and interact with individuals.

## Technologies
Project is created with:
- Python.
- Flask.
- Web pages: HTML, CSS, Boostrap.
- Scikit-learn library

## Get Started
This is how to run the project by following the instructions for sitting up the project locally.
### Prerequisites
- Anaconda.
- Spyder, Python Development Environment.
- Local server such as, Xampp server.
### Installation
- Clone the project from repository,
  - Press clone, copy the link with HTTPS.
  - Use git bash to clone, in bash type,
  ```
  $ git clone <HTTP link>
  ```
  - Or download the zip file.
- Run the local srever.
- Open Anaconda.
- Select Spyder environment.
- Open the project folder.
  - Go to file.
  - Select open folder.
  - Choose the folder.
- Install all the required libraries.
  - Use the package manager [pip](https://pip.pypa.io/en/stable/) for installing.
  ```
  pip install <package_name>
  ```
- Run the project.
- Open terminal/command prompt from project directory and run the file ```app.py``` by executing the command ```python app.py```
- Enter the localhost API in the browser.

## Approach
### Step 1: The Dataset
Data gathering has a significant factor in solving the problem and this will allow capturing the record of previous cases of autism. The classifier can be more efficient as it will be based on the dataset from which it was built. This project used dataset published in Kaggle.
### Step 2: Data pre-processing
Data pre-processing is an important step in ML, the data requires pre-processing in order to yield useful insight into the data before feeding it to the algorithms. This process includes data cleaning, data reduction and data transformation in order to achieve better results from the model.
### Step 3: Training model
The model is trained using a training dataset by a supervised learning method, typically contains examples or samples used to fit the model which learns from these data .This involves ML algorithms that used to build a predictive model.
### Step 4: Testing model
The test model is used to evaluate the performance and estimates how well the model is trained and better fits. Typically, provides the final estimate of validating the model.
### Step 5: Re-training model
Re-training comes after the testing model if the model does not fit well with data and has a desire to re-train the dataset to meet the goals and tasks. Therefore, choosing the right models of prediction and classification data contribute in building an accurate and reliable model.
### Step 6: Classifier
They are algorithms tending to predict the class of data. The classes are called labels/targets or categories, it uses supervised learning to decide which data learned from the input data.

## Conclusion
This paper aims at developing a screening model using ML techniques for detecting and assessing the behaviors associated with ASD among toddlers age 12 - 30 months.  Early detection ASD can help to limit the challenges that are facing children such as communications ,social skills, learning disabilities , daily problems and many others. The goal is to apply supervised learning algorithms for a dataset from previous medical cases that can predict the autistic among toddlers.  The uses of the new screening methods can encourage the use of technology in clinical environments for autism and empower clinicians with tools that provide useful knowledge for better decision making.

