# Project Title: Apartment Price Prediction

## Overview

This project aims to build a model that can predict the price of an apartment based on various features such as size, location, number of bedrooms, and more. The model will be trained on a dataset of past apartment listings, and will be able to make predictions on the price of new apartments based on the features provided.

## Data

The data for this project consists of a dataset of past apartment listings, including various features such as size, location, number of bedrooms and bathrooms, price and more. The data was scraped from https://www.4zida.rs/ during the month of July 2021. The data was cleaned and preprocessed to remove any missing or irrelevant information.We  also used one-hot encoding to convert categorical variables such as the location of the apartment (e.g., by neighborhood or zip code) into numerical form.

The data will be split into training and testing sets, with the model being trained on the training data and evaluated on the test data.

## Methodology

We used several models to find out which one performs best.We used regression,random forest,gradient boost and gradient boost models. Regression give us pretty solid results, but Gradient boost model performs best with score near 90%.We also tried to use feed foward neural network but this is typical regression problem and ann gived us bad results(model tends to overfitt).

## Evaluation

We will evaluate the performance of the model using mean absolute error (MAE) and root mean squared error (RMSE). A lower error rate indicates a better performing model.

## Future Work

There are several directions in which this project could be extended. Some possibilities include:

- Incorporating more data and features to the model, as we scraped near 2000 apartments.
- Testing the model on a larger, more diverse dataset.
- Implementing the model as a web app or API to allow for real-time predictions on new apartment listings.

 
