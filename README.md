# CSCI2470_final_project
final project
## Inspiration
As the digital currency market develops, we will inevitably be involved in the torrent. We hope to use our knowledge to explore the changing trends behind digital currencies.

## What it does
We hope to use LSTM or RNN models to predict the future price trend of digital currencies.

## How we built it
**Data is collected and preprocessed**
Through the API documentation provided by the application, use URL and HTTP requests to specify the type of data we want to obtain (including Timestamp, Open, High, Low, Close) and the date range (we took the data of one year in "days" ). Then use the `requests` library to send a request, get the JSON format data returned by the API and save it to a CSV file.
Through the function "preprocess_data", we process the time series data into a format suitable for training an LSTM (Long Short-Term Memory Network) model:
We extract features and targets and normalize the feature data by instantiating a `MinMaxScaler` object and fit_transform. Then create a time step for the LSTM model to ensure that each feature data "X[i]" corresponds to a target variable "y[i]". Finally, use the `train_test_split` function to divide the training set and the test set, and the proportion of the test set is initially set to 80%. And set random seeds before partitioning and scramble the data to ensure the reproducibility of each partitioning result.
It returns the features and targets of the divided training and test sets, which will be used to train the LSTM model and evaluate model performance.

**Model implementation is complete or nearly complete**
We used the “closing price” in the training set, brought the data into the LSTM model, and obtained a prediction result. 
![alt text](prediction_result1.jpg)

## Challenges we ran into
During our initial training, the loss was very large, even hundreds of millions.
We try to regularize the target, and the loss is significantly reduced. And after visualizing the loss results, we found that this is in line with the ideal situation.
[Imgur](https://i.imgur.com/RPLcn5h.jpg)

And when we output the test results, we found that the results were in an overfit state, so we added a dropout layer to solve the problem. For the problem of poor visualization of the fitting results, we increased the number of training times and found that the effect would improve.
[Imgur](https://i.imgur.com/LrvSviY.jpg)
[Imgur](https://i.imgur.com/GTZDOYS.jpg)
[Imgur](https://i.imgur.com/RPLcn5h.jpg)
## Accomplishments that we're proud of
This is our prediction of future "closing price" based on historical " Open, High, Low, Closing price" data:
[Imgur](https://i.imgur.com/RPLcn5h.jpg)
## What we learned
By changing the activation function, such as softmax and relu, there will be different effects. Adding a dropout layer can solve the problem of image moving downwards

## What's next for project
In the future, we will expand the number of feature types and add dimensions, and even add quantitative “policy” data to our LSTM models. At the same time, we will also try to change the model architecture to further improve the comprehensiveness and accuracy of model predictions.


