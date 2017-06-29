# Garden Image Browser

This is a proof of concept to categorize plants and image search similar looking plants using image recognition.

## Screen Shot

![front page of example app](https://raw.githubusercontent.com/panchishin/garden/master/screenshots/example1.png)

## Execution

### Download Images

First download the images `python downloadimages.py`.

### Train the model

*Optional* : There is already a saved model in `meta-data`.

You can retrain the model using `python traincnn.py`.  Warning, if you don't have a Nvidia GPU with 4GB ram on the card this could take 40 hours.  With a GPU it will take about 2 hours.

### Start the service

In linux use `./start.sh` to start the rest service.  Navigate to `http://localhost:9090/view/index.html` once the service has begun.

## Tech background

The underlying technology is TensorFlow using a CNN (convolutional neural net).

As a proof of concept, this application uses a small database
of 12,000 images of plants spread across 35 categories.  Compared to MNIST, a benchmark
machine learning data set, this data set is very small and has a lot of categories.



