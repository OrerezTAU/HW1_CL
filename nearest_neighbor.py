from traceback import format_list

from scipy.spatial.distance import cdist
from sklearn.datasets import fetch_openml
import numpy.random
from scipy.spatial import distance

# Fetch MNIST dataset
mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target']

# Define training and test set of images

idx = numpy.random.RandomState(0).choice(70000,11000)

train = data[idx[:10000],:].astype('int')
train_labels = labels[idx[:10000]]

test = data[idx[10000:],:].astype('int')
test_labels = labels[idx[10000:]]

num_label_categories = 5


def knn_algo (train_images, labels_vec, query_img, k):
    """
    An implementation of k-nearest neighbors algorithm.
    :param train_images: a set of train images
    :param labels_vec: a vector of labels, corresponding to the image. Assumes labels are integers.
    :param query_img: a query image
    :param k: number of nearest neighbors
    :return: The most likely label for the query image, given the training images,
     using k-nearest neighbors with Euclidean distance
    """
    distances = cdist([query_img], train_images, metric='euclidean') # Calc Euclidean distances between query image and train images
    min_k_indices = numpy.argpartition(distances, k)[0][:k] # Get array of the indices of the k minimal distances
    closest_label = find_closest_label(min_k_indices, labels_vec,num_label_categories)
    return closest_label

def find_closest_label(indices, labels_arr,num_labels):
    """
    A helper function to find the closest label
    :param indices: A numpy array of indices for the labels array
    :param labels_arr: A numpy array that contains num_labels different labels
    :param num_labels: The number of different labels that may appear in the labels array
    :return: The label that appears the most times out of the labels whose indices are in the indices array
    If there is more than one label that appears most in the indices array, returns the first one.
    """
    count_arr = numpy.zeros(num_labels)
    for index in indices:
        label = int(labels_arr[index])
        count_arr[label] += 1
    most_common_label = numpy.argmax(count_arr)
    return most_common_label

def test_knn():
    """
    Test the k-nearest neighbors algorithm
    """
    # Initialize counters
    correct_predictions = 0

    # Define k (e.g., k = 3)
    k = 1

    # Loop through each test image and predict its label
    for i in range(len(test)):
        query_image = test[i]
        true_label = int(test_labels[i])

        # Call the k-NN function
        predicted_label = knn_algo(train, train_labels, query_image, k)

        # Check if the prediction matches the true label
        if predicted_label == true_label:
            correct_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / len(test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

test_knn()