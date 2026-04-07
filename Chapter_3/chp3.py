from sklearn.datasets import fetch_openml

def datadownload():
    mnist = fetch_openml('mnist_784', version=1)
