
from model import Model
from mnisdata import mnist

if __name__ == "__main__":
   
    print("Creating dataset")

    X_train, y_train,X_test,y_test = mnist(ntrain=6000, ntest=1000, digit_range=[0, 10])
    model = Model()
    model.fit(
        X_train, y_train,
        X_test, y_test,
        early=4,
        epoches=3
    )



#https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py
