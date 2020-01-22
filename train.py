import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score,classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pickle

digits = datasets.load_digits()

n_samples = len(digits.images)

data = digits.images.reshape((n_samples, -1))/16

classifier = MLPClassifier(hidden_layer_sizes=(100, ), 
                    max_iter=1000, alpha=0.03,
                    solver='sgd', verbose=10, 
                    tol=0.00001, random_state=1,
                    learning_rate_init=.1)

X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)


classifier.fit(X_train, y_train)

pickle.dump(classifier,open("model.pkl","wb"))


loaded_model = pickle.load(open("model.pkl","rb"))
result = loaded_model.predict([X_test[27]])

print(result[0])

predicted = classifier.predict(X_test)
print(classification_report(y_test,predicted))
print(accuracy_score(y_test, predicted))

