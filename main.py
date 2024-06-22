from knn_classifier.knn import KNeighborsClassifier
from knn_classifier.data_loader import load_images_from_folder, preprocess_data, split_data
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np

def main():
    # Load images and labels
    data, labels = load_images_from_folder("./Face_Dataset/train")

    # Preprocess data
    data = preprocess_data(data)

    # Split data into train and test sets
    trainX, testX, trainY, testY = split_data(data, labels)

    # Evaluate KNN for different values of k and plot accuracy
    accuracies = []
    ks = range(1, 30)
    for k in ks:
        knn = KNeighborsClassifier(k=k)
        knn.fit(trainX, trainY)
        accuracy = knn.evaluate(testX, testY)
        accuracies.append(accuracy)

    # Visualize accuracy vs. k
    plt.figure()
    plt.plot(ks, accuracies)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Performance of k-NN")
    plt.grid(True)
    plt.show()

    # Train final model and evaluate
    final_model = KNeighborsClassifier(k=3)
    final_model.fit(trainX, trainY)

    print("Final model accuracy =", final_model.evaluate(testX, testY))
    print(classification_report(testY, final_model.predict(testX), target_names=np.unique(labels)))

if __name__ == "__main__":
    main()
