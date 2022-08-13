from Network import *

import matplotlib.pyplot as plt


def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).

    Argument:
    x -- A numpy matrix of shape (n, m)

    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """

    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)

    # Divide x by its norm.
    x = x / x_norm

    return x


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = np.load("train_data.npy"), np.load("train_label.npy"), np.load(
        "test_data.npy"), np.load("test_label.npy")
    X_train = normalizeRows(X_train)
    X_test = normalizeRows(X_test)
    y_train = np.eye(10)[y_train.ravel().astype(int)]
    y_test = np.eye(10)[y_test.ravel().astype(int)]
    print(X_train.shape, X_test.shape)
    """
    This neural network  is allowed 2 hidden layers 
    The model of hidden layers is : FC1 -> ReLU -> FC2 ->ReLU 
    :param hidden_dims_1, hidden_dims_2: number of neural in each hidden layer 
    :param optimizer: SGD momentum hyper parameter {lr -> learning rate, momentum -> momentum unit}
    :param p: Dropout method: Neural retention rate
    :param regular_act: l2 regularization
    :param BN: 'train' stands for training network; 'test' stands for testing network, evaluate method default 'test'
    """

    model = DFN(hidden_dims_1=200,
                hidden_dims_2=10,
                optimizer="momentum(lr=0.01, momentum=0.90)",
                # regular_act="l2(lambd=0.001)",
                p=0.99,
                # BN='train'
                )
    train_acc = model.fit(X_train, y_train, n_epochs=300, batch_size=128, verbose=False)
    acc, confusion_matrix = model.evaluate(X_test, y_test)

    """
    Plot training accuracy figure
    """
    plt.figure()
    x = np.arange(0, len(train_acc), 1)
    plt.plot(x, train_acc)
    plt.title('Training Acc --- ' + str(len(train_acc)) + ' Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Acc Rate')
    plt.show()

    """
    Plot confusion matrix
    """
    plt.figure()
    plt.imshow(confusion_matrix, cmap=plt.cm.Blues)
    indices = range(len(confusion_matrix))
    plt.xticks(indices, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.yticks(indices, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.colorbar()
    plt.xlabel('Prediction Value')
    plt.ylabel('True Value')
    plt.title('Confusion Matrix')
    plt.show()

    """
    Output the test(validation) accuracy
    """
    print("Test accuracy:{}".format(acc))
