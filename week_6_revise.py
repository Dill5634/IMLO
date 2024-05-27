import numpy as np
import matplotlib.pyplot as plt

# Initialize lists to store intermediate results
linear_classify_results = []
loss_results = []
E_n_results = []
rlc_results = []

def linear_classify(x, theta, theta_0):
    """
    Classifies the input x using the linear classifier defined by theta and theta_0.
    """
    result = int(np.sign(np.dot(theta, x) + theta_0))
    linear_classify_results.append((x, theta, theta_0, result))
    return result

def Loss(prediction, actual):
    """
    Computes the loss based on the prediction and actual values.
    """
    result = 0 if prediction == actual else 1
    loss_results.append((prediction, actual, result))
    return result

def E_n(h, data, labels, L, theta, theta_0):
    """
    Computes the training error E_n.
    """
    loss = 0
    for i in range(L):
        x = data[:, i]
        y = labels[0, i]
        prediction = h(x, theta, theta_0)
        current_loss = Loss(prediction, y)
        loss += current_loss
    training_error = loss / L
    E_n_results.append((theta, theta_0, training_error))
    return training_error

def random_linear_classifier(data, labels, params={}, hook=None):
    """
    Implements the random linear classifier algorithm.
    """
    k = params.get('k', 100)  # if k is not in params, default to 100
    (d, n) = data.shape

    best_theta = None
    best_theta_0 = None
    best_error = float('inf')

    for _ in range(k):
        theta = np.random.randn(d)
        theta_0 = np.random.randn()
        error = E_n(linear_classify, data, labels, n, theta, theta_0)
        
        if error < best_error:
            best_error = error
            best_theta = theta
            best_theta_0 = theta_0
        
        # Call the hook function, if provided
        if hook is not None:
            hook((theta, theta_0))

    rlc_results.append((best_theta, best_theta_0, best_error))
    return best_theta, best_theta_0

def perceptron_with_offset(data, labels, params={}, hook=None):
    """
    The Perceptron learning algorithm with offset.

    :param data: A d x n matrix where d is the number of data dimensions and n the number of examples.
    :param labels: A 1 x n matrix with the label (actual value) for each data point.
    :param params: A dict, containing a key T, which is a positive integer number of steps to run
    :param hook: An optional hook function that is called in each iteration of the algorithm.
    :return: Returns the learned theta and theta_0
    """
    T = params.get('T', 100)  # if T is not in params, default to 100
    (d, n) = data.shape

    # Initialize weights and bias
    theta = np.zeros(d)
    theta_0 = 0

    # Training loop
    for t in range(T):
        for i in range(n):
            x = data[:, i]
            y = labels[0, i]
            if y * (np.dot(theta, x) + theta_0) <= 0:
                theta = theta + y * x
                theta_0 = theta_0 + y

        if hook is not None:
            hook((theta, theta_0))

    return theta, theta_0

def perceptron(data, labels, params={}, hook=None):
    """
    The Perceptron learning algorithm without offset (through-origin).

    :param data: A d x n matrix where d is the number of data dimensions and n the number of examples.
    :param labels: A 1 x n matrix with the label (actual value) for each data point.
    :param params: A dict, containing a key T, which is a positive integer number of steps to run
    :param hook: An optional hook function that is called in each iteration of the algorithm.
    :return: Returns the learned theta
    """
    T = params.get('T', 100)  # if T is not in params, default to 100
    (d, n) = data.shape

    # Initialize weights
    theta = np.zeros(d)

    # Training loop
    for t in range(T):
        for i in range(n):
            x = data[:, i]
            y = labels[0, i]
            if y * np.dot(theta, x) <= 0:
                theta = theta + y * x

        if hook is not None:
            hook((theta, 0))  # hook expects a tuple

    return theta

def augment_data_with_ones(data):
    """
    Augments the data by adding a column of ones.

    :param data: A d x n matrix.
    :return: A (d+1) x n matrix with a row of ones added.
    """
    (d, n) = data.shape
    augmented_data = np.vstack([data, np.ones((1, n))])
    return augmented_data
def plot_separator(plot_axes, theta, theta_0=0):
    """
    Plots the decision boundary defined by theta and theta_0.
    """
    if len(theta.shape) > 1 and theta.shape[0] > 2:
        theta = theta[:-1]  # Remove the augmented dimension
    
    x_vals = np.array(plot_axes.get_xlim())
    y_vals = - (theta_0 + theta[0] * x_vals) / theta[1]
    plot_axes.plot(x_vals, y_vals, '--')



def transform_polynomial_basis(x, order):
    """
    Transforms the input vector x to its polynomial basis of the given order.

    :param x: A 2D input data vector.
    :param order: The order of the polynomial basis (0 to 4).
    :return: The transformed x with the polynomial basis applied.
    """
    if order < 0 or order > 4:
        raise ValueError("Order must be between 0 and 4")
    
    if order == 0:
        return np.ones((1, x.shape[1]))
    
    transformed = x.copy()
    if order >= 1:
        transformed = np.vstack((transformed, x**2))
    if order >= 2:
        transformed = np.vstack((transformed, (x[0] * x[1]).reshape(1, -1)))
    if order >= 3:
        transformed = np.vstack((transformed, x**3))
    if order == 4:
        transformed = np.vstack((transformed, (x[0]**2 * x[1]).reshape(1, -1)))
        transformed = np.vstack((transformed, (x[0] * x[1]**2).reshape(1, -1)))

    return transformed






if __name__ == '__main__':
    """
    We'll define data X with its labels y, plot the data, and then run either the random_linear_classifier or the
    perceptron learning algorithm, to find a hypothesis h from the class of linear classifiers.
    We then plot the best hypothesis, as well as compute the training error. 
    """

    # Let's create some training data and labels:
    #X = np.array([[2, 3, 9, 12],
    #              [5, 2, 6, 5]])
    #y = np.array([[1, -1, 1, -1]])

    # To test your algorithm on a larger dataset, uncomment the following code. It generates uniformly distributed
    # random data in 2D, along with their labels.
    X = np.random.uniform(low=-5, high=5, size=(2, 20))  
    y = np.sign(np.dot(np.transpose([[3], [4]]), X) + 6) 
    
    X_augmented = augment_data_with_ones(X)

    

    # Plot positive data green, negative data red:
    colors = np.choose(y > 0, np.transpose(np.array(['r', 'g']))).flatten()
    plt.ion()  # enable matplotlib interactive mode
    fig, ax = plt.subplots()  # create an empty plot and retrieve the 'ax' handle
    ax.scatter(X[0, :], X[1, :], c=colors, marker='o')
    # Set up a pretty 2D plot:
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.grid(True, which='both')
    ax.axhline(color='black', linewidth=0.5)
    ax.axvline(color='black', linewidth=0.5)
    ax.set_title("Linear classification")

    # We'll define a hook function that we'll use to plot the separator at each step of the learning algorithm:
    def hook(params):
        (th, th0) = params
        plot_separator(ax, th, th0)
        plt.pause(0.01)  # Add a short pause to update the plot interactively

    # Run the RLC or Perceptron: (uncomment the following lines to call the learning algorithms)
    #theta, theta_0 = random_linear_classifier(X, y, {"k": 100}, hook=hook)
    theta, theta_0 = perceptron(X, y, {"T": 100}, hook=hook)
    theta, theta_0 = perceptron_with_offset(X, y, {"T": 100}, hook=None)
    theta_augmented  = perceptron(X_augmented,y,{"T":100},hook=None)
    # Plot the returned separator:
    plot_separator(ax, theta_augmented, 0)
    
    
    
    
    # Clear previous plots and plot the final separator:
    ax.clear()
    ax.scatter(X[0, :], X[1, :], c=colors, marker='o')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.grid(True, which='both')
    ax.axhline(color='black', linewidth=0.5)
    ax.axvline(color='black', linewidth=0.5)
    ax.set_title("Linear classification")
    plot_separator(ax, theta, theta_0)
    print(f"\nFinal Best Theta: {theta}, Final Best Theta_0: {theta_0}")
    

    # Run the RLC, plot E_n over various k:
    k_values = [1, 10, 50, 100, 200, 500, 1000]
    avg_errors = []

    for k in k_values:
        errors = []
        for _ in range(10):  # Run the algorithm 10 times for each k
            theta, theta_0 = random_linear_classifier(X, y, {"k": k})
            error = E_n(linear_classify, X, y, len(y[0]), theta, theta_0)
            errors.append(error)
        avg_errors.append(np.mean(errors))

    # Plotting k vs. average E_n
    plt.figure()
    plt.plot(k_values, avg_errors, marker='o')
    plt.xlabel('k')
    plt.ylabel('Average Training Error E_n')
    plt.title('Effect of Hyperparameter k on Training Error')
    plt.grid(True)
    plt.show()

    print("Finished.")

# Ensure the plots are displayed
plt.show(block=True)


