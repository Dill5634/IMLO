import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(f, df, x0, step_size_fn, max_iter):
    """
    Performs gradient descent on the given function f, with its gradient df.

    :param f: A function whose input is an x, a column vector, and returns a scalar.
    :param df: A function whose input is an x, a column vector, and returns a column vector representing the gradient of f at x.
    :param x0: An initial value of x, x0, which is a column vector.
    :param step_size_fn: A function that takes the iteration index and returns the step size for that iteration.
    :param max_iter: The number of iterations to perform.

    :return x: the value at the final step
    :return fs: the list of values of f found during all the iterations (including f(x0))
    :return xs: the list of values of x found during all the iterations (including x0)
    """
    x = x0
    fs = [f(x)]
    xs = [x]

    for i in range(max_iter):
        step_size = step_size_fn(i)
        x = x - step_size * df(x)
        fs.append(f(x))
        xs.append(x)

    return x, fs, xs



def transform_polynomial_basis_1d(x, order):
    """
    Transforms a single 1-dimensional data point x with a polynomial basis transformation.

    :param x: A numpy array with a single value (d=1).
    :param order: Can be 0, 1, 2 or 3.
    :return: The transformed data point x as a list.
    """
    if order == 0:
        return [1]
    if order == 1:
        return [1, x]
    if order == 2:
        return [1, x, x ** 2]
    if order == 3:
        return [1, x, x ** 2, x ** 3]

def data_linear_trivial():
    X = np.array([[-6], [-4], [-2], [0], [2], [4], [6]])
    Y = np.array([[-6], [-4], [-2], [0], [2], [4], [6]])
    return X, Y

def data_linear_simple():
    X = np.array([[-6], [-4], [-2], [0], [2], [4], [6]])
    Y = np.array([[-5], [-5], [-3], [1], [2], [5], [6]])
    return X, Y

def data_linear_offset():
    X = np.array([[-6], [-4], [-2], [0], [2], [4], [6]])
    Y = np.array([[-6], [-4], [-2], [0], [2], [4], [6]]) + 6
    return X, Y

def data_quadratic():
    X = np.array([[-6], [-4], [-2], [0], [2], [4], [6]])
    Y = np.array([[12], [3], [1], [0], [1], [3], [12]])
    return X, Y

def data_polynomial():
    X = np.array([[-6], [-4], [-2], [0], [2], [4], [6], [8]])
    Y = np.array([[12], [3], [1], [0], [1], [3], [12], [3]])
    return X, Y

def plot_line_2d(axes, theta, line_style, xmin=-10, xmax=10):
    """
    Takes a 2-dimensional theta and plots the line defined by (theta[1], theta[0]) into the given axes.
    """
    p1_y = theta[0] * xmin + theta[1]
    p2_y = theta[0] * xmax + theta[1]
    axes.plot([xmin, xmax], [p1_y.flatten(), p2_y.flatten()], line_style)

if __name__ == '__main__':
    """
    We'll implement gradient descent, and test it on a simple quadratic function. We'll go on implementing the
    closed-form OLS solution, as well as solving the OLS objective with gradient descent, with various provided data.
    We'll analyse the convergence of gradient descent with various step sizes, and plot the loss (convergence plot).
    The final exercise uses polynomial basis transformation to fit more complex data with linear hypotheses.
    """

    # Exercise 1: Gradient descent to find the minimum of a simple quadratic function
    # -----------
    def f1(x):
        # Our function is f(x) = (3x+4)^2
        return float((3 * x + 4) ** 2)

    def df1(x):
        # 1 (c): Todo: Implement the derivative here
        return float(18 * x + 24)

    # 1 (e): Plot the function:
    x_values = np.linspace(-10, 10, 400)
    y_values = [f1(x) for x in x_values]

    fig_simple, ax_simple = plt.subplots()  # create an empty plot and retrieve the 'ax' handle
    ax_simple.plot(x_values, y_values, label="f(x) = (3x + 4)^2")
    ax_simple.set_xlabel('x')
    ax_simple.set_ylabel('f(x)')
    ax_simple.legend()
    ax_simple.grid(True, which='both')

    # Set and plot the initial value:
    x0 = np.array([-8.0])  # Initial guess
    ax_simple.plot(x0, f1(x0), 'ro', label="Initial x0")

    # Run gradient descent to find the minimum of the function:
    last_x, fs, xs = gradient_descent(f1, df1, x0, step_size_fn=lambda i: 0.01, max_iter=50)

    # Plot the found 'x' value and f(x)
    ax_simple.plot(last_x, f1(last_x), 'go', label="Found minimum")
    ax_simple.legend()

    # Plot each step of gradient descent, to see how it converges/diverges
    for i, x in enumerate(xs):
        ax_simple.plot(x, f1(x), 'bo')
        ax_simple.text(x, f1(x), f"iter {i}", fontsize=8)

    plt.show()

    print(f"Initial x0: {x0}")
    print(f"Found minimum x: {last_x}")
    print(f"Function values: {fs}")
    print(f"x values: {xs}")

    # Exercise 2: Least Squares Regression
    # -----------

    # Get example data
    X, Y = data_linear_trivial()

    # Augment data with a column of ones
    X_augmented = np.array([transform_polynomial_basis_1d(x[0],order=2) for x in X])
  
    # Compute theta* using the analytical OLS solution
    theta_star = np.linalg.inv(X_augmented.T @ X_augmented) @ X_augmented.T @ Y

    # Plot the data and the resulting line
    fig, ax = plt.subplots()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 8)
    ax.grid(True, which='both')
    ax.axhline(color='black', linewidth=0.5)
    ax.axvline(color='black', linewidth=0.5)
    ax.set_title("Least squares regression")
    ax.scatter(X, Y, label="Data")

    # Plot the resulting hypothesis into the plot
    plot_line_2d(ax, theta_star.flatten(), line_style='r-', xmin=-1, xmax=5)
    ax.legend()

    plt.show()

    # Exercise 2.3 - Solution using gradient descent:
    # ------------

    # Implement the loss function
    def squared_loss(x, y, theta):
        return (y - np.dot(x, theta)) ** 2

    # Implement the OLS objective function (using the loss)
    def ols_objective(X, Y, theta):
        return np.mean([squared_loss(X[i], Y[i], theta) for i in range(len(Y))])

    # Implement the partial derivative of the squared loss w.r.t. theta
    def d_squared_loss_theta(x, y, theta):
        return -2 * (y - np.dot(x, theta)) * x

    # Implement the partial derivative of the OLS objective w.r.t. theta (using the partial derivative of the squared loss)
    def d_ols_objective_theta(X, Y, theta):
        return np.mean([d_squared_loss_theta(X[i], Y[i], theta) for i in range(len(Y))], axis=0)

    # Gradient of our OLS objective
    def ols_objective_grad(X, Y, theta):
        return d_ols_objective_theta(X, Y, theta)

    # Define the function that we want to minimise as the OLS objective over our dataset (X_augmented, Y)
    def f_ols(theta):
        return ols_objective(X_augmented, Y, theta)

    # Its gradient
    def df_ols(theta):
        return ols_objective_grad(X_augmented, Y, theta)

    # Set an initial value for theta_init
    theta_init = np.zeros(X_augmented.shape[1])

    # We define a step size function - let's return a constant step size, independent of the iteration i
    def step_size_fn(i):
        return 0.001  
    # Now we're ready to run gradient descent to minimise f_ols:

    last_x, fs, xs = gradient_descent(f_ols, df_ols, theta_init, step_size_fn=step_size_fn, max_iter=300)


    # Plot the resulting hypothesis into the plot
    plot_line_2d(ax, last_x, line_style='b-', xmin=-1, xmax=5)
    ax.legend()

    plt.show()

    # Plot the loss over the iterations
    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(fs, label="Loss over iterations")
    ax_loss.set_xlabel('Iteration')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    ax_loss.grid(True, which='both')
    ax_loss.axhline(color='black', linewidth=0.5)
    ax_loss.axvline(color='black', linewidth=0.5)
    ax_loss.set_title("Convergence of gradient descent")

    plt.show()

    print("Finished.")

    
        
    
        