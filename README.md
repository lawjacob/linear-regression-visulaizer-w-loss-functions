
# Gradient Descent Visualization Tool

This tool provides an interactive visualization of gradient descent for linear regression with customizable loss functions. It shows both the regression line fitting process and the loss function landscape in real-time.

# Quick Start for Linear Regression Beginners

The simplest way to use this tool is to experiment with the basic parameters of linear regression using the default L2 loss (Mean Squared Error):

```python
from gradient_descent_viz import GradientDescentAnimation

# Basic usage with different starting points
animation = GradientDescentAnimation(
    x, y,
    l2_loss, l2_gradient,
    theta0=2.0,    # Starting intercept
    theta1=2.0,    # Starting slope
    learning_rate=0.01
)
animation.animate()

## Basic Usage

from gradient_descent_viz import GradientDescentAnimation

# Create animation with default L2 loss
animation = GradientDescentAnimation(x, y, l2_loss, l2_gradient)
animation.animate()

# Create animation with L1 loss and custom learning rate
animation = GradientDescentAnimation(x, y, l1_loss, l1_gradient, learning_rate=0.005)
animation.animate()
```

## Available Parameters for Further Manipulation of Loss Function

The `GradientDescentAnimation` class accepts the following parameters:

- `x`: Input features (numpy array)
- `y`: Target values (numpy array) 
- `loss_fn`: Loss function to use
- `gradient_fn`: Gradient function corresponding to the loss function
- `theta0`: Initial intercept value (default: 2.0)
- `theta1`: Initial slope value (default: 2.0)
- `learning_rate`: Learning rate for gradient descent (default: 0.01)

The `animate()` method accepts:

- `frames`: Number of animation frames (default: 200)
- `interval`: Interval between frames in milliseconds (default: 20)

## Built-in Loss Functions

### L2 Loss (Mean Squared Error)
```python
# Squared error loss
animation = GradientDescentAnimation(x, y, l2_loss, l2_gradient)
```

### L1 Loss (Mean Absolute Error)
```python
# Absolute error loss
animation = GradientDescentAnimation(x, y, l1_loss, l1_gradient)
```

### Huber Loss
```python
# Huber loss with custom delta
delta = 1.0
animation = GradientDescentAnimation(x, y, 
    lambda e: huber_loss(e, delta), 
    lambda e: huber_gradient(e, delta))
```

## Defining Custom Loss Functions

To create a custom loss function, you need to define two functions:
1. The loss function itself
2. Its gradient(derivative) with respect to the error

Example:

```python
# Custom loss function
def custom_loss(error):
    return np.log(1 + error**2)

def custom_gradient(error):
    return 2 * error / (1 + error**2)

# Use custom loss function
animation = GradientDescentAnimation(x, y, custom_loss, custom_gradient)
```

Requirements for custom functions:
- Must accept a numpy array of errors as input
- Must return a numpy array of the same shape
- Loss function should return the point-wise loss values
- Gradient function should return the derivative of the loss with respect to the error

## Visualization Details

The tool creates two plots:

1. Left plot: Shows the data points and the current regression line
   - Gray dots: Training data points
   - Blue line: Current prediction line

2. Right plot: Shows the loss landscape
   - Gray dots: Current error points
   - Red line: Loss function curve
   - Text: Current total loss value

## Example Implementations

### Pseudo-Huber Loss
```python
def pseudo_huber_loss(error, delta=1.0):
    return delta**2 * (np.sqrt(1 + (error/delta)**2) - 1)

def pseudo_huber_gradient(error, delta=1.0):
    return error / np.sqrt(1 + (error/delta)**2)

animation = GradientDescentAnimation(x, y,
    lambda e: pseudo_huber_loss(e, delta=1.0),
    lambda e: pseudo_huber_gradient(e, delta=1.0),
    learning_rate=0.01)
```

### Log-Cosh Loss
```python
def log_cosh_loss(error):
    return np.log(np.cosh(error))

def log_cosh_gradient(error):
    return np.tanh(error)

animation = GradientDescentAnimation(x, y, 
    log_cosh_loss, 
    log_cosh_gradient,
    learning_rate=0.01)
```

## Tips for Choosing Parameters

- For L1 loss, use a smaller learning rate (around 0.005) due to the constant gradient
- For custom loss functions, start with a small learning rate (0.01 or less) and adjust as needed
- If the animation is unstable, reduce the learning rate
- If the convergence is too slow, increase the learning rate or number of frames
