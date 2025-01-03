import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Define different loss functions and their gradients
def l2_loss(error):
    return 0.5 * error**2

def l2_gradient(error):
    return error

def l1_loss(error):
    return np.abs(error)

def l1_gradient(error):
    return np.sign(error)

def huber_loss(error, delta=1.0):
    mask = np.abs(error) <= delta
    squared_loss = 0.5 * error**2
    linear_loss = delta * np.abs(error) - 0.5 * delta**2
    return np.where(mask, squared_loss, linear_loss)

def huber_gradient(error, delta=1.0):
    mask = np.abs(error) <= delta
    return np.where(mask, error, delta * np.sign(error))

# Generate synthetic data with quadratic relationship
x = np.linspace(0, 10, 20)
y = 10 - 0.5 * x + 0.1 * x**2 + np.random.normal(0, 1, 20)

# Normalize data
x = (x - x.mean()) / x.std()
y = (y - y.mean()) / y.std()

# Create a class to handle the animation with customizable loss function
class GradientDescentAnimation:
    def __init__(self, x, y, loss_fn, gradient_fn, theta0=2.0, theta1=2.0, learning_rate=0.01):
        self.x = x
        self.y = y
        self.theta0 = theta0
        self.theta1 = theta1
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.gradient_fn = gradient_fn
        
        # Setup the figure
        plt.style.use('seaborn-v0_8')
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 5))
        self.fig.suptitle('Linear Regression Gradient Descent Visualization')
        
        # Initialize plots
        self.scatter1 = self.ax1.scatter(x, y, c='gray')
        self.line1, = self.ax1.plot([], [], 'b-', label='Prediction')
        self.ax1.set_xlabel('x')
        self.ax1.set_ylabel('y')
        self.ax1.legend()
        
        self.scatter2 = self.ax2.scatter([], [], c='gray')
        self.error_line, = self.ax2.plot([], [], 'r-')
        self.ax2.set_xlabel('ŷ - y')
        self.ax2.set_ylabel('Error')
        self.total_loss_text = self.ax2.text(0.02, 0.95, '', transform=self.ax2.transAxes)
        
        # Set axis limits
        self.ax1.set_xlim(min(x)-0.5, max(x)+0.5)
        self.ax1.set_ylim(min(y)-0.5, max(y)+0.5)
        self.ax2.set_xlim(-3, 3)
        self.ax2.set_ylim(-0.5, 5)
    
    def calculate_predictions(self):
        return self.theta0 + self.theta1 * self.x
    
    def calculate_error(self, y_pred):
        return y_pred - self.y
    
    def calculate_total_loss(self, error):
        return np.sum(self.loss_fn(error))
    
    def update(self, frame):
        # Calculate predictions and error
        y_pred = self.calculate_predictions()
        error = self.calculate_error(y_pred)
        total_loss = self.calculate_total_loss(error)
        
        # Update parameters using gradient descent
        gradients = self.gradient_fn(error)
        d_theta0 = np.mean(gradients)
        d_theta1 = np.mean(gradients * self.x)
        
        self.theta0 -= self.learning_rate * d_theta0
        self.theta1 -= self.learning_rate * d_theta1
        
        # Update first plot
        self.line1.set_data(self.x, y_pred)
        
        # Update second plot
        error_points = np.linspace(min(error), max(error), 100)
        error_curve = self.loss_fn(error_points)
        self.scatter2.set_offsets(np.c_[error, self.loss_fn(error)])
        self.error_line.set_data(error_points, error_curve)
        
        # Update total loss text
        self.total_loss_text.set_text(f'Total Loss = {total_loss:.2f}')
        
        return self.line1, self.scatter2, self.error_line, self.total_loss_text
    
    def animate(self, frames=200, interval=20):
        self.anim = FuncAnimation(self.fig, self.update, frames=frames, 
                                interval=interval, blit=True)
        plt.tight_layout()
        plt.show()

# Example usage with different loss functions:

# Initialize parameters
theta0 = 2.0  # intercept
theta1 = 0.0  # slope
learning_rate = 0.01

# L1 Loss
animation_l1 = GradientDescentAnimation(x, y, l1_loss, l1_gradient, learning_rate=0.005)
animation_l1.animate()

# L2 Loss
# animation_l2 = GradientDescentAnimation(x, y, l2_loss, l2_gradient)
# animation_l2.animate()

# Huber Loss
# animation_huber = GradientDescentAnimation(x, y, 
#                                          lambda e: huber_loss(e, delta=1.0),
#                                          lambda e: huber_gradient(e, delta=1.0),
#                                          learning_rate=0.01)
# animation_huber.animate()
