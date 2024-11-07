import matplotlib.pyplot as plt

# Parameters
x = 2  # Starting point
lr = 0.01  # Learning rate
precision = 0.000001  # Precision for stopping
previous_step_size = 1  # Initial step size
max_iter = 10000  # Maximum iterations
iters = 0  # Iteration counter
gd = []  # Store values for plotting

# Gradient function for y = (x + 3)^2
gf = lambda x: 2 * (x + 3)  # Derivative of (x + 3)^2

# Gradient Descent loop
while precision < previous_step_size and iters < max_iter:
    prev = x
    x = x - lr * gf(prev)  # Update x
    previous_step_size = abs(x - prev)  # Calculate step size
    iters += 1
    gd.append(x)  # Store x value for each iteration
    print("Iteration:", iters, "Value:", x)

print("Local Minima:", x)

# Plotting the descent path
plt.plot(gd)
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.title('Gradient Descent Path to Local Minima')
plt.show()
