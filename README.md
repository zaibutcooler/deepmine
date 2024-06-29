# Deepmine

Deepmine is a PyTorch-like neural network library implemented in Ruby. I made this project to help me in understanding the inner workings of neural networks, gradients, and the training process.

## Features

- Tensors for handling multi-dimensional arrays
- Neural network modules including Linear, ReLU, Sigmoid, and Tanh
- Optimizers such as SGD (Stochastic Gradient Descent)
- Loss functions including Mean Squared Error (MSE)
- Backpropagation for training models

## Training Process

### Step 1: Generate Data

Start by generating sample data for training:

```ruby
def generate_data(n_samples = 100)
  x = Array.new(n_samples) { [rand] }
  y = x.map { |xi| [3 * xi.first + 2 + rand(-0.1..0.1)] }
  [x, y]
end

x_data, y_data = generate_data
x_tensor = Tensor.new(x_data)
y_tensor = Tensor.new(y_data)
```

### Step 2: Define the Model

Create a simple neural network model:

```ruby
class DummyModel < NN::Module
  def initialize
    super()
    @linear1 = NN::Linear.new(1, 10)
    @relu = NN::ReLU.new
    @linear2 = NN::Linear.new(10, 1)
    @parameters = @linear1.parameters + @linear2.parameters
  end

  def forward(input)
    x = @linear1.call(input)
    x = @relu.call(x)
    @linear2.call(x)
  end
end

model = DummyModel.new
```

### Step 3: Initialize the Optimizer

Set up the optimizer for training:

```ruby
optimizer = Optim::SGD.new(model.parameters, lr = 0.01)
```

### Step 4: Training Loop

Train the model using a loop:

```ruby
n_epochs = 1000

n_epochs.times do |epoch|
  # Forward pass
  predictions = model.call(x_tensor)
  
  # Compute loss
  loss = Loss::MSE.compute(predictions, y_tensor)

  # Backward pass
  loss_grad = Loss::MSE.backward(predictions, y_tensor)
  predictions.backward!(loss_grad)

  # Update parameters
  optimizer.step
  optimizer.zero_grad

  # Print progress
  puts "Epoch #{epoch + 1}, Loss: #{loss.data.first}" if (epoch + 1) % 100 == 0
end
```

### Summary

This process involves generating data, defining a model, initializing an optimizer, and running a training loop where the model is trained using backpropagation and gradient descent. This step-by-step guide helps you understand the training process of a neural network using Deepmine.
