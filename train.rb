require_relative 'lib/deepmine/tensor'
require_relative 'lib/deepmine/nn/module'
require_relative 'lib/deepmine/nn/linear'
require_relative 'lib/deepmine/nn/relu'
require_relative 'lib/deepmine/nn/sigmoid'
require_relative 'lib/deepmine/nn/tanh'
require_relative 'lib/deepmine/optim/optimizer'
require_relative 'lib/deepmine/optim/sgd'
require_relative 'lib/deepmine/loss/mse'

include Deepmine
# Generating Data
def generate_data(n_samples = 100)
  x = Array.new(n_samples) { [rand] }
  y = x.map { |xi| [3 * xi.first + 2 + rand(-0.1..0.1)] }
  [x, y]
end

x_data, y_data = generate_data
x_tensor = Tensor.new(x_data)
y_tensor = Tensor.new(y_data)

# Initializing the model
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
optimizer = Optim::SGD.new(model.parameters, lr = 0.01)
n_epochs = 1000

# Training Process
n_epochs.times do |epoch|
  predictions = model.call(x_tensor)
  loss = Loss::MSE.compute(predictions, y_tensor)

  loss_grad = Loss::MSE.backward(predictions, y_tensor)
  predictions.backward!(loss_grad)

  optimizer.step
  optimizer.zero_grad

  puts "Epoch #{epoch + 1}, Loss: #{loss.data.first}" if (epoch + 1) % 100 == 0
end
