module Deepmine
    module Optim
      class SGD < Optimizer
        def initialize(parameters, lr = 0.01)
          super(parameters)
          @lr = lr
        end
  
        def step
          @parameters.each do |param|
            next unless param.grad
            param.data = param.data.map.with_index { |x, i| x - @lr * param.grad[i] }
          end
        end
      end
    end
  end
  