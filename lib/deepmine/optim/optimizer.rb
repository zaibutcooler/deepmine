module Deepmine
    module Optim
      class Optimizer
        def initialize(parameters)
          @parameters = parameters
        end
  
        def step
          raise NotImplementedError
        end
  
        def zero_grad
          @parameters.each { |param| param.grad = nil }
        end
      end
    end
  end
  