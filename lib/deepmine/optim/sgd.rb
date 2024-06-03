module Deepmine
  module Optim
    class SGD < Optimizer
      def initialize(parameters, lr = 0.01)
        super(parameters)
        @lr = lr
      end

      def step
        @parameters.each do |param|
          param.data = param.data.zip(param.grad).map { |d, g| d - @lr * g }
        end
      end
    end
  end
end
