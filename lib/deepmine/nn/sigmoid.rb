module Deepmine
    module NN
      class Sigmoid < Module
        def forward(input)
          Tensor.new(input.data.map { |x| x.map { |xi| 1.0 / (1.0 + Math.exp(-xi)) } })
        end
      end
    end
  end
  