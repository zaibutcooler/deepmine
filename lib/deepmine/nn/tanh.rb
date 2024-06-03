module Deepmine
    module NN
      class Tanh < Module
        def forward(input)
          Tensor.new(input.data.map { |x| x.map { |xi| Math.tanh(xi) } })
        end
      end
    end
  end
  