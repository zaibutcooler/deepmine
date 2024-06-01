module Deepmine
  module NN
    class ReLU < Module
      def forward(input)
        Tensor.new(input.data.map { |x| x.map { |xi| [xi, 0].max } })
      end
    end
  end
end
