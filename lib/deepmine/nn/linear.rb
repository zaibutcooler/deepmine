module Deepmine
  module NN
    class Linear < Module
      def initialize(input_size, output_size)
        super()
        @weights = Tensor.new(Array.new(input_size) { Array.new(output_size) { rand } })
        @bias = Tensor.new(Array.new(output_size) { rand })
        @parameters = [@weights, @bias]
      end

      def forward(input)
        output = input.data.map do |x|
          @weights.data.transpose.map do |w|
            x.zip(w).map { |xi, wi| xi * wi }.sum
          end
        end
        Tensor.new(output.map { |o| o.zip(@bias.data).map { |oi, bi| oi + bi } })
      end
    end
  end
end
