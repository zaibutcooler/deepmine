module Deepmine
    module NN
      class Conv2D < Module
        attr_reader :in_channels, :out_channels, :kernel_size, :stride, :padding
  
        def initialize(in_channels, out_channels, kernel_size, stride=1, padding=0)
          super()
          @in_channels = in_channels
          @out_channels = out_channels
          @kernel_size = kernel_size
          @stride = stride
          @padding = padding
          @weights = Tensor.new(Array.new(out_channels) { Array.new(in_channels) { Array.new(kernel_size) { Array.new(kernel_size) { rand } } } })
          @bias = Tensor.new(Array.new(out_channels) { rand })
          @parameters = [@weights, @bias]
        end
  
        def forward(input)
          batch_size, _, input_height, input_width = input.data.size, input.data[0].size, input.data[0][0].size, input.data[0][0][0].size
          output_height = (input_height - @kernel_size + 2 * @padding) / @stride + 1
          output_width = (input_width - @kernel_size + 2 * @padding) / @stride + 1
  
          output = Array.new(batch_size) { Array.new(@out_channels) { Array.new(output_height) { Array.new(output_width) { 0 } } } }
  
          input.data.each_with_index do |batch, b|
            @out_channels.times do |oc|
              (0...output_height).each do |i|
                (0...output_width).each do |j|
                  sum = 0
                  @in_channels.times do |ic|
                    (0...@kernel_size).each do |ki|
                      (0...@kernel_size).each do |kj|
                        x = i * @stride + ki - @padding
                        y = j * @stride + kj - @padding
                        if x >= 0 && y >= 0 && x < input_height && y < input_width
                          sum += batch[ic][x][y] * @weights.data[oc][ic][ki][kj]
                        end
                      end
                    end
                  end
                  output[b][oc][i][j] = sum + @bias.data[oc]
                end
              end
            end
          end
  
          Tensor.new(output)
        end
      end
    end
  end
  