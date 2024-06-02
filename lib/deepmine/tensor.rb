module Deepmine
  class Tensor
    attr_accessor :data, :grad

    def initialize(data)
      @data = data
      @grad = nil
    end

    def +(other)
      Tensor.new(@data.zip(other.data).map { |a, b| a + b })
    end

    def *(other)
      Tensor.new(@data.zip(other.data).map { |a, b| a * b })
    end

    def sum
      @data.sum
    end

    def mean
      @data.sum / @data.size.to_f
    end

    def zero_grad!
      @grad = Array.new(@data.size, 0)
    end

    def backward!(grad)
      @grad = grad
    end
  end
end
