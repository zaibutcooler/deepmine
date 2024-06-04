module Deepmine
    module Loss
      class MSE
        def self.compute(pred, target)
          Tensor.new([pred.data.zip(target.data).map { |p, t| (p - t)**2 }.sum / pred.data.size])
        end
  
        def self.backward(pred, target)
          pred.data.zip(target.data).map { |p, t| 2 * (p - t) / pred.data.size.to_f }
        end
      end
    end
  end
  