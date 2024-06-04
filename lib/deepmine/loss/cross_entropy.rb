module Deepmine
    module Loss
      class CrossEntropy
        def self.compute(pred, target)
          batch_size = pred.data.size
          loss = -pred.data.zip(target.data).map do |p, t|
            p.map.with_index { |pi, idx| t[idx] * Math.log(pi + 1e-9) }.sum
          end.sum / batch_size
          Tensor.new([loss])
        end
  
        def self.backward(pred, target)
          batch_size = pred.data.size
          Tensor.new(pred.data.zip(target.data).map do |p, t|
            p.map.with_index { |pi, idx| (pi - t[idx]) / batch_size }
          end)
        end
      end
    end
  end
  