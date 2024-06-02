module Deepmine
  module NN
    class Module
      def initialize
        @parameters = []
      end

      def parameters
        @parameters
      end

      def forward(input)
        raise NotImplementedError
      end

      def call(input)
        forward(input)
      end
    end
  end
end
