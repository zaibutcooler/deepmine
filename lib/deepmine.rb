require_relative 'deepmine/tensor'
require_relative 'deepmine/autograd'
require_relative 'deepmine/nn/module'
require_relative 'deepmine/nn/linear'
require_relative 'deepmine/nn/relu'
require_relative 'deepmine/optim/optimizer'
require_relative 'deepmine/optim/sgd'

module Deepmine

  def self.version
    "0.1.0"
  end

  def self.do_something
    puts "I am swimming"
  end

end