#include "node_operators.h"
#include "expression_graph.h"

#include "tensors/tensor_operators.h"

namespace marian {

ConstantNode::ConstantNode(Ptr<ExpressionGraph> graph,
                           const Shape& shape,
                           const NodeInitializer& init)
      : Node(graph, shape, graph->getBackend()->valueType()),
        init_(new NodeInitializer(init)),
        initialized_(false) {
    setTrainable(false);
  }


size_t ConstantNode::allocate() {
  size_t elements = 0;
  if(!val_) {
    graph()->allocateForward(shared_from_this());
    elements = val_->shape().elements();
  }
  return elements;
}

void ConstantNode::init() {
  if(!initialized_) {
    (*init_)(val_);
    initialized_ = true;
  }
  init_.reset();
}

ParamNode::ParamNode(Ptr<ExpressionGraph> graph,
                     const Shape& shape,
                     const NodeInitializer& init,
                     bool fixed)
    : Node(graph, shape, graph->getBackend()->valueType()), // TODO: add value_type
      init_(new NodeInitializer(init)),
      initialized_(false) {
  setTrainable(!fixed);
  setMemoize(graph->isInference());
}

void ParamNode::init() {
  if(!initialized_) {
    (*init_)(val_);
    initialized_ = true;
  }
  init_.reset();
}
}
