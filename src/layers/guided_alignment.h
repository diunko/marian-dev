#pragma once

#include "marian.h"

namespace marian {

static inline Expr guidedAlignmentCost(Ptr<ExpressionGraph> graph,
                                       Ptr<data::CorpusBatch> batch,
                                       Ptr<Options> options,
                                       Expr att) {
  using namespace keywords;

  int dimTrg =   att->shape()[-4];
  int dimSrc =   att->shape()[-3];
  int dimBatch = att->shape()[-2];

  std::cerr << dimBatch << " " << dimSrc << " " << dimTrg << std::endl;

  auto aln = graph->constant({dimTrg, dimSrc, dimBatch, 1},
                             inits::from_vector(batch->getGuidedAlignment()));

  debug(att, "att");
  debug(aln, "aln");

  std::string guidedCostType
      = options->get<std::string>("guided-alignment-cost");

  Expr alnCost;
  float eps = 1e-6;
  if(guidedCostType == "mse") {
    alnCost = sum(flatten(square(att - aln))) / (2 * dimBatch);
  } else if(guidedCostType == "mult") {
    alnCost = -log(sum(flatten(att * aln)) + eps) / dimBatch;
  } else if(guidedCostType == "ce") {
    alnCost = -sum(flatten(aln * log(att + eps))) / dimBatch;
  } else {
    ABORT("Unknown alignment cost type {}", guidedCostType);
  }

  float guidedScalar = options->get<float>("guided-alignment-weight");
  return guidedScalar * alnCost;
}
}
