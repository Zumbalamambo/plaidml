// Copyright 2019, Intel Corporation

#include "llvm/ADT/Optional.h"
#include "mlir/ADT/TypeSwitch.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"

#include "pmlc/dialect/eltwise/ir/ops.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/transforms/passes.h"
#include "pmlc/dialect/pxa/ir/ops.h"

#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

// #include "tile/targets/cpu/heatmap.h" TODO: Lubo for heatmap

namespace pmlc::dialect::pxa {

enum MulOperationType {
  NoneMulOpType,
  FloatTy,
  IntTy,
};

struct GemmOperationMatch {
  explicit GemmOperationMatch() {}

  MulOperationType mulOpType;
  mlir::AffineLoadOp in1Op;
  mlir::AffineLoadOp in2Op;
  AffineReduceOp outOp;
};

using BlockArgumentSet = llvm::SmallPtrSet<mlir::BlockArgument, 8>;
using BlockArgumentMap = llvm::DenseMap<mlir::BlockArgument, unsigned>;

// Number of tensors for the matrix multiplication
const unsigned kNumTensors = 3;
// Number of searching index, i.e., M, N, K
const unsigned kNumIndex = 3;

class Stencil {
 public:
  explicit Stencil(mlir::FuncOp& funcOp); // Lubo const MLIR_AutoStencilPass& opts);
  // Main function
  void DoStenciling(mlir::AffineParallelOp op, mlir::FuncOp func);

 private:
  mlir::FuncOp& func;

  // Index in tensors
  BlockArgumentSet tensorIdxs[kNumTensors];
  // Stride one index for the tensors
  BlockArgumentSet strideOne[kNumTensors];
  // The index used by the output tensor
  BlockArgumentSet outIdxs;
  // The accumulation index
  BlockArgumentSet accIdxs;
  // All used index
  BlockArgumentSet allIdxs;

  // Target tensors, the first two are load, the third is aggregate
  llvm::SmallVector<Value, kNumTensors> tensors;

  // Target tensors strides, the first two are load, the third is aggregate
  llvm::SmallVector<mlir::StrideInfo, kNumTensors> tensorsStrides;

  // The best tensors' order
  unsigned bestTensorsOrder[kNumTensors];
  // The best index (M, N, K)
  mlir::BlockArgument bestIdxs[kNumIndex];
  // The best tiles for (M, N, K)
  unsigned bestTiles[kNumIndex];

  // The best performance
  double bestPerf;
  // The current op
  mlir::AffineParallelOp curOp;

  // Tensors' order
  unsigned tensorsOrder[kNumTensors];

  // M, N, K in inner block
  mlir::BlockArgument innerIdxs[kNumIndex];

  // The matrix_idx for the next search
  unsigned nextMatrixIdx[kNumIndex] = {2, 3, 1};

  // M, N, K's tiles
  unsigned tiles[kNumIndex];

  // Is this a stencillable operation
  mlir::Optional<GemmOperationMatch> getStencillableOperation(mlir::AffineParallelOp op, mlir::FuncOp func);

  // Collect the tensors in the block
  bool CollectTensors(GemmOperationMatch& match);

  // Collect the StrideInfo of the tensors in the block
  bool ComputeStrideInfo(GemmOperationMatch& match);

  // Collect the stride one indexes.
  void strideOneIdxs(mlir::StrideInfo* strideInfoPtr, llvm::SmallVectorImpl<mlir::BlockArgument>* idxs);

  // Collect the index used by value's RefineOp
  BlockArgumentSet RefUsedIdxs(llvm::Optional<mlir::StrideInfo> strideInfo); // Lubo , bool with_conflict);
 // Lubo: TODO: Check these methods are all in use.; 
  // The throughput and startup cost of M*N*K matrix multiplication
  std::pair<double, unsigned> Throughput(unsigned m, unsigned n, unsigned k);
  // Evaluate the performance of the current searching state
  double Evaluate();
  // Search tensors' order
  void SearchTensorsOrder();
  // Search the index, i.e., M, N, K, for the inner block
  void SearchIndex(unsigned matrix_idx);
  // For (M, N, K) in the inner block, search their tiles
  void SearchTiles(unsigned idx);
  
  // Transform the current AffineParallelOp
  void Transform();

  
  // Test if idx is conflict with any index in innerIdxs
  bool ConflictInnerIndex(mlir::BlockArgument idx);
  // Test if idx in tensors[tensor_idx] is stride one index
  bool IsStrideOne(mlir::BlockArgument idx, unsigned tensor_idx);
  // Test if idx in the tensors are stride one
  bool ValidateStrideOne(mlir::BlockArgument idx, unsigned matrix_idx);
  // Test if idx exists in tensorIdxs[tensor_idx]
  bool IndexExists(mlir::BlockArgument idx, unsigned tensor_idx);
  // Test if idx exists in the right place
  bool ValidateIndexExistance(mlir::BlockArgument idx, unsigned matrix_idx);

  int64_t idxRange(mlir::BlockArgument idx);
  void getAllIndex(mlir::AffineParallelOp op, llvm::SmallVectorImpl<std::pair<mlir::BlockArgument, unsigned>>* idxs);
};

Stencil::Stencil(mlir::FuncOp& funcOp) : func(funcOp) { // Lubo const MLIR_AutoStencilPass& opts) : options(opts) {
  // for (unsigned i = 0; i < kHeatmapSize; ++i) {
  //   kHeatmap.emplace(std::make_tuple(kHeatmapKeys[i][0], kHeatmapKeys[i][1], kHeatmapKeys[i][2]), kHeatmapValues[i]);
  // }
}

// Collect the index used by value's RefineOp Lubo
// If with_conflict is true, specify the conflict between index, i.e., both index can't be in the inner block
BlockArgumentSet Stencil::RefUsedIdxs(llvm::Optional<mlir::StrideInfo> strideInfo) { // Lubo , bool with_conflict) {
  BlockArgumentSet used_idxs;
  if (strideInfo.hasValue()) {
    for (auto kv : strideInfo.getValue().strides) {
      used_idxs.insert(kv.first);
    }
  }

  return used_idxs;
}

bool Stencil::CollectTensors(GemmOperationMatch& match) {    
  IVLOG(1, "Lubo:Lubo:Lubo29.4:" << match.in1Op << ":" << match.in2Op << ":" << match.outOp);
    
  // Lubo end
  // const mlir::AffineLoadOp& in1Op = match.in1Op;
  // const mlir::AffineLoadOp& in2Op = matchin2Op;
  // const AffineReduceOp& outOp = match.outOp;
  //       // Lubo
  // IVLOG(1, "Lubo:Lubo:Lubo29.5" << matchPtr->matchedOp << ":" << gemmMatchPtr->in1Op << ":" << gemmMatchPtr->in2Op << ":" << gemmMatchPtr->outOp);
  // IVLOG(1, "Lubo:Lubo:Lubo29.6" << in1Op << ":" << in2Op << ":" << outOp);
  // Lubo end
// Lubo
// Lubo
  // IVLOG(1, "Lubo:Lubo:Lubo31.5" << matchPtr->matchedOp << ":" << gemmMatchPtr->in1Op << ":" << gemmMatchPtr->in2Op << ":" << gemmMatchPtr->outOp);
  // // Lubo end
  // // Lubo
  // IVLOG(1, "Lubo:Lubo:Lubo31.6" << matchPtr->matchedOp << ":" << gemmMatchPtr->in1Op << ":" << gemmMatchPtr->in2Op << ":" << gemmMatchPtr->outOp);
  // Lubo end
  if (!match.in1Op ||
      !match.in2Op ||
      !match.outOp) {
        // Lubo
        // Lubo
  // IVLOG(1, "Lubo:Lubo:Lubo31.8" << matchPtr->matchedOp << ":" << gemmMatchPtr->in1Op << ":" << gemmMatchPtr->in2Op << ":" << gemmMatchPtr->outOp);
  // Lubo end
  //IVLOG(1, "Lubo:Lubo:Lubo31.9" << gemmMatch.matchedOp << ":" << gemmMatch.in1Op << ":" << gemmMatch.in2Op << ":" << gemmMatch.outOp);
  // Lubo end

        // Lubo
  IVLOG(1, "Lubo:Lubo:Lubo32");
  // Lubo end
        return false;
  }

// Lubo
  // IVLOG(1, "Lubo:Lubo:Lubo51: " << match.in1Op << ":" << match.in1Op.getMemRefOperandIndex()); // Lubo  << ":" << gemmMatchPtr->in1Op->getMemRef());
  // IVLOG(1, "Lubo:Lubo:Lubo52: " << gemmMatchPtr->in1Op << ":" << gemmMatchPtr->in1Op.getOperand(gemmMatchPtr->in1Op.getMemRefOperandIndex())); // Lubo  << ":" << gemmMatchPtr->in1Op->getMemRef());
  
  // Lubo end

  tensors.push_back(match.in1Op.getMemRef());
  tensors.push_back(match.in2Op.getMemRef());
  tensors.push_back(match.outOp.out());
  // Lubo
  IVLOG(1, "Lubo:Lubo:Lubo52");
  // Lubo end


// Lubo
  IVLOG(1, "Lubo:Lubo:Lubo33");
  // Lubo end
  return tensors.size() == kNumTensors;
}

bool Stencil::ComputeStrideInfo(GemmOperationMatch& match) {
  if (match.in1Op == nullptr ||
      match.in2Op == nullptr ||
      match.outOp == nullptr) {
        return false;
  }

  tensorsStrides.push_back(*(computeStrideInfo(match.in1Op).getPointer()));
  tensorsStrides.push_back(*(computeStrideInfo(match.in2Op).getPointer()));
  tensorsStrides.push_back(*(computeStrideInfo(match.outOp).getPointer()));
  return true;
}

mlir::Optional<GemmOperationMatch> Stencil::getStencillableOperation(mlir::AffineParallelOp op, mlir::FuncOp func) {
  auto *body = op.getBody();
  // Get the instructions in the body and match for load, load, mulXXX, reduce add operations.
  // For everything else we fail.
  for (mlir::Block::reverse_iterator iterB = body->rbegin(), iterE = body->rend(); iterB != iterE; ++iterB) {
    IVLOG(1, "Lubo:Lubo:Lubo8" << mlir::debugString(*iterB));
  }

  // Lubo GemmOperationMatch* gemmMatchedPtr = nullptr;
  mlir::Optional<GemmOperationMatch> ret;
  func.walk([&](AffineReduceOp reduceOp) {
    IVLOG(1, "Lubo:Lubo:Lubo9"); //  << mlir::debugString((reduceOp)));

    // Not check the reduceOp aggregation.
    if (reduceOp.agg() != AggregationKind::add) {
      op.emitRemark("the reduce operation is not addition");
      return;
    }

    // Get the in tensors for the reduce op.
    Value reduceIn = reduceOp.val();
    MulOperationType mulOpType = MulOperationType::NoneMulOpType;

    // Make sure the in for the reduce is a result of a multiplication.
    auto valDef = reduceIn.getDefiningOp();

    if (valDef == nullptr) {
      op.emitRemark("the source of the reduce operation is not defined in this block");
      return;
    }

    mlir::MulFOp mulfOp  = llvm::dyn_cast_or_null<mlir::MulFOp>(valDef);
    mlir::MulIOp muliOp  = llvm::dyn_cast_or_null<mlir::MulIOp>(valDef);
    if (mulfOp == nullptr && muliOp == nullptr) {
      op.emitRemark("The source of the reduce is not a multiplication operation");
      return;
    }

    mlir::AffineLoadOp lhs;
    mlir::AffineLoadOp rhs;
    if (mulfOp != nullptr) {
      mulOpType = MulOperationType::FloatTy;
      lhs = llvm::dyn_cast_or_null<mlir::AffineLoadOp>(mulfOp.lhs().getDefiningOp());
      rhs = llvm::dyn_cast_or_null<mlir::AffineLoadOp>(mulfOp.rhs().getDefiningOp());
    }
    else if (muliOp != nullptr) {
      mulOpType = MulOperationType::IntTy;
      lhs = llvm::dyn_cast_or_null<mlir::AffineLoadOp>(muliOp.lhs().getDefiningOp());
      rhs = llvm::dyn_cast_or_null<mlir::AffineLoadOp>(muliOp.rhs().getDefiningOp());
    }
    else {
      op.emitRemark("unhandled multiplication type in stenciling");
    }

// Lubo
    IVLOG(1, "Lubo:Lubo:Lubo19" << mulOpType << ":" << lhs << ":" << rhs << ":" << reduceOp);
    // Lubo end
    // Now verify the types of the operands of the mulOp must be affine.load operations.
    if (lhs == nullptr || rhs == nullptr) {
      op.emitRemark("the lhs or rhs of the mul operation are not an affne.load operations");
      return;
    }

    // TODO: Need a bit better liveness analysis here to make sure the parameters 
    // of any of the above 4 operations are not used in operations with
    // side effects - store, calls, etc.

    // Fill the values for the in/out/type of multiplication, etc.
    GemmOperationMatch gemmMatch = GemmOperationMatch();
    gemmMatch.mulOpType = mulOpType;
    gemmMatch.in1Op = lhs;
    gemmMatch.in2Op = rhs;
    gemmMatch.outOp = reduceOp;
    ret = gemmMatch;
  });

  return ret;
}

// Get the indexes with stride of one.
// TODO: This might be useful as an utility method.
void Stencil::strideOneIdxs(mlir::StrideInfo* strideInfoPtr, llvm::SmallVectorImpl<mlir::BlockArgument>* idxs) {
  if (strideInfoPtr != nullptr) {
    for (auto [arg, scale] : strideInfoPtr->strides) {
      if (scale == 1) {
        idxs->push_back(arg);
      }
    }
  }
}

// Test if idx in tensors[tensor_idx] is stride one index
bool Stencil::IsStrideOne(mlir::BlockArgument idx, unsigned tensor_idx) {
  return strideOne[tensor_idx].find(idx) != strideOne[tensor_idx].end();
}

// Test if idx in the tensors are stride one
bool Stencil::ValidateStrideOne(mlir::BlockArgument idx, unsigned matrix_idx) {
  switch (matrix_idx) {
    case 0: {
      // Test if M is stride one for B(1) and C(2)
      return IsStrideOne(idx, tensorsOrder[1]) && IsStrideOne(idx, tensorsOrder[2]);
    }
    case 1: {
      // N is not restricted for stride one
      return true;
    }
    case 2: {
      // Test if K is stride one for A(0)
      return IsStrideOne(idx, tensorsOrder[0]);
    }
    default: {
      throw std::runtime_error("Wrong matrix_idx.");
    }
  }
  return false;
}

bool Stencil::IndexExists(mlir::BlockArgument idx, unsigned tensor_idx) {
  return tensorIdxs[tensor_idx].find(idx) != tensorIdxs[tensor_idx].end();
}

// Confirm if idx exists in the right place
bool Stencil::ValidateIndexExistance(mlir::BlockArgument idx, unsigned matrix_idx) {
  switch (matrix_idx) {
    case 0: {
      // Test if M exists in B and C, does not exist in A
      return !IndexExists(idx, tensorsOrder[0]) &&  //
             IndexExists(idx, tensorsOrder[1]) &&   //
             IndexExists(idx, tensorsOrder[2]);
    }
    case 1: {
      // Test if N exists in A and C, does not exist in B
      return IndexExists(idx, tensorsOrder[0]) &&   //
             !IndexExists(idx, tensorsOrder[1]) &&  //
             IndexExists(idx, tensorsOrder[2]);
    }
    case 2: {
      // Test if K exists in A and B, does not exist in C
      return IndexExists(idx, tensorsOrder[0]) &&  //
             IndexExists(idx, tensorsOrder[1]) &&  //
             !IndexExists(idx, tensorsOrder[2]);
    }
    default: {
      throw std::runtime_error("Wrong matrix_idx.");
    }
  }
  return false;
}

// Search for matrix index (0 for M, 1 for N, 2 for K)
void Stencil::SearchIndex(unsigned matrix_idx) {
  if (matrix_idx >= kNumIndex) {
    // We have the index and then search the tiles for these index
    SearchTiles(0);
    return;
  }
  auto& idxs = (matrix_idx == kNumIndex - 1) ? allIdxs : outIdxs;
  for (auto idx : idxs) {
    if (// Lubo conflicts !ConflictInnerIndex(idx) &&            //
        ValidateStrideOne(idx, matrix_idx) &&  //
        ValidateIndexExistance(idx, matrix_idx)) {
      innerIdxs[matrix_idx] = idx;
      SearchIndex(nextMatrixIdx[matrix_idx]);
    }
  }
}

void Stencil::SearchTensorsOrder() {
  // A B C, Search M(0) first as M is most restricted index
  tensorsOrder[0] = 0;
  tensorsOrder[1] = 1;
  tensorsOrder[2] = 2;
  SearchIndex(0);
  // B A C, Search M(0) first as M is most restricted index
  tensorsOrder[0] = 1;
  tensorsOrder[1] = 0;
  SearchIndex(0);
}

void Stencil::SearchTiles(unsigned idx) {
  if (idx >= kNumIndex) {
    double performance = Evaluate();
    if (performance < bestPerf) {
      bestPerf = performance;
      for (unsigned i = 0; i < kNumTensors; ++i) {
        bestTensorsOrder[i] = tensorsOrder[i];
      }
      for (unsigned i = 0; i < kNumIndex; ++i) {
        bestIdxs[i] = innerIdxs[i];
        bestTiles[i] = tiles[i];
      }
    }
    return;
  }
}

// TODO: Lubo this might need to go into an utility.
int64_t Stencil::idxRange(mlir::BlockArgument idx) {
  auto pf = mlir::cast<mlir::AffineParallelOp>(idx.getOwner()->getParentOp());
  auto ranges = pf.getConstantRanges();
  if (ranges != llvm::None) {
    return (*ranges.getPointer())[idx.getArgNumber()];
  }
  return -1;
}

double Stencil::Evaluate() {
  unsigned tot_inner_loop = tiles[0] * tiles[1] * tiles[2];
  double throughput;
  unsigned startup_cost;
  std::tie(throughput, startup_cost) = Throughput(tiles[0], tiles[1], tiles[2]);
  if (throughput == 0) {
    return std::numeric_limits<double>::max();
  }
  double inner_time = tot_inner_loop / throughput;
  // Lubo IVLOG(3, "Inner: loop = " << tot_inner_loop << " inner_time = " << inner_time);
  for (unsigned i = 0; i < kNumIndex; ++i) {
    // Lubo IVLOG(3, idxName(innerIdxs[i]).str() << ": " << tiles[i]);
  }

  llvm::DenseMap<mlir::BlockArgument, unsigned> middle_idxs;
  for (auto idx : accIdxs) {
    middle_idxs.try_emplace(idx, idxRange(idx));
  }
  for (unsigned i = 0; i < kNumIndex; ++i) {
    auto it = middle_idxs.find(innerIdxs[i]);
    if (it != middle_idxs.end()) {
      it->second = (it->second - 1) / tiles[i] + 1;
    }
  }
  unsigned tot_middle_loop = 1;
  for (auto& kvp : middle_idxs) {
    tot_middle_loop *= kvp.second;
  }
  // Lubo IVLOG(3, "Middle: loop = " << tot_middle_loop);
  for (auto& kvp : middle_idxs) {
    if (kvp.second > 1) {
      // Lubo IVLOG(3, idxName(kvp.first).str() << ": " << kvp.second);
    }
  }

  llvm::DenseMap<mlir::BlockArgument, unsigned> outer_idxs;
  for (auto idx : outIdxs) {
    outer_idxs.try_emplace(idx, idxRange(idx));
  }
  for (unsigned i = 0; i < kNumIndex; ++i) {
    auto it = outer_idxs.find(innerIdxs[i]);
    if (it != outer_idxs.end()) {
      it->second = (it->second - 1) / tiles[i] + 1;
    }
  }
  unsigned tot_outer_loop = 1;
  for (auto& kvp : outer_idxs) {
    tot_outer_loop *= kvp.second;
  }
  // Lubo IVLOG(3, "Outer: loop = " << tot_outer_loop);
  for (auto& kvp : outer_idxs) {
    if (kvp.second > 1) {
      // Lubo IVLOG(3, idxName(kvp.first).str() << ": " << kvp.second);
    }
  }

  unsigned outer_batches = (tot_outer_loop - 1) / std::thread::hardware_concurrency() + 1;
  double perf = outer_batches * tot_middle_loop * (startup_cost + inner_time);
  // Lubo IVLOG(3, "Performance = " << perf);

  return perf;
}

std::pair<double, unsigned> Stencil::Throughput(unsigned m, unsigned n, unsigned k) {
  // Lubo TODO: HeatMap
  // auto iter = kHeatmap.find(std::make_tuple(m, n, k));
  // if (iter != kHeatmap.end()) {
  //   return std::make_pair(iter->second, 7); // Lubo options.startup_cost());
  // }
  // // We mainly care about M and K. If both (m, n - 1, k) and (m, n + 1, k) exist,
  // // we may use their average value for prediction
  // auto iter0 = kHeatmap.find(std::make_tuple(m, n - 1, k));
  // Lubo: TODO: HeatMap
   // if (n == 1 || iter0 != kHeatmap.end()) {
  //   auto iter1 = kHeatmap.find(std::make_tuple(m, n + 1, k));
  //   if (iter1 != kHeatmap.end()) {
  //     return std::make_pair((n > 1) ? ((iter0->second + iter1->second) / 2) : iter1->second, 32 /*startup_cose Lubo */ ); // Lubo options.startup_cost());
  //   }
  // }

  // TODO: Add support for the special cases.
  // If we cannot find (m, n, k) in the heatmap, try the special cases
  // Lubo for (const auto& spec : options.special_stencils()) {
  //   bool match = true;
  //   for (const auto& rule : spec.idxs()) {
  //     if (rule.name() == "m") {
  //       if (rule.size() > 0 && static_cast<unsigned>(rule.size()) != m) {
  //         match = false;
  //         break;
  //       }
  //     } else if (rule.name() == "n") {
  //       if (rule.size() > 0 && static_cast<unsigned>(rule.size()) != n) {
  //         match = false;
  //         break;
  //       }
  //     } else if (rule.name() == "k") {
  //       if (rule.size() > 0 && static_cast<unsigned>(rule.size()) != k) {
  //         match = false;
  //         break;
  //       }
  //     }
  //   }
  //   if (match) {
  //     return std::make_pair(0.001, spec.startup_cost());
  //   }
  // }
  return std::make_pair(0.1, 32); // Lubo 
}

// Lubo TODO: Move this to an utility?
void Stencil::getAllIndex(mlir::AffineParallelOp op, llvm::SmallVectorImpl<std::pair<mlir::BlockArgument, unsigned>>* idxs) {
  auto rangesPtr = op.getConstantRanges();
  if (!rangesPtr) {
    return;
  }

  auto ranges = *rangesPtr;
  auto blkArgs = op.getBody()->getArguments();
  for (unsigned i = 0; i < ranges.size(); i++) {
    unsigned range = ranges[i];
    idxs->push_back(std::make_pair(blkArgs[i], range));
  }
}

void Stencil::Transform() {
  // TODO: Lubo Transform needs to be fully implemented.
  // if (bestPerf == std::numeric_limits<double>::max()) {
  //   // Lubo IVLOG(1, "No tile plan for stencil.");
  //   return;
  // }

  llvm::SmallVector<std::pair<mlir::BlockArgument, unsigned>, 8> idxs;
  getAllIndex(curOp, &idxs);

  // Lubo
  // 
  // Need matcher for load, load, mul, reduce sequernce.
  // example https://reviews.llvm.org/D74401
  // Line 2548
  // 
  // AffineValueMap getRangesValueMap(); // Lubho this returns 
  // To get the generated induction vars op..getBody().getArguments()
  // Lubo auto avm = curOp.getRangesValueMap();
  // Lubo end
  llvm::DenseMap<mlir::BlockArgument, unsigned> bestTileByArg;
  for (unsigned i = 0; i < kNumIndex; ++i) {
    bestTileByArg[bestIdxs[i]] = bestTiles[i];
  }
  llvm::SmallVector<int64_t, 8> inner_sizes;
  for (const auto& idx : idxs) {
    auto it = bestTileByArg.find(idx.first);
    inner_sizes.push_back(it == bestTileByArg.end() ? 1 : it->second);
  }
  // The first tiling: split outer&middle and inner
  // Lubo Tile(curOp, inner_sizes);
  // Lubo mlir::AffineParallelOp inner = mlir::dyn_cast<mlir::AffineParallelOp>(curOp.region().front().front());

  // Lubo: TODO: More transformation here to lower to the tiles (insert AffineParallelOp or to xsmm directly). 
  // Set AffineParallelOp tags
  // setOpAttrUnit(curOp, curOp.getBodyBuilder(), "mac");
  // setOpAttrUnit(inner, inner.getBodyBuilder(), "mac_inner");
  // setOpAttrUnit(inner, inner.getBodyBuilder(), "xsmm");

  // // Set RefineOp tags
  // setOpAttrUnit(tensors[bestTensorsOrder[0]].getDefiningOp(), inner.getBodyBuilder(), "A");
  // setOpAttrUnit(tensors[bestTensorsOrder[1]].getDefiningOp(), inner.getBodyBuilder(), "B");
  // setOpAttrUnit(tensors[bestTensorsOrder[2]].getDefiningOp(), inner.getBodyBuilder(), "C");

  // Set index tags
  // StringRef m_idx = idxName(bestIdxs[0]);
  // setIdxAttrUnit(inner, m_idx, "stencil");
  // setIdxAttrUnit(inner, m_idx, "stencil_m");
  // StringRef n_idx = idxName(bestIdxs[1]);
  // setIdxAttrUnit(inner, n_idx, "stencil");
  // setIdxAttrUnit(inner, n_idx, "stencil_n");
  // StringRef k_idx = idxName(bestIdxs[2]);
  // setIdxAttrUnit(inner, k_idx, "stencil");
  // setIdxAttrUnit(inner, k_idx, "stencil_k");
}
void Stencil::DoStenciling(mlir::AffineParallelOp op, mlir::FuncOp func) {
  // Initialization
  tensors.clear();
  tensorsStrides.clear();
  bestPerf = std::numeric_limits<double>::max();
  curOp = op;
  auto matchOpt = getStencillableOperation(op, func);
  if (!matchOpt.hasValue()) {
    return;
  }
  
  GemmOperationMatch& match = matchOpt.getValue();  
  // Lubo
    IVLOG(1, "Lubo:Lubo:Lubo23");
    // Lubo end
  if (!CollectTensors(match)) {
    // Lubo
    IVLOG(1, "Lubo:Lubo:Lubo16");
    // Lubo end
    return;
  }

  // Lubo
    IVLOG(1, "Lubo:Lubo:Lubo24");
    // Lubo end
  if (!ComputeStrideInfo(match)) {
    // Lubo
    IVLOG(1, "Lubo:Lubo:Lubo17");
    // Lubo end
    return;
  }
  
  // The last tensor is the output.
  tensorIdxs[kNumIndex - 1] = RefUsedIdxs(tensorsStrides[kNumIndex - 1]);
  outIdxs = tensorIdxs[kNumIndex - 1];
  accIdxs.clear();
  for (unsigned i = 0; i < kNumIndex - 1; ++i) {
    tensorIdxs[i] = RefUsedIdxs(tensorsStrides[i]);
    for (auto idx : tensorIdxs[i]) {
      if (outIdxs.find(idx) == outIdxs.end()) {
        accIdxs.insert(idx);
      }
    }
  }

  allIdxs = accIdxs;
  allIdxs.insert(outIdxs.begin(), outIdxs.end());

  // Collect stride-one index
  for (unsigned i = 0; i < kNumTensors; ++i) {
    llvm::SmallVector<mlir::BlockArgument, 8> idxs;
    strideOneIdxs(&tensorsStrides[i], &idxs);
    strideOne[i].insert(idxs.begin(), idxs.end());
  }

  // Search tensors' order, inner index and their tiles
  SearchTensorsOrder();

  // Transform
  Transform();
  
  IVLOG(1, "Lubo:Lubo:Lubo18");
}

void StencilPass::runOnFunction() {
  IVLOG(1, "Lubo:Lubo:Lubo2");
// Lubo #if defined(_WIN32) || defined(_WIN64)
//   // As XSMM is not stable on Windows now, we disable this pass on Windows.
//   // When the XSMM issue is solved, remove this section.
//   return;
// #endif

  // Lubo auto reqs = vertexai::tile::stripe::FromProto(options.reqs());
  auto func = this->getFunction();
  // Lubo if (options.only_po2().size() != kNumIndex || options.only_even().size() != kNumIndex) {
  // Lubo   throw std::runtime_error("The size of only_po2 array or only_even array is incorrect.");
  // Lubo }
  Stencil as(func); // Lubo options);
  func.walk([&](mlir::AffineParallelOp op) {
    as.DoStenciling(op, func);
    // Lubo as.DoStenciling(op);
    IVLOG(1, "Lubo:Lubo:Lubo1");
  });
}

std::unique_ptr<mlir::Pass> createStencilPass() {
  return std::make_unique<StencilPass>();
}

}  // namespace pmlc::dialect::pxa

