#pragma once

#include <utility>
#include <memory>
#include <set>
#include <vector>

#include "graph.h"
#include "helpers.h"
#include "operators.h"
#include "hashtable.h"

#include "nn/fullyconn.h"
#include "nn/rnn.h"
#include "nn/gru.h"
#include "nn/mrnn.h"
#include "nn/lstm.h"
#include "nn/scrn.h"

namespace ad {
namespace nn {

Var L2(const Var& x);
Var L2(std::vector<Var> x);

} // nn
} // ad
