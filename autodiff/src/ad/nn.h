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
#include "nn/lstm.h"

namespace ad {
namespace nn {

Var L2(const Var& x);

} // nn
} // ad
