#ifndef DAISY_HPP
#define DAISY_HPP

#include "algos/Bruteforce.hpp"
#include "algos/LbBruteforce.hpp"
#include "algos/Coconut.hpp"
#include "algos/Messi.hpp"
#include "algos/ParIS.hpp"
#include "algos/Sing.hpp"
#include "algos/hodyssey/Odyssey.hpp"
#include "algos/Sofa.hpp"
#include "algos/Hercules.hpp"
#include "algos/DumpyOS.hpp"
#include "algos/Fresh.hpp"

// Optional: built only with BUILD_FAISS=ON and the benchmark/faiss submodule present.
#ifdef FAISS_ENABLED
#include "algos/Faiss.hpp"
#endif

#endif
