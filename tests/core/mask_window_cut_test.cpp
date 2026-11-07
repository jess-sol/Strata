// The cut of a decode window to the rows one mask reply may hold (include/strata/core/token_mask.hpp): CPU only,
// header-only.
#include "strata/core/token_mask.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

struct Window { int T, chain_n; };

Window cut(int T, int T_mtp, int chain_n, int max_rows) {
    strata::core::mask_window_cut(T, chain_n, T_mtp, max_rows);
    return {T, chain_n};
}
}  // namespace

int main() {
    Window w = cut(5, 5, 0, 8);
    check(w.T == 5 && w.chain_n == 0, "a window within the limit is left as it is");
    w = cut(8, 4, 4, 8);
    check(w.T == 8 && w.chain_n == 4, "a window at the limit is left as it is");
    w = cut(12, 12, 0, 8);
    check(w.T == 8 && w.chain_n == 0, "12 rows of suffix drafts become the 8 a reply holds");
    w = cut(10, 4, 6, 8);
    check(w.T == 8 && w.chain_n == 4, "the chained drafts behind the MTP rows keep what still fits");
    w = cut(12, 10, 2, 8);
    check(w.T == 8 && w.chain_n == 0, "a limit inside the MTP rows leaves no chained draft");
    std::printf("mask_window_cut_test: %d checks passed\n", checks);
    return 0;
}
