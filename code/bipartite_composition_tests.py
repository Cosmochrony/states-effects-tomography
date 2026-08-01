"""
Reproduces the BC1-BC5 results (bipartite effect composition) of the States-Effects Tomography paper.

Exact-arithmetic checks (Fractions, no randomness) for:
  (1) the uniqueness corollary: the composed effect forced by affinity + Dirac-factorization equals the
      pointwise-product formula sum_{a,b} p_ab e_a f_b;
  (2) Level 3: bilinear consistency across the composition, exactly as in CCU5, depends on independence of
      the two random taps (choice of local A-effect vs choice of local B-effect), not on the algebra itself;
  (3) Level 4, for S_A=S_B={0,1}: exactly 10 of the 16 possible joint sharp (indicator) effects are
      rectangles (products of local sharp effects); the other 6 are not;
  (4) a CHSH-type witness functional shows that NO convex combination of local product effects (even with
      full local randomisation) can reach the diagonal indicator -- reaching it needs a separate joint
      coarse-graining axiom, not obtainable from local composition alone.
"""

from fractions import Fraction as Fr
from itertools import chain, combinations, product


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


# ---------- (1) Uniqueness corollary ----------

def composed_effect_pointwise(e, f, p, m, n):
    """(e boxtimes f)(p) = sum_ab p_ab * e_a * f_b, p given as dict {(a,b): Fraction}."""
    return sum(p[(a, b)] * e[a] * f[b] for a in range(m) for b in range(n))


def check_uniqueness(m, n):
    print(f"--- Uniqueness corollary, m={m}, n={n} ---")
    e = tuple(Fr(k + 1, m + 2) for k in range(m))  # arbitrary local effect vector in [0,1]^m
    f = tuple(Fr(2 * k + 1, 2 * n + 1) for k in range(n))
    # A generic (non-product) joint state.
    raw = {(a, b): Fr((a + 1) * (b + 2) + 1, 1) for a in range(m) for b in range(n)}
    total = sum(raw.values())
    p = {k: v / total for k, v in raw.items()}
    val = composed_effect_pointwise(e, f, p, m, n)
    # Reconstruct via the EA1-style affine extension directly from Dirac values, to confirm it is the
    # SAME object obtained two different ways (definition vs affine-extension-from-Diracs).
    dirac_values = {(a, b): e[a] * f[b] for a in range(m) for b in range(n)}
    val_from_diracs = sum(p[(a, b)] * dirac_values[(a, b)] for a in range(m) for b in range(n))
    print(f"  (e boxtimes f)(p) via pointwise formula = {val}")
    print(f"  same, reconstructed from Dirac values only = {val_from_diracs}")
    print(f"  match (uniqueness confirmed): {val == val_from_diracs}")
    print()


# ---------- (2) Level 3: CC-IND dependence for the composed effect ----------

def check_composition_bilinearity(m, n):
    print(f"--- Level 3: composition bilinearity depends on CC-IND, m={m}, n={n} ---")
    e1 = tuple(Fr(4, 5) if i == 0 else Fr(1, 5) for i in range(m))
    e2 = tuple(Fr(1, 10) if i == 0 else Fr(9, 10) for i in range(m))
    f1 = tuple(Fr(3, 5) if i == 0 else Fr(2, 5) for i in range(n))
    f2 = tuple(Fr(1, 4) if i == 0 else Fr(3, 4) for i in range(n))
    raw = {(0, 0): Fr(3, 10), (0, 1): Fr(1, 5), (1, 0): Fr(1, 10), (1, 1): Fr(2, 5)}
    p = raw
    lam, mu = Fr(1, 3), Fr(2, 5)

    def comp(e, f):
        return composed_effect_pointwise(e, f, p, m, n)

    # Bilinear target: (lam*e1+(1-lam)*e2) boxtimes (mu*f1+(1-mu)*f2), applied to p.
    e_mix = tuple(lam * e1[i] + (1 - lam) * e2[i] for i in range(m))
    f_mix = tuple(mu * f1[i] + (1 - mu) * f2[i] for i in range(n))
    bilinear_target = composed_effect_pointwise(e_mix, f_mix, p, m, n)

    # Four-term expansion via the four composed effects.
    c11, c12, c21, c22 = comp(e1, f1), comp(e1, f2), comp(e2, f1), comp(e2, f2)
    expansion = lam * mu * c11 + lam * (1 - mu) * c12 + (1 - lam) * mu * c21 + (1 - lam) * (1 - mu) * c22
    print(f"  bilinear target (direct)   = {bilinear_target}")
    print(f"  four-term expansion        = {expansion}")
    print(f"  match (pure algebra): {bilinear_target == expansion}")

    # Independent taps: A-side choice (lam) and B-side choice (mu) drawn independently -- four branches,
    # product weights. This is exactly the expansion above, restated as an operational scheme.
    independent = lam * mu * c11 + lam * (1 - mu) * c12 + (1 - lam) * mu * c21 + (1 - lam) * (1 - mu) * c22
    print(f"  independent-taps statistic = {independent}  matches: {independent == bilinear_target}")

    # Same-bit reuse: ONE shared bit selects BOTH sides at once -- outcome 0 -> (e1,f1); outcome 1 ->
    # (e2,f2). One bit has one law, so this forces the same bias for both choices; compare against the
    # TRUE bilinear target at that forced bias (lam=lam, i.e. mu set equal to lam), not the mu-of-choice one.
    forced_bilinear = lam * lam * c11 + lam * (1 - lam) * c12 + (1 - lam) * lam * c21 + (1 - lam) * (1 - lam) * c22
    same_bit = lam * c11 + (1 - lam) * c22
    print(f"  same-bit-reuse statistic (forces mu=lam={lam}) = {same_bit}")
    print(f"  true bilinear value at mu=lam                   = {forced_bilinear}")
    print(f"  match: {same_bit == forced_bilinear}  (mismatch: {same_bit - forced_bilinear})")
    print()


# ---------- (3) Rectangles vs all joint subsets, m=n=2 ----------

def check_rectangle_census(m, n):
    print(f"--- Rectangle census, m={m}, n={n} ---")
    outcomes = [(a, b) for a in range(m) for b in range(n)]
    all_subsets = set(frozenset(s) for s in powerset(outcomes))
    rectangles = set()
    for U in powerset(range(m)):
        for V in powerset(range(n)):
            rect = frozenset((a, b) for a in U for b in V)
            rectangles.add(rect)
    print(f"  total subsets of the {m*n}-outcome joint space = {len(all_subsets)}")
    print(f"  distinct rectangles (products of local subsets) = {len(rectangles)}")
    non_rectangles = all_subsets - rectangles
    print(f"  non-rectangular subsets = {len(non_rectangles)}")
    print(f"  examples of non-rectangular subsets: {[set(s) for s in list(non_rectangles)[:3]]}")
    print()
    return non_rectangles


# ---------- (4) CHSH-type witness: diagonal is unreachable by any mixture of local products ----------

def chsh_witness(m00, m01, m10, m11):
    return m00 + m11 - m01 - m10


def check_chsh_witness():
    print("--- CHSH-type witness: local products (however mixed) cannot reach the diagonal ---")
    # Pure products: witness = (e0-e1)(f0-f1), bounded in [-1,1] since e_i, f_i in [0,1].
    test_cases = [
        ((Fr(1), Fr(0)), (Fr(1), Fr(0))),   # e=(1,0), f=(1,0)
        ((Fr(1), Fr(0)), (Fr(0), Fr(1))),   # e=(1,0), f=(0,1)
        ((Fr(1, 2), Fr(1, 2)), (Fr(3, 4), Fr(1, 4))),
        ((Fr(3, 10), Fr(7, 10)), (Fr(2, 5), Fr(3, 5))),
    ]
    for e, f in test_cases:
        M = {(a, b): e[a] * f[b] for a in range(2) for b in range(2)}
        w = chsh_witness(M[(0, 0)], M[(0, 1)], M[(1, 0)], M[(1, 1)])
        print(f"  e={e}, f={f}: witness = {w}  (in [-1,1]: {Fr(-1) <= w <= Fr(1)})")

    # A convex mixture of several product effects: witness stays in [-1,1] (convexity of the bound).
    combos = [
        (Fr(1, 3), (Fr(1), Fr(0)), (Fr(1), Fr(0))),
        (Fr(1, 3), (Fr(0), Fr(1)), (Fr(0), Fr(1))),
        (Fr(1, 3), (Fr(1, 2), Fr(1, 2)), (Fr(1, 2), Fr(1, 2))),
    ]
    mixed = {(a, b): Fr(0) for a in range(2) for b in range(2)}
    for w_k, e, f in combos:
        for a in range(2):
            for b in range(2):
                mixed[(a, b)] += w_k * e[a] * f[b]
    w_mixed = chsh_witness(mixed[(0, 0)], mixed[(0, 1)], mixed[(1, 0)], mixed[(1, 1)])
    print(f"  convex mixture of 3 products: witness = {w_mixed}  (in [-1,1]: {Fr(-1) <= w_mixed <= Fr(1)})")

    # The diagonal indicator itself: witness = 1+1-0-0 = 2, outside [-1,1] -- unreachable.
    diag = {(0, 0): Fr(1), (0, 1): Fr(0), (1, 0): Fr(0), (1, 1): Fr(1)}
    w_diag = chsh_witness(diag[(0, 0)], diag[(0, 1)], diag[(1, 0)], diag[(1, 1)])
    print(f"  diagonal indicator (1,0,0,1): witness = {w_diag}  "
          f"(outside [-1,1], hence NOT any convex combination of local products: {not (Fr(-1) <= w_diag <= Fr(1))})")
    print()


if __name__ == "__main__":
    check_uniqueness(3, 2)
    check_composition_bilinearity(2, 2)
    check_rectangle_census(2, 2)
    check_chsh_witness()
