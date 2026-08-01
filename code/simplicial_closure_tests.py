"""
Reproduces the CV5-CV7 results (simplicial closure under the P1 postulate) of the States-Effects
Tomography paper.

Under the classical-randomisation postulate P1, checks that:
  (1) F_A already contains every Dirac vertex of Delta(S_A) -- so conv(F_A) = Delta(S_A) exactly;
  (2) F_AB (product of independently, locally conditioned states) already contains every joint Dirac
      vertex of Delta(S_A x S_B) -- so conv(F_AB) = Delta(S_A x S_B) exactly;
  (3) (2) holds for a non-rank-one W too, not only the rank-one case -- confirming that rank(W) governs
      composition of the primitive family (Effective Composition's Theorem 3) but not the shape of the
      closed convex hull.

Deterministic: no randomness.
"""

from fractions import Fraction as Fr
from itertools import chain, combinations


def powerset_nonempty(n):
    idx = range(n)
    return chain.from_iterable(combinations(idx, r) for r in range(1, n + 1))


def dirac_vertices(n):
    return {tuple(Fr(1) if i == j else Fr(0) for j in range(n)) for i in range(n)}


def local_family(weights):
    m = len(weights)
    fam = {}
    for U in powerset_nonempty(m):
        mass = sum(weights[i] for i in U)
        fam[frozenset(U)] = tuple(weights[i] / mass if i in U else Fr(0) for i in range(m))
    return fam


def check_local_contains_vertices(name, weights):
    m = len(weights)
    fam = local_family(weights)
    pts = set(fam.values())
    verts = dirac_vertices(m)
    print(f"--- {name}: F_A contains all Dirac vertices? {verts.issubset(pts)} ---")
    return pts


def joint_product_family_from_marginals(weights_A, weights_B):
    """F_AB built as products of independently, locally conditioned marginals -- well defined for ANY W,
    rank one or not, since it only uses the A- and B-marginals of W."""
    fam_A = local_family(weights_A)
    fam_B = local_family(weights_B)
    m, n = len(weights_A), len(weights_B)
    joint = set()
    for pA in fam_A.values():
        for pB in fam_B.values():
            joint.add(tuple(pA[a] * pB[b] for a in range(m) for b in range(n)))
    return joint, m, n


def joint_dirac_vertices(m, n):
    return {tuple(Fr(1) if (a, b) == (a0, b0) else Fr(0) for a in range(m) for b in range(n))
            for a0 in range(m) for b0 in range(n)}


def check_joint_contains_vertices(label, weights_A, weights_B):
    joint, m, n = joint_product_family_from_marginals(weights_A, weights_B)
    verts = joint_dirac_vertices(m, n)
    print(f"--- {label}: F_AB (from marginals alpha={weights_A}, beta={weights_B}) ---")
    print(f"    contains all {m*n} joint Dirac vertices? {verts.issubset(joint)}")
    print(f"    |F_AB| = {len(joint)}  (ambient simplex dimension = {m*n - 1})")


if __name__ == "__main__":
    check_local_contains_vertices("S_A, m=2, alpha=(3/10,7/10)", (Fr(3, 10), Fr(7, 10)))
    check_local_contains_vertices("S_A, m=3, alpha=(1/2,1/3,1/6)", (Fr(1, 2), Fr(1, 3), Fr(1, 6)))

    # Rank-one W = alpha (x) beta: the case already governed by EffComp T3.
    check_joint_contains_vertices("rank-one W", (Fr(3, 10), Fr(7, 10)), (Fr(2, 5), Fr(3, 5)))

    # Non-rank-one (correlated) W: T3 does NOT give product closure of the raw family, but the marginals
    # alone -- used to build F_AB here -- still yield every joint Dirac via singleton x singleton
    # conditioning, independently of rank. This is the computational content of CV6's rank-independence.
    W_correlated = {(0, 0): Fr(1, 2), (0, 1): Fr(1, 10), (1, 0): Fr(1, 10), (1, 1): Fr(3, 10)}
    alpha_marg = (W_correlated[(0, 0)] + W_correlated[(0, 1)], W_correlated[(1, 0)] + W_correlated[(1, 1)])
    beta_marg = (W_correlated[(0, 0)] + W_correlated[(1, 0)], W_correlated[(0, 1)] + W_correlated[(1, 1)])
    # confirm this W is indeed not rank one: w_00 * w_11 != w_01 * w_10
    not_rank_one = W_correlated[(0, 0)] * W_correlated[(1, 1)] != W_correlated[(0, 1)] * W_correlated[(1, 0)]
    print(f"--- non-rank-one W: w00*w11 != w01*w10 ? {not_rank_one} ---")
    check_joint_contains_vertices("non-rank-one W (marginals only)", alpha_marg, beta_marg)
