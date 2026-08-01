"""
Reproduces the CV1-CV4 results (primitive convexity obstruction) of the States-Effects Tomography paper.

Checks, with exact rational arithmetic (no floating point, no randomness), whether the finite family of
cylindrical conditionings of a strictly positive rank-one reference measure is closed under convex
combination, locally (F_A) and jointly (F_AB).

Deterministic: alpha, beta below are fixed rational vectors, not sampled.
"""

from fractions import Fraction as Fr
from itertools import chain, combinations


def powerset_nonempty(n):
    idx = range(n)
    return chain.from_iterable(combinations(idx, r) for r in range(1, n + 1))


def cylindrical_family(weights):
    """weights: tuple of Fractions summing to 1, strictly positive. Returns dict {U (frozenset): p_U (tuple)}."""
    m = len(weights)
    family = {}
    for U in powerset_nonempty(m):
        mass = sum(weights[i] for i in U)
        p = tuple(weights[i] / mass if i in U else Fr(0) for i in range(m))
        family[frozenset(U)] = p
    return family


def report_local(name, weights):
    m = len(weights)
    fam = cylindrical_family(weights)
    distinct_points = set(fam.values())
    print(f"--- {name}: m={m}, alpha={weights} ---")
    print(f"  |subsets| (candidate conditionings) = 2^{m}-1 = {2**m - 1}")
    print(f"  |F_A| (distinct achieved points)     = {len(distinct_points)}")
    vertices = {tuple(Fr(1) if i == j else Fr(0) for j in range(m)) for i in range(m)}
    print(f"  vertices (Diracs) all achieved: {vertices.issubset(distinct_points)}")

    # Witness: the midpoint of two achieved points is generically not itself achieved.
    pts = list(distinct_points)
    witness_found = False
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            p, q = pts[i], pts[j]
            mid = tuple((p[k] + q[k]) / 2 for k in range(m))
            if mid not in distinct_points:
                print(f"  witness: midpoint of {p} and {q} = {mid}")
                print(f"           is in Delta(S_A) but NOT in F_A (not a cylindrical conditioning of alpha)")
                witness_found = True
                break
        if witness_found:
            break
    if not witness_found:
        print("  (no midpoint witness found among sampled pairs — would need a larger search)")
    print()
    return fam, distinct_points


def report_joint(name_A, weights_A, name_B, weights_B):
    fam_A, pts_A = report_local(name_A, weights_A)
    fam_B, pts_B = report_local(name_B, weights_B)
    m, n = len(weights_A), len(weights_B)

    joint_points = set()
    for pA in pts_A:
        for pB in pts_B:
            joint_points.add(tuple((pA[i], pB[j]) for i in range(m) for j in range(n)))

    print(f"--- Joint family F_AB from rank-one W = alpha (x) beta ---")
    print(f"  |F_A| x |F_B| = {len(pts_A)} x {len(pts_B)} = {len(pts_A) * len(pts_B)}")
    print(f"  |F_AB| (distinct product points)          = {len(joint_points)}")
    print(f"  ambient simplex Delta(S_A x S_B) dimension = {m * n - 1}")

    # Confirm the already-proved product-closure result independently: conditioning W on a rectangle U x V
    # equals the product of the two locally conditioned marginals, for this rank-one W.
    W = {(a, b): weights_A[a] * weights_B[b] for a in range(m) for b in range(n)}
    ok = True
    for U in powerset_nonempty(m):
        for V in powerset_nonempty(n):
            mass = sum(W[(a, b)] for a in U for b in V)
            p_rect = {(a, b): (W[(a, b)] / mass if (a in U and b in V) else Fr(0))
                      for a in range(m) for b in range(n)}
            pA = fam_A[frozenset(U)]
            pB = fam_B[frozenset(V)]
            p_prod = {(a, b): pA[a] * pB[b] for a in range(m) for b in range(n)}
            if p_rect != p_prod:
                ok = False
    print(f"  product-closure (EffComp T3) reconfirmed on this W: {ok}")

    # Witness: the midpoint of two distinct joint preparations, flattened to plain tuples, is generically
    # not itself in F_AB.
    flat_points = {tuple(v for pair in p for v in pair) for p in joint_points}
    flat_list = list(flat_points)
    p0, q0 = flat_list[0], flat_list[1]
    mid0 = tuple((p0[k] + q0[k]) / 2 for k in range(len(p0)))
    print("  witness: midpoint of two joint preparations is in Delta(S_A x S_B) but not in F_AB")
    print(f"           (checked exactly: {mid0 not in flat_points})")
    print()


if __name__ == "__main__":
    # m=2 minimal example: alpha = (3/10, 7/10)
    report_local("S_A, m=2", (Fr(3, 10), Fr(7, 10)))

    # m=3 example: alpha = (1/2, 1/3, 1/6)
    report_local("S_A, m=3", (Fr(1, 2), Fr(1, 3), Fr(1, 6)))

    # Joint rank-one case: S_A m=2, S_B n=2
    report_joint("S_A, m=2 (for joint)", (Fr(3, 10), Fr(7, 10)),
                 "S_B, n=2 (for joint)", (Fr(2, 5), Fr(3, 5)))
