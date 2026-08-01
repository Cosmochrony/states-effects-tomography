"""
Reproduces the EA1-EA5 results (mono-system effect classification) of the States-Effects Tomography paper.

Exact-arithmetic checks (Fractions, no randomness) for the affine-effect classification on Delta(S), |S|=3:
  (1) minimality of the coordinate-effect separating family: |S|-1 coordinates already separate all of
      Delta(S); |S|-2 do not (explicit counterexample pair);
  (2) the canonical coordinate effects sum to the unit effect (POVM completeness), with no extra postulate;
  (3) Boolean/coexistence closure (complement + disjoint sums of a single partition's effects) generates
      exactly the 2^|S| {0,1}-valued vertices of the effect cube [0,1]^S -- and no more;
  (4) effect-randomisation closure (convex hull of those vertices) reconstructs an ARBITRARY point of the
      full cube [0,1]^S via the explicit multilinear (independent-coin) formula, confirming conv(vertices)
      = the whole cube, not merely asserting it abstractly.
"""

from fractions import Fraction as Fr
from itertools import chain, combinations, product


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def coordinate_effect(a, m):
    return tuple(Fr(1) if i == a else Fr(0) for i in range(m))


def indicator(U, m):
    return tuple(Fr(1) if i in U else Fr(0) for i in range(m))


def check_minimality(m):
    print(f"--- Minimality of the coordinate-separating family, m={m} ---")
    # (1) dropping ONE coordinate still separates. Fix an arbitrary p, then ask: given a hypothetical q
    # that agrees with p on coordinates 0..m-2 (the kept ones) and lies in Delta(S), what must q's last
    # coordinate be? Normalisation forces it to equal p's -- so q = p necessarily, i.e. no two DISTINCT
    # points of Delta(S) can agree on all m-1 kept coordinates.
    head = [Fr(1, 2 * (m - 1))] * (m - 1)
    tail = Fr(1) - sum(head)
    p = tuple(head + [tail])
    q_last_forced = Fr(1) - sum(head)  # any q in Delta(S) agreeing with p on coords 0..m-2 has this last coord
    print(f"  p = {p}; any q in Delta(S) agreeing with p on the kept {m-1} coordinates has last "
          f"coordinate {q_last_forced} = p's last coordinate ({q_last_forced == p[-1]}) -- forced equal, "
          f"so {m-1} coordinates already separate.")

    # (2) dropping TWO coordinates (keeping only m-2) does NOT separate: exhibit two distinct points
    # agreeing on the first m-2 coordinates but differing on the remaining two.
    if m >= 3:
        shared = [Fr(1, 2 * m)] * (m - 2)
        rem = Fr(1) - sum(shared)
        pA = tuple(shared + [rem, Fr(0)])
        pB = tuple(shared + [Fr(0), rem])
        assert pA != pB
        agree = pA[:m - 2] == pB[:m - 2]
        print(f"  keeping only {m-2} coordinates: pA={pA}, pB={pB} agree on the first {m-2} "
              f"({agree}) yet pA != pB -- {m-2} coordinates do NOT separate.")
    print()


def check_unit_from_partition(m):
    print(f"--- Unit effect from the canonical partition, m={m} ---")
    coords = [coordinate_effect(a, m) for a in range(m)]
    total = tuple(sum(c[i] for c in coords) for i in range(m))
    print(f"  sum of the {m} coordinate effects = {total}  (unit effect u, obtained, not separately postulated)")
    print()


def check_boolean_closure(m):
    print(f"--- Boolean/coexistence closure, m={m} ---")
    subsets = list(powerset(range(m)))
    indicators = {indicator(U, m) for U in subsets}
    all_cube_vertices = {v for v in product([Fr(0), Fr(1)], repeat=m)}
    print(f"  |subsets| = 2^{m} = {len(subsets)}; |distinct indicator effects| = {len(indicators)}")
    print(f"  indicators == all {{0,1}}-vertices of the cube [0,1]^{m}? {indicators == all_cube_vertices}")
    # complement check: u - e_0 should equal the indicator of S \ {0}
    u = tuple(Fr(1) for _ in range(m))
    e0 = coordinate_effect(0, m)
    complement = tuple(u[i] - e0[i] for i in range(m))
    expected = indicator(set(range(m)) - {0}, m)
    print(f"  u - e_0 = {complement} == indicator(S\\{{0}}) = {expected}: {complement == expected}")
    print()
    return indicators


def check_randomisation_closure(m, x):
    print(f"--- Effect-randomisation closure reconstructs an arbitrary point of the cube, m={m} ---")
    print(f"  target x = {x} (NOT required to sum to 1 -- this is the cube, not the simplex)")
    total_weight = Fr(0)
    reconstruction = [Fr(0)] * m
    for U in powerset(range(m)):
        w = Fr(1)
        for a in range(m):
            w *= x[a] if a in U else (Fr(1) - x[a])
        total_weight += w
        ind = indicator(U, m)
        for i in range(m):
            reconstruction[i] += w * ind[i]
    reconstruction = tuple(reconstruction)
    print(f"  sum of weights over all 2^{m} vertices = {total_weight} (must be exactly 1)")
    print(f"  reconstructed point = {reconstruction}")
    print(f"  matches target exactly: {reconstruction == x}")
    print()


if __name__ == "__main__":
    m = 3
    check_minimality(m)
    check_unit_from_partition(m)
    check_boolean_closure(m)
    check_randomisation_closure(m, (Fr(1, 2), Fr(1, 3), Fr(3, 4)))
