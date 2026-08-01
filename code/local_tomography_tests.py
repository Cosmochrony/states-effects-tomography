"""
Reproduces the LT1-LT6 results and corollary (local tomography) of the States-Effects Tomography paper.

Exact-arithmetic checks (Fractions, no randomness) for:
  (1) the specific result: full local coordinate families, composed via AX-COMP, reconstruct the joint
      state exactly (trivial once BC1 is known: (e_a boxtimes f_b)(p) = p_ab).
  (2) a COUNTEREXAMPLE to the naive "general formulation": minimal per-side AFFINE-separating families
      (EA2 sense, size m-1/n-1) can fail to separate the joint simplex via their products, even though
      each side separates its own simplex perfectly well.
  (3) the CORRECTED general sufficient condition: each side's family must LINEARLY SPAN its full local
      vector space (size >= m, >= n respectively), not merely affinely separate its own simplex.
  (4) the mn-1 redundancy: dropping any single one of the mn full-coordinate products still allows exact
      reconstruction, by the same normalisation argument as EA2, now at the joint level.
  (5) the conceptual point: a physically unavailable effect (e.g. the diagonal indicator) has a
      well-defined, computable expectation once the state is reconstructed by tomography -- physical
      availability of a joint effect and mathematical computability of its expectation are different things.
  (6) LT3's corollary: if the unit effect u=(1,...,1) lies in the linear span of a family that already
      separates Delta(S) (EA2 sense), the family necessarily SPANS R^S -- separation + unit effect is
      equivalent to spanning, not merely sufficient for it.
"""

from fractions import Fraction as Fr


def coord(a, m):
    return tuple(Fr(1) if i == a else Fr(0) for i in range(m))


def composed_value(e, f, p, m, n):
    return sum(p[(a, b)] * e[a] * f[b] for a in range(m) for b in range(n))


def random_looking_joint_state(m, n):
    raw = {(a, b): Fr((a + 1) * (b + 2) + (a * b) + 1, 1) for a in range(m) for b in range(n)}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


# ---------- (1) Full coordinate product reconstruction ----------

def check_full_reconstruction(m, n):
    print(f"--- (1) Full-coordinate product reconstruction, m={m}, n={n} ---")
    p = random_looking_joint_state(m, n)
    reconstructed = {}
    for a in range(m):
        for b in range(n):
            reconstructed[(a, b)] = composed_value(coord(a, m), coord(b, n), p, m, n)
    match = all(reconstructed[k] == p[k] for k in p)
    print(f"  p (target)        = {p}")
    print(f"  reconstructed     = {reconstructed}")
    print(f"  exact match: {match}")
    print()


# ---------- (2) Counterexample: minimal per-side families fail jointly ----------

def check_minimal_family_counterexample():
    print("--- (2) Counterexample: EA2-minimal per-side families, products do NOT separate the joint simplex ---")
    m, n = 2, 2
    # EA2-minimal separating families: drop coordinate 0 on each side, keep only {e_1}, {f_1}.
    e1, f1 = coord(1, 2), coord(1, 2)
    q = {(0, 0): Fr(1, 10), (0, 1): Fr(2, 10), (1, 0): Fr(3, 10), (1, 1): Fr(4, 10)}
    qp = {(0, 0): Fr(3, 10), (0, 1): Fr(1, 10), (1, 0): Fr(2, 10), (1, 1): Fr(4, 10)}
    assert sum(q.values()) == 1 and sum(qp.values()) == 1
    assert q != qp
    val_q = composed_value(e1, f1, q, m, n)
    val_qp = composed_value(e1, f1, qp, m, n)
    print(f"  q  = {q}")
    print(f"  q' = {qp}")
    print(f"  q != q': {q != qp}")
    print(f"  (e_1 boxtimes f_1)(q)  = {val_q}")
    print(f"  (e_1 boxtimes f_1)(q') = {val_qp}")
    print(f"  indistinguishable by the ONLY product in this minimal family: {val_q == val_qp}")
    print("  yet e_1 alone separates Delta(S_A) (m=2) and f_1 alone separates Delta(S_B) (n=2) -- EA2 holds")
    print("  on each side individually; the naive general claim ('local separation' => 'product separation')")
    print("  is FALSE as stated. Correction needed: see (3).")
    print()


# ---------- (3) Corrected sufficient condition: linear spanning, not mere affine separation ----------

def check_spanning_condition():
    print("--- (3) Corrected condition: full coordinate families SPAN (size m, n), and that suffices ---")
    m, n = 2, 2
    # Full families (size m=2, n=2) span R^2 trivially (standard basis) -- already shown to work in (1).
    # A non-coordinate but still SPANNING family of size 2: {e_0, e_0+e_1} (e_0+e_1 = constant-1 effect).
    e0 = coord(0, 2)
    e_sum = tuple(coord(0, 2)[i] + coord(1, 2)[i] for i in range(2))  # = (1,1), the unit effect
    f0 = coord(0, 2)
    f_sum = tuple(coord(0, 2)[i] + coord(1, 2)[i] for i in range(2))
    q = {(0, 0): Fr(1, 10), (0, 1): Fr(2, 10), (1, 0): Fr(3, 10), (1, 1): Fr(4, 10)}
    qp = {(0, 0): Fr(3, 10), (0, 1): Fr(1, 10), (1, 0): Fr(2, 10), (1, 1): Fr(4, 10)}
    family = [(e0, f0), (e0, f_sum), (e_sum, f0), (e_sum, f_sum)]
    values_q = [composed_value(e, f, q, m, n) for e, f in family]
    values_qp = [composed_value(e, f, qp, m, n) for e, f in family]
    print(f"  spanning family (non-coordinate basis {{e_0, e_0+e_1}} each side): 4 products")
    print(f"  values on q  = {values_q}")
    print(f"  values on q' = {values_qp}")
    print(f"  separates q from q': {values_q != values_qp}")
    print()


# ---------- (4) mn-1 redundancy, joint-level echo of EA2 ----------

def check_mn_minus_one_redundancy(m, n):
    print(f"--- (4) mn-1 redundancy: drop one product, reconstruction still exact, m={m}, n={n} ---")
    p = random_looking_joint_state(m, n)
    known = {}
    dropped = (m - 1, n - 1)
    for a in range(m):
        for b in range(n):
            if (a, b) == dropped:
                continue
            known[(a, b)] = composed_value(coord(a, m), coord(b, n), p, m, n)
    # Reconstruct the dropped coordinate from normalisation.
    reconstructed_dropped = Fr(1) - sum(known.values())
    print(f"  dropped product = e_{dropped[0]} boxtimes f_{dropped[1]}, true value = {p[dropped]}")
    print(f"  reconstructed from normalisation (1 - sum of the other {m*n-1}) = {reconstructed_dropped}")
    print(f"  match: {reconstructed_dropped == p[dropped]}")
    print()


# ---------- (5) Unavailable effect, computable expectation after reconstruction ----------

def check_unavailable_effect_expectation():
    print("--- (5) Diagonal indicator: physically unavailable, but its expectation is computable post-tomography ---")
    m, n = 2, 2
    p = random_looking_joint_state(m, n)
    # Reconstruct p exactly from the mn coordinate products (as in (1)).
    reconstructed = {(a, b): composed_value(coord(a, m), coord(b, n), p, m, n) for a in range(m) for b in range(n)}
    diag = {(0, 0): Fr(1), (0, 1): Fr(0), (1, 0): Fr(0), (1, 1): Fr(1)}  # NOT available via AX-COMP (BC4)
    true_expectation = sum(p[k] * diag[k] for k in p)
    computed_from_reconstruction = sum(reconstructed[k] * diag[k] for k in reconstructed)
    print(f"  true diagonal expectation E_p[1_diag]        = {true_expectation}")
    print(f"  computed from tomographically reconstructed p = {computed_from_reconstruction}")
    print(f"  match (computable despite physical unavailability): {true_expectation == computed_from_reconstruction}")
    print()


# ---------- (6) LT3 corollary: separation + unit effect <=> spanning ----------

def rank(vectors, dim):
    """Exact-arithmetic rank via Gaussian elimination over Fractions."""
    rows = [list(v) for v in vectors]
    r = 0
    for col in range(dim):
        pivot = None
        for i in range(r, len(rows)):
            if rows[i][col] != 0:
                pivot = i
                break
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        pivot_val = rows[r][col]
        rows[r] = [x / pivot_val for x in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][col] != 0:
                factor = rows[i][col]
                rows[i] = [rows[i][k] - factor * rows[r][k] for k in range(dim)]
        r += 1
        if r == len(rows):
            break
    return r


def in_span(target, vectors, dim):
    """Is `target` in the span of `vectors`? True iff adding it doesn't raise the rank."""
    return rank(vectors, dim) == rank(vectors + [target], dim)


def check_lt3_corollary(m):
    print(f"--- (6) LT3 corollary: separation + unit effect <=> spanning, m={m} ---")
    a0 = 0
    ea2_family = [coord(a, m) for a in range(m) if a != a0]  # m-1 elements, separates Delta(S) per EA2
    u = tuple(Fr(1) for _ in range(m))

    r_ea2 = rank(ea2_family, m)
    print(f"  EA2-minimal family ({m-1} coordinate effects, dropping index {a0}): rank = {r_ea2} (expect {m-1})")

    u_in_span = in_span(list(u), ea2_family, m)
    print(f"  unit effect u in span(EA2-minimal family)? {u_in_span} (expect False -- u has a nonzero "
          f"component at the dropped index, the family's vectors do not)")

    with_u = ea2_family + [list(u)]
    r_with_u = rank(with_u, m)
    print(f"  rank(EA2-minimal family union {{u}}) = {r_with_u} (expect {m}, i.e. full spanning)")

    # Contrast: the family alone (without u) has rank m-1 < m -- confirms LT2's failure mode is exactly
    # the missing dimension that u supplies.
    print(f"  spanning achieved only after adding u: {r_ea2 < m and r_with_u == m}")
    print()


if __name__ == "__main__":
    check_full_reconstruction(3, 2)
    check_minimal_family_counterexample()
    check_spanning_condition()
    check_mn_minus_one_redundancy(2, 2)
    check_unavailable_effect_expectation()
    check_lt3_corollary(3)
