"""
Reproduces the CCU1-CCU6 results (CC-unification audit) of the States-Effects Tomography paper.

Checks, with exact rational arithmetic, whether a doubly-randomised statistic
    (lambda*e + (1-lambda)*f) applied to (mu*p + (1-mu)*q)
is reproduced by physically running two coin flips -- one selecting the preparation, one selecting the
effect -- in a single trial and averaging the reported outcome. Two schemes are compared:

  (A) INDEPENDENT taps: the prep-coin and the effect-coin are statistically independent (CC-IND holds).
      -> must reproduce the bilinear algebraic value exactly.
  (B) SAME-BIT reuse: prep-coin and effect-coin are the same physical bit (CC-IND fails).
      -> generically does NOT reproduce the bilinear value; this is the witness that CC-IND is a real,
         load-bearing clause of the CC postulate, not a redundant restatement.
"""

from fractions import Fraction as Fr


def affine_apply(v, x):
    """v, x: tuples of Fractions of the same length. Returns <v,x> = sum v_i x_i."""
    return sum(vi * xi for vi, xi in zip(v, x))


def mix(a, b, w):
    """w*a + (1-w)*b, componentwise."""
    return tuple(w * ai + (1 - w) * bi for ai, bi in zip(a, b))


def bilinear_value(e, f, lam, p, q, mu):
    """Direct algebraic evaluation: (lam*e + (1-lam)*f) applied to (mu*p + (1-mu)*q)."""
    mixed_effect = mix(e, f, lam)
    mixed_state = mix(p, q, mu)
    return affine_apply(mixed_effect, mixed_state)


def bilinear_expansion(e, f, lam, p, q, mu):
    """Four-term expansion: lam*mu*e(p) + lam*(1-mu)*e(q) + (1-lam)*mu*f(p) + (1-lam)*(1-mu)*f(q)."""
    ep, eq = affine_apply(e, p), affine_apply(e, q)
    fp, fq = affine_apply(f, p), affine_apply(f, q)
    return lam * mu * ep + lam * (1 - mu) * eq + (1 - lam) * mu * fp + (1 - lam) * (1 - mu) * fq


def independent_taps_statistic(e, f, lam, p, q, mu):
    """Two INDEPENDENT coins: prep-coin ~ Bernoulli(mu) picks p/q; effect-coin ~ Bernoulli(lam) picks e/f,
    drawn independently. Expected reported 'yes' = sum over the 4 branches of their joint (product)
    probability times that branch's outcome."""
    branches = [
        (lam * mu, affine_apply(e, p)),
        (lam * (1 - mu), affine_apply(e, q)),
        ((1 - lam) * mu, affine_apply(f, p)),
        ((1 - lam) * (1 - mu), affine_apply(f, q)),
    ]
    return sum(w * val for w, val in branches)


def same_bit_statistic(e, f, lam, p, q):
    """ONE shared bit ~ Bernoulli(lam): outcome 0 -> prepare p AND measure e; outcome 1 -> prepare q AND
    measure f. (mu is not free here -- reusing the same physical bit forces the same marginal law lam for
    both choices.) Expected reported 'yes' = lam*e(p) + (1-lam)*f(q)."""
    return lam * affine_apply(e, p) + (1 - lam) * affine_apply(f, q)


if __name__ == "__main__":
    # Generic (non-Dirac, non-matched) states and effects on S = {0,1}, so no accidental cancellation
    # hides the effect being tested.
    p = (Fr(7, 10), Fr(3, 10))
    q = (Fr(1, 5), Fr(4, 5))
    e = (Fr(9, 10), Fr(1, 10))
    f = (Fr(2, 5), Fr(3, 5))
    lam = Fr(1, 3)
    mu = Fr(1, 3)  # kept equal to lam so the same-bit scheme below is even definable

    direct = bilinear_value(e, f, lam, p, q, mu)
    expansion = bilinear_expansion(e, f, lam, p, q, mu)
    print(f"Bilinear algebraic value (direct)   = {direct}")
    print(f"Bilinear algebraic value (expanded) = {expansion}")
    print(f"Direct == expanded (pure algebra, no operational content): {direct == expansion}")
    print()

    indep = independent_taps_statistic(e, f, lam, p, q, mu)
    print(f"Independent-taps statistic (CC-IND holds)   = {indep}")
    print(f"Matches bilinear value exactly: {indep == direct}")
    print()

    same_bit = same_bit_statistic(e, f, lam, p, q)
    print(f"Same-bit-reuse statistic (CC-IND fails)     = {same_bit}")
    print(f"Matches bilinear value exactly: {same_bit == direct}")
    print(f"  (mismatch size: {same_bit - direct})")
    print()

    # Second example with mu != lam, to show independence also lets the two mixing weights differ freely
    # -- something the same-bit scheme cannot even express (one bit, one bias).
    mu2 = Fr(2, 5)
    direct2 = bilinear_value(e, f, lam, p, q, mu2)
    indep2 = independent_taps_statistic(e, f, lam, p, q, mu2)
    print(f"With independent taps and mu={mu2} != lam={lam}:")
    print(f"  bilinear value      = {direct2}")
    print(f"  independent-taps    = {indep2}")
    print(f"  match: {direct2 == indep2}")
