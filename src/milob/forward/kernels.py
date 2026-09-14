"""
Geometry Green's functions for the diffusion and correlation-diffusion
operator.

Each kernel is generic in the wavenumber and knows nothing about which
technique produced it; technique-specific physics lives in
:mod:`milob.forward.dispersion`. New geometries are added here without
touching that module.
"""

import functools

import numpy as np
import scipy.integrate
import scipy.special
import mpmath


def Reff(n1, n2):
    """
    Compute the effective reflection coefficient at a refractive-index boundary.

    Accounts for the internally reflected fraction of diffuse light. Results
    are memoised, since a fit calls the forward model many times at a fixed
    refractive index.

    Parameters
    ----------
    n1 : float
        Refractive index inside the medium.
    n2 : float
        Refractive index outside.

    Returns
    -------
    float
        Effective reflectance. NaN when ``n2`` exceeds ``n1``, a known
        limitation.

    References
    ----------
    Haskell, R. C. et al. (1994). Journal of the Optical Society of America A,
    11(10), 2727-2741.
    """
    return _reff_cached(float(n1), float(n2))


@functools.lru_cache(maxsize=None)
def _reff_cached(n1, n2):
    def R_F(theta, n1, n2):
        theta_p = np.asin((n1 / n2) * np.sin(theta))
        frac_1 = 0.5 * ((n1 * np.cos(theta_p) - n2 * np.cos(theta)) / (n1 * np.cos(theta_p) + n2 * np.cos(theta))) ** 2
        frac_2 = 0.5 * ((n1 * np.cos(theta) - n2 * np.cos(theta_p)) / (n1 * np.cos(theta) + n2 * np.cos(theta_p))) ** 2
        return frac_1 + frac_2

    def R_phi(n1, n2):
        theta_c = np.asin(n2 / n1)

        def integrand_1(theta):
            return 2 * np.sin(theta) * np.cos(theta)

        def integrand_2(theta, n1, n2):
            return 2 * np.sin(theta) * (np.cos(theta)) * R_F(theta, n1, n2)

        return (scipy.integrate.quad(integrand_2, 0, theta_c, args=(n1, n2))[0]
                + scipy.integrate.quad(integrand_1, theta_c, np.pi / 2)[0])

    def R_J(n1, n2):
        theta_c = np.asin(n2 / n1)

        def integrand_1(theta):
            return 3 * np.sin(theta) * ((np.cos(theta)) ** 2)

        def integrand_2(theta, n1, n2):
            return 3 * np.sin(theta) * ((np.cos(theta)) ** 2) * R_F(theta, n1, n2)

        return (scipy.integrate.quad(integrand_2, 0, theta_c, args=(n1, n2))[0]
                + scipy.integrate.quad(integrand_1, theta_c, np.pi / 2)[0])

    return (R_phi(n1, n2) + R_J(n1, n2)) / (2 - R_phi(n1, n2) + R_J(n1, n2))


def si_kernel(rho, K, mua, musp, R_eff, z=0.0):
    """
    Semi-infinite Green's function, by the method of images.

    Solves the Helmholtz-form diffusion operator under the extrapolated-zero
    boundary condition. The wavenumber may be real, for continuous-wave work,
    or complex, for frequency-domain work.

    Parameters
    ----------
    rho : float or array-like
        Source-detector distance in cm.
    K : complex or array-like
        Wavenumber in cm^-1, the square root of the dispersion relation.
    mua : float
        Absorption coefficient in cm^-1.
    musp : float
        Reduced scattering coefficient in cm^-1.
    R_eff : float
        Effective reflectance at the medium-air boundary.
    z : float
        Detector depth in cm. Default 0, the boundary.

    Returns
    -------
    np.ndarray of complex
        Green's function for a unit source, broadcast over ``rho`` and ``K``.
    """
    mut = mua + musp
    z0 = 1.0 / mut
    zb = (2.0 / (3.0 * mut)) * (1.0 + R_eff) / (1.0 - R_eff)
    r1 = np.sqrt(rho**2 + (z - z0)**2)
    r2 = np.sqrt(rho**2 + (z + z0 + 2.0 * zb)**2)
    prefactor = 3.0 * mut / (4.0 * np.pi)
    return prefactor * (np.exp(-K * r1) / r1 - np.exp(-K * r2) / r2)


def two_layer_kernel(rho, z, K_sq, mua, musp, n, depth, R_eff, a=30.0, m=4000):
    """
    Two-layer Green's function for a finite layer over a semi-infinite one.

    Solved by a Fourier-Bessel transform in the radial coordinate, matched
    across the interface under the extrapolated-zero boundary condition. Takes
    the dispersion relation before its square root, since each transverse mode
    adds its own eigenvalue first.

    Only two layers are implemented; other lengths raise NotImplementedError.
    Uses mpmath at 20 decimal digits, since the series terms are unstable in
    double precision. Setting the precision changes process-global state.

    Parameters
    ----------
    rho : float
        Source-detector separation in cm.
    z : float
        Detector depth in cm, with 0 the boundary.
    K_sq : sequence of complex
        Dispersion relation per layer, top layer first, before the transverse
        eigenvalue is added.
    mua, musp : sequence of float
        Absorption and reduced scattering per layer, in cm^-1.
    n : sequence of float
        Refractive index per layer.
    depth : sequence of float
        Thickness in cm of each bounded layer; the last layer is
        semi-infinite.
    R_eff : sequence of float
        Effective reflectance at each boundary, the top surface first.
    a : float
        Radius in cm of the disk truncating the Fourier-Bessel series, which
        must be large relative to ``rho``.
    m : int
        Number of terms in the series.

    Returns
    -------
    complex
        Green's function for a unit source.

    Raises
    ------
    NotImplementedError
        If the number of layers is not two.

    References
    ----------
    Liemert, A., & Kienle, A. (2010). Optics Express, 18(9), 9266-9279.
    """
    mua = np.asarray(mua, dtype=float)
    musp = np.asarray(musp, dtype=float)
    n = np.asarray(n, dtype=float)
    R_eff = np.asarray(R_eff, dtype=float)
    depth = np.asarray(depth, dtype=float)
    K_sq = [complex(k) for k in K_sq]

    n_layers = len(mua)
    if n_layers != 2:
        raise NotImplementedError(
            "two_layer_kernel currently only implements the validated "
            "2-layer solution (Liemert & Kienle 2010); a general N-layer "
            f"recursive solver is future work. Got {n_layers} layers."
        )
    if not (len(musp) == len(n) == len(R_eff) == len(K_sq) == 2 and len(depth) == 1):
        raise ValueError(
            "mua, musp, n, R_eff, K_sq must all have length 2, and depth "
            "length 1, for the 2-layer solution."
        )

    mua_1, mua_2 = mua
    musp_1, musp_2 = musp
    n1, n2 = n
    R_eff_1, R_eff_2 = R_eff
    depth_1 = depth[0]
    K1_sq, K2_sq = K_sq

    mpmath.mp.dps = 20
    sinh = np.vectorize(mpmath.sinh)
    cosh = np.vectorize(mpmath.cosh)
    exp = np.vectorize(mpmath.exp)

    mut_1 = mua_1 + musp_1
    mut_2 = mua_2 + musp_2
    # 1/(3*mut) -- a length, NOT the true (v-carrying) diffusion coefficient;
    # kept as a separate name from D to avoid that confusion (the legacy
    # code called this "D_1"/"D_2"). K1_sq/K2_sq already carry the v-scaled
    # physics via dispersion.k2().
    ell_1 = 1.0 / (3.0 * mut_1)
    ell_2 = 1.0 / (3.0 * mut_2)
    zb_1 = 2.0 * ell_1 * (1.0 + R_eff_1) / (1.0 - R_eff_1)
    zb_2 = 2.0 * ell_2 * (1.0 + R_eff_2) / (1.0 - R_eff_2)
    z0_1 = 1.0 / mut_1

    l2 = 30.0 * depth_1
    aprime = a + zb_1
    s_ns = scipy.special.jn_zeros(0, m) / aprime

    n1_sq = n1**2
    n2_sq = n2**2

    def vG_1(bessel_zeros):
        K1 = np.sqrt(K1_sq + bessel_zeros**2)
        K2 = np.sqrt(K2_sq + bessel_zeros**2)

        beta_3 = sinh(K2 * (l2 + zb_2))
        gamma_3 = cosh(K2 * (l2 + zb_2))

        # Direct + image source term within layer 1 (homogeneous semi-infinite part)
        direct_term = (exp(-K1 * abs(z - z0_1)) - exp(-K1 * (z + z0_1 + 2 * zb_1))) / (2 * ell_1 * K1)

        # Correction from the layer 1 / layer 2 interface boundary conditions
        interface_prefactor = (sinh(K1 * (z0_1 + zb_1)) * sinh(K1 * (z + zb_1))) / (ell_1 * K1 * exp(K1 * (depth_1 + zb_1)))
        reflection_numerator = ell_1 * K1 * n1_sq * beta_3 - ell_2 * K2 * n2_sq * gamma_3
        reflection_denominator = (ell_1 * K1 * n1_sq * beta_3 * cosh(K1 * (depth_1 + zb_1))
                                   + ell_2 * K2 * n2_sq * gamma_3 * sinh(K1 * (depth_1 + zb_1)))

        g1 = direct_term + interface_prefactor * (reflection_numerator / reflection_denominator)
        return g1

    num1 = vG_1(s_ns)
    num2 = scipy.special.j0(s_ns * rho)
    den = (scipy.special.j1(s_ns * aprime))**2

    f = np.sum((num1 * num2) / den)
    f = f / (np.pi * (aprime**2))

    return f


def n_layer_kernel(rho, z, K_sq, mua, musp, n, depth, R_eff_top, R_eff_bottom=None,
                    s_max_factor=30.0, n_points=480):
    """
    General N-layer Green's function, with a finite or semi-infinite base.

    Uses a transfer-matrix recursion across interfaces combined with a
    Gauss-Legendre quadrature of the inverse Hankel transform. Independent of
    :func:`two_layer_kernel`, which solves the two-layer case by a different
    numerical route. Validated for one to four layers.

    A single layer covers both a semi-infinite medium and a finite slab, the
    latter through its own closed form.

    Uses mpmath at 20 decimal digits, since the hyperbolic terms overflow
    double precision for thick or absorbing layers.

    Parameters
    ----------
    rho : float
        Source-detector separation in cm.
    z : float
        Detector depth in cm, within the top layer. Only the boundary is
        validated.
    K_sq : sequence of complex
        Dispersion relation per layer, source layer first, before the
        transverse eigenvalue is added.
    mua, musp : sequence of float
        Absorption and reduced scattering per layer, in cm^-1.
    n : sequence of float
        Refractive index per layer.
    depth : sequence of float
        Thickness in cm of each bounded layer, top first. One shorter than the
        layer count when the base is semi-infinite, otherwise the same length.
    R_eff_top : float
        Effective reflectance at the top boundary.
    R_eff_bottom : float, optional
        Effective reflectance at the base. None (default) makes the base
        semi-infinite.
    s_max_factor : float
        Upper integration limit of the Hankel transform, as a multiple of the
        source layer's reduced scattering.
    n_points : int
        Number of quadrature points.

    Returns
    -------
    complex
        Green's function for a unit source.

    References
    ----------
    Liemert, A., & Kienle, A. (2010). Journal of Biomedical Optics, 15(2),
    025003.
    """
    mua = np.asarray(mua, dtype=float)
    musp = np.asarray(musp, dtype=float)
    n = np.asarray(n, dtype=float)
    depth = np.asarray(depth, dtype=float)
    K_sq = [complex(k) for k in K_sq]

    N = len(mua)
    if not (len(musp) == len(n) == len(K_sq) == N):
        raise ValueError("mua, musp, n, K_sq must all have the same length (n_layers).")

    finite_bottom = R_eff_bottom is not None
    expected_depth_len = N if finite_bottom else N - 1
    if len(depth) != expected_depth_len:
        raise ValueError(
            f"depth must have length {expected_depth_len} "
            f"({'finite' if finite_bottom else 'semi-infinite'} bottom layer, "
            f"n_layers={N}), got {len(depth)}."
        )
    mpmath.mp.dps = 20
    sinh = np.vectorize(mpmath.sinh)
    cosh = np.vectorize(mpmath.cosh)
    exp = np.vectorize(mpmath.exp)

    # 1-based padding: ell[k], n_[k], K_sq_[k], l[k] refer to layer/interface
    # k in the paper's own 1-based notation, avoiding off-by-one translation
    # errors against Eqs. (14)-(19).
    ell = [None] + [1.0 / (3.0 * (mua[k - 1] + musp[k - 1])) for k in range(1, N + 1)]
    n_ = [None] + list(n)
    K_sq_ = [None] + K_sq
    l = [None] + list(depth) + ([] if finite_bottom else [None])

    def alpha(k, s):
        return np.sqrt(K_sq_[k] + s**2)

    z0 = 1.0 / (mua[0] + musp[0])
    zb_top = 2.0 * ell[1] * (1.0 + R_eff_top) / (1.0 - R_eff_top)
    if finite_bottom:
        zb_bottom = 2.0 * ell[N] * (1.0 + R_eff_bottom) / (1.0 - R_eff_bottom)

    s_max = s_max_factor * musp[0]
    x, w = np.polynomial.legendre.leggauss(n_points)
    s = 0.5 * s_max * (x + 1.0)
    ws = 0.5 * s_max * w

    if N == 1:
        a1 = alpha(1, s)
        if finite_bottom:
            # Closed-form 1-D Green's function for a single finite slab
            # bounded by extrapolated-zero conditions on both sides (not in
            # Liemert & Kienle -- their N-layer recursion assumes N>=2, a
            # layer 2 to recurse into). Standard Sturm-Liouville construction
            # for y'' - alpha^2*y = -delta(z-z0)/ell with Dirichlet BCs at
            # z=-zb_top and z=l1+zb_bottom: y1=sinh(alpha*(z+zb_top)) satisfies
            # the top BC, y2=sinh(alpha*(l1+zb_bottom-z)) satisfies the bottom
            # BC, and matching continuity + the flux jump at z0 gives
            # phi = y1(z_min)*y2(z_max) / (ell*alpha*W) where the Wronskian
            # W = y1*y2' - y1'*y2 = -alpha*sinh(alpha*(l1+zb_top+zb_bottom))
            # (constant, since y1/y2 solve the same homogeneous ODE) --
            # verified against the ODE, both boundary conditions, and the
            # flux-jump condition directly (see validation notes). Already a
            # single ratio of comparably-scaled sinh terms (never a
            # difference of near-equal large quantities), so -- unlike the
            # N>=2 outer combination -- this needs no further algebraic
            # rewrite for numerical stability.
            l1 = l[1]
            z_min, z_max = min(z, z0), max(z, z0)
            num = sinh(a1 * (z_min + zb_top)) * sinh(a1 * (l1 + zb_bottom - z_max))
            den = ell[1] * a1 * sinh(a1 * (l1 + zb_top + zb_bottom))
            phi_s = num / den
        else:
            phi_s = (exp(-a1 * abs(z - z0)) - exp(-a1 * (z + z0 + 2.0 * zb_top))) / (2.0 * ell[1] * a1)
    else:
        if N == 2:
            if finite_bottom:
                a2 = alpha(2, s)
                arg = a2 * (l[2] + zb_bottom)
                beta, gamma = sinh(arg), cosh(arg)
            else:
                beta, gamma = 1.0, 1.0
        else:
            a_Nm1 = alpha(N - 1, s)
            a_N = alpha(N, s)
            ratio_sq = (n_[N] / n_[N - 1])**2
            if finite_bottom:
                arg = a_N * (l[N] + zb_bottom)
                beta = (ell[N - 1] * a_Nm1 * sinh(arg) * cosh(a_Nm1 * l[N - 1])
                        + ell[N] * a_N * ratio_sq * cosh(arg) * sinh(a_Nm1 * l[N - 1]))
                gamma = (ell[N - 1] * a_Nm1 * sinh(arg) * sinh(a_Nm1 * l[N - 1])
                         + ell[N] * a_N * ratio_sq * cosh(arg) * cosh(a_Nm1 * l[N - 1]))
            else:
                beta = ell[N - 1] * a_Nm1 * cosh(a_Nm1 * l[N - 1]) + ell[N] * a_N * ratio_sq * sinh(a_Nm1 * l[N - 1])
                gamma = ell[N - 1] * a_Nm1 * sinh(a_Nm1 * l[N - 1]) + ell[N] * a_N * ratio_sq * cosh(a_Nm1 * l[N - 1])

            # Eq. (18): recurse k = N, N-1, ..., 4 down to beta_3/gamma_3.
            # No-op when N == 3 (beta/gamma above already IS beta_3/gamma_3).
            for k in range(N, 3, -1):
                a_km2 = alpha(k - 2, s)
                a_km1 = alpha(k - 1, s)
                ratio_sq_k = (n_[k - 1] / n_[k - 2])**2
                new_beta = (ell[k - 2] * a_km2 * cosh(a_km2 * l[k - 2]) * beta
                            + ell[k - 1] * a_km1 * ratio_sq_k * sinh(a_km2 * l[k - 2]) * gamma)
                new_gamma = (ell[k - 2] * a_km2 * sinh(a_km2 * l[k - 2]) * beta
                             + ell[k - 1] * a_km1 * ratio_sq_k * cosh(a_km2 * l[k - 2]) * gamma)
                beta, gamma = new_beta, new_gamma

        a1 = alpha(1, s)
        a2 = alpha(2, s)
        ratio_sq_12 = (n_[2] / n_[1])**2
        l1 = l[1]

        # Eq. (14)/(16) rewritten as direct_term + interface_prefactor * ratio
        # (the same numerically stable decomposition used in ``two_layer_kernel``,
        # generalised to whatever beta/gamma the branch above produced) rather
        # than evaluated literally as printed. The literal form subtracts two
        # sinh(...) terms that are individually within a constant factor of
        # each other for alpha_1*(l1+zb1) >> 1 -- exactly the regime the paper
        # itself flags in Appendix B as needing special handling to "avoid
        # numerical errors due to the inverse Fourier transform". Rewriting
        # cosh/sinh in exponential form and factoring out the common
        # exp[alpha_1*(l1+zb_top)] growth analytically (rather than relying on
        # floating/arbitrary-precision subtraction to cancel it after the
        # fact) removes that cancellation entirely: every term below is a
        # decaying exponential or a ratio of comparably-scaled quantities.
        A = ell[1] * a1 * beta
        B = ell[2] * a2 * ratio_sq_12 * gamma
        Nd = A * cosh(a1 * (l1 + zb_top)) + B * sinh(a1 * (l1 + zb_top))
        direct_term = (exp(-a1 * abs(z - z0)) - exp(-a1 * (z + z0 + 2.0 * zb_top))) / (2.0 * ell[1] * a1)
        interface_prefactor = (sinh(a1 * (z0 + zb_top)) * sinh(a1 * (z + zb_top))
                                / (ell[1] * a1 * exp(a1 * (l1 + zb_top))))
        phi_s = direct_term + interface_prefactor * ((A - B) / Nd)

    phi_s = np.array([complex(v) for v in phi_s])
    integrand = phi_s * s * scipy.special.j0(s * rho)
    return complex(np.sum(ws * integrand) / (2.0 * np.pi))


