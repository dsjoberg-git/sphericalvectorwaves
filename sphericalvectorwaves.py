# Implementation of spherical vector waves and various routines. Based
# on the definitions in Hansen (1988) "Spherical near-field antenna
# measurements".
#
# Evaluation at the poles theta=0 and pi is handled by approximating
# theta=tol and theta=pi-tol for tol=1e-7.
#
# Daniel Sjöberg, 2024-05-04

import numpy as np
from scipy.special import spherical_jn, spherical_yn, factorial, lpmn, lpmv
from scipy.special import factorial, binom
from scipy.special import jacobi
import wigners, pyshtools
import warnings

# Some spherical bessel functions not defined in scipy
def spherical_h1(n, z, derivative=False):
    return spherical_jn(n, z, derivative) + 1j*spherical_yn(n, z, derivative)

def spherical_h2(n, z, derivative=False):
    return spherical_jn(n, z, derivative) - 1j*spherical_yn(n, z, derivative)

# Hansen's single index function, take into account python starts counting at 0
def j_from_smn(s, m, n):
    jj = 2*(n*(n + 1) + m - 1) + s 
    return int(jj - 1)

def smn_from_j(j):
    jj = j + 1
    if np.mod(jj, 2) == 1: # jj is odd
        s = 1
    else:
        s = 2
    n = np.floor(np.sqrt((jj - s)/2 + 1))
    m = (jj - s)/2 + 1 - n*(n + 1)
    return int(s), int(m), int(n)

# Functions of z = kr
def znc(n, c, z, derivative=False):
    if c == 1:
        z_func = spherical_jn(n, z, derivative)
    elif c == 2:
        z_func = spherical_yn(n, z, derivative)
    elif c == 3:
        z_func = spherical_h1(n, z, derivative)
    else:
        z_func = spherical_h2(n, z, derivative)
    return z_func

def Rsnc(s, n, c, z):
    if s == 1:
        R_func = znc(n, c, z)
    else:
        R_func = znc(n, c, z, derivative=True) + znc(n, c, z)/z
    return R_func

# Functions of x = cos(theta)
def mylpmv(m, n, x, derivative=False, tol=1e-20):
    """Scale the scipy implementation to the one used in Hansen. The scipy.special.lpmv implementation is fast, but gives NaN for n >= 86. Note that it contains the factor (-1)**m, which needs to be compensated for."""
    if derivative == False:
        y = (-1)**m*lpmv(m, n, x)
    else:
        if m == 0:
            y = -1/np.sqrt(1 - x**2)*lpmv(1, n, x)
        else:
            y = (n + m)*(n - m + 1)/np.sqrt(1 - x**2)*lpmv(m - 1, n, x) + m*x/(1 - x**2)*lpmv(m, n ,x)
        y = (-1)**m*y
    return y
    
def mylpmv_pyshtools(m, n, x, derivative=False):
    """Use pyshtools for the Legendre functions, since this package can compute them to high order."""
    normalization = '4pi' # '4pi' seems stable for high orders
    csphase = 1           # Do not include Condon-Shortley phase (-1)**m
    cnorm = 1             # Normalize each complex mode equally (no special treatment of m=0)
    if derivative == False:
        y = pyshtools.legendre.legendre_lm(n, m, x, normalization=normalization, csphase=csphase, cnorm=cnorm)/np.sqrt(2)
    else:
        if m == 0:
            y = np.sqrt(n*(n + 1))/np.sqrt(1 - x**2)*pyshtools.legendre.legendre_lm(n, 1, x, normalization=normalization, csphase=csphase, cnorm=cnorm)
        else:
            Pnm = pyshtools.legendre.legendre_lm(n, m, x, normalization=normalization, csphase=csphase, cnorm=cnorm)
            Pnm_m1 = pyshtools.legendre.legendre_lm(n, m - 1, x, normalization=normalization, csphase=csphase, cnorm=cnorm)
            y = -np.sqrt((n + m)*(n - m + 1))/np.sqrt(1 - x**2)*Pnm_m1 + m*x/(1 - x**2)*Pnm
        y = y/np.sqrt(2)
    return y
    
def lpmn_norm(m, n, x, derivative=False):
    """Normalized associated Legendre function."""
    mm = np.abs(m) # Should only be called with non-negative m
    if n >= 86:
        y = mylpmv_pyshtools(mm, n, x, derivative=derivative)
    else:
        prefactor = np.sqrt((2*n + 1)/2*factorial(n - mm)/factorial(n + mm))
        y = prefactor*mylpmv(mm, n, x, derivative=derivative)
    return y

# Spherical vector waves
def Fsmnc(s, m, n, c, kr, theta, phi, tol=1e-7):
    if np.isscalar(theta):
        if np.abs(theta) < tol:
            theta = tol
        elif np.abs(theta - np.pi) < tol:
            theta = np.pi - tol
    else:
        theta[np.abs(theta) < tol] = tol
        theta[np.abs(theta - np.pi) < tol] = np.pi - tol
    x = np.cos(theta)
    y = np.sin(theta)
    if m == 0:
        prefactor = 1/np.sqrt(2*np.pi*n*(n + 1))
    else:
        prefactor = 1/np.sqrt(2*np.pi*n*(n + 1))*(-m/np.abs(m))**m
    if s == 1:
        F_theta = prefactor*Rsnc(1, n, c, kr)*1j*m*lpmn_norm(m, n, x)/y*np.exp(1j*m*phi)
        F_phi = -prefactor*Rsnc(1, n, c, kr)*(-y)*lpmn_norm(m, n, x, derivative=True)*np.exp(1j*m*phi)
        F_r = 0*F_phi # Write like this so F_r.shape == F_phi.shape
    elif s == 2:
        F_r = prefactor*n*(n + 1)/kr*znc(n, c, kr)*lpmn_norm(m, n, x)*np.exp(1j*m*phi)
        F_theta = prefactor*Rsnc(2, n, c, kr)*(-y)*lpmn_norm(m, n, x, derivative=True)*np.exp(1j*m*phi)
        F_phi = prefactor*Rsnc(2, n, c, kr)*1j*m*lpmn_norm(m, n, x)/y*np.exp(1j*m*phi)
    else: # Check normalization w/o sqrt(n(n+1))
        prefactor = 1/np.sqrt(2*np.pi)*(-m/np.abs(m))**m
        F_r = prefactor*znc(n, c, kr, derivative=True)*lpmn_norm(m, n, x)*np.exp(1j*m*phi)
        F_theta = prefactor*znc(n, c, kr)/kr*(-y)*lpmn_norm(m, n, x, derivative=True)*np.exp(1j*m*phi)
        F_phi = prefactor*znc(n, c, kr)*1j*m*lpmn_norm(m, n, x)/(kr*y)*np.exp(1j*m*phi)
    return F_r, F_theta, F_phi

# Far field angular functions
def Ksmn(s, m, n, theta, phi, tol=1e-7):
    if np.isscalar(theta):
        if np.abs(theta) < tol:
            theta = tol
        elif np.abs(theta - np.pi) < tol:
            theta = np.pi - tol
    else:
        theta[np.abs(theta) < tol] = tol
        theta[np.abs(theta - np.pi) < tol] = np.pi - tol
    x = np.cos(theta)
    y = np.sin(theta)
    if s == 1:
        if m == 0:
            prefactor = np.sqrt(2/(n*(n + 1)))*(-1j)**(n + 1)
        else:
            prefactor = np.sqrt(2/(n*(n + 1)))*(-m/np.abs(m))**m*np.exp(1j*m*phi)*(-1j)**(n + 1)
        Ktheta = prefactor*1j*m*lpmn_norm(m, n, x)/y
        Kphi = -prefactor*(-y)*lpmn_norm(m, n, x, derivative=True)
    else:
        if m == 0:
            prefactor = np.sqrt(2/(n*(n + 1)))*(-1j)**n
        else:
            prefactor = np.sqrt(2/(n*(n + 1)))*(-m/np.abs(m))**m*np.exp(1j*m*phi)*(-1j)**n
        Ktheta = prefactor*(-y)*lpmn_norm(m, n, x, derivative=True)
        Kphi = prefactor*1j*m*lpmn_norm(m, n, x)/y
    return Ktheta, Kphi

# Plane wave expansion coefficients
def PlaneWave(E0, theta0, phi0, N):
    """Generate all expansion coefficients for a plane wave up to order N."""
    J = 2*N*(N + 2)
    Q = np.zeros(J, dtype=complex)
    Etheta = E0[0]*np.cos(theta0)*np.cos(phi0) + E0[1]*np.cos(theta0)*np.sin(phi0) - E0[2]*np.sin(theta0)
    Ephi = -E0[0]*np.sin(phi0) + E0[1]*np.cos(phi0)
    for j in range(J): # Note that range(J) = 0, 1, 2, ..., J-1
        s, m, n = smn_from_j(j)
        Ktheta, Kphi = Ksmn(s, -m, n, theta0, phi0)
        Q[j] = (-1)**m*np.sqrt(4*np.pi)*1j*(Etheta*Ktheta + Ephi*Kphi)
    return Q

# Compute the regular field at a point given the expansion coefficients
def FieldValue(Q, kr, theta, phi, c=1):
    J = len(Q)
    Er, Etheta, Ephi = 0j, 0j, 0j
    for j in range(J):  # Note that range(J) = 0, 1, 2, ..., J-1
        s, m, n = smn_from_j(j)
        Fr, Ftheta, Fphi = Fsmnc(s, m, n, c, kr, theta, phi)
        Er = Er + Q[j]*Fr
        Etheta = Etheta + Q[j]*Ftheta
        Ephi = Ephi + Q[j]*Fphi
    return Er, Etheta, Ephi

# Functions for rotation and translation
def dnmum(n, mu, m, theta):
    """Rotation coefficients, by eqn (A2.5) in Hansen. Some checks necessary to assure the arguments for the jacobi polynomial are correct; if not, use symmetries (A2.7), (A2.8), (A2.9) in Hansen."""
    d_func = lambda _n, _mu, _m, _theta: np.sqrt(factorial(_n + _mu) * factorial(_n - _mu) / factorial(_n + _m) / factorial(_n - _m))*np.cos(_theta/2)**(_mu + _m) * np.sin(_theta/2)**(_mu - _m) * jacobi(_n - _mu, _mu - _m, _mu + _m)(np.cos(_theta))
    if mu - m > -1 and mu + m > -1:
        d = d_func(n, mu, m, theta)
    elif mu - m <= -1 and mu + m > -1:
        d = d_func(n, m, mu, theta)*(-1)**(mu + m)
    elif mu - m > -1 and mu + m <= -1:
        d = d_func(n, -m, -mu, -theta)*(-1)**(mu + m)
    else: # mu - m <= -1 and mu + m <= -1
        d = d_func(n, -mu, -m, theta)*(-1)**(mu + m)
    return d
    
def dnmum_old(n, mu, m, theta):
    """Rotation coefficients, by eqn (A2.3) in Hansen."""
    factor = np.sqrt(factorial(n + mu) * factorial(n - mu) / factorial(n + m) / factorial(n - m))
    sigma_min = np.max([-m - mu, 0])
    sigma_max = np.min([n - mu, n - m])
    d = 0
    for sigma in range(sigma_min, sigma_max + 1):
        d = d + binom(n + m, n - mu - sigma) * binom(n - m, sigma) * (-1)**(n - mu -sigma) * np.cos(theta/2)**(2*sigma + mu + m) * np.sin(theta/2)**(2*n - 2*sigma - mu - m)
    d = factor*d
    return d

def wigner3j(j1, j2, j3, m1, m2, m3):
    return wigners.wigner_3j(j1, j2, j3, m1, m2, m3)
    
def wigner3j_old(j1, j2, j3, m1, m2, m3):
    """Compute the Wigner 3j-symbols."""
    if m1 + m2 + m3 == 0:
        K = np.max([0, j2 - j3 - m1, j1 - j3 + m2])
        N = np.min([j1 + j2 - j3, j1 - m1, j2 + m2])
        factor = (-1)**(j1 - j2 - m3) * np.sqrt(factorial(j1 + j2 - j3) * factorial(j1 - j2 + j3) * factorial(-j1 + j2 + j3) / factorial(j1 + j2 + j3 + 1)) * np.sqrt(factorial(j1 - m1) * factorial(j1 + m1) * factorial(j2 - m2) * factorial(j2 + m2) * factorial(j3 - m3) * factorial(j3 + m3))
        w = 0
        for k in range(K, N + 1):
            w = w + (-1)**k / factorial(k) / factorial(j1 + j2 - j3 - k) / factorial(j1 - m1 - k) / factorial(j2 + m2 - k) / factorial(j3 - j2 + m1 + k) / factorial(j3 - j1 - m2 + k)
        w = factor*w
    else:
        w = 0
    return w

def a_linearization(mu, n, nu, p):
    """Linearization coefficients."""
    a = (2*p + 1) * np.sqrt(factorial(n + mu) * factorial(nu - mu) / factorial(n - mu) / factorial(nu + mu)) * wigner3j(n, nu, p, 0, 0, 0) * wigner3j(n, nu, p, mu, -mu, 0)
    return a

def Csncsigmamunu(s, n, c, sigma, mu, nu, kA, asymptotic=False):
    """Translation coefficients."""
    if not asymptotic:
        factor = np.sqrt((2*n + 1) * (2*nu + 1) / n / (n + 1) / nu / (nu + 1)) * np.sqrt(factorial(nu + mu) * factorial(n - mu) / factorial(nu - mu) / factorial(n + mu)) * (-1)**mu * 0.5 * 1j**(n - nu)
        if s == sigma:
            delta_s_sigma = 1
        else:
            delta_s_sigma = 0
        if 3 - s == sigma:
            delta_3ms_sigma = 1
        else:
            delta_3ms_sigma = 0
        C = 0
        for p in range(np.abs(n - nu), n + nu + 1):
            C = C + 1j**(-p) * (delta_s_sigma*(n*(n + 1) + nu*(nu + 1) - p*(p + 1)) + delta_3ms_sigma*(2j*mu*kA)) * a_linearization(mu, n, nu, p) * znc(p, c, kA)
        C = factor*C
    else:
        if c == 1:
            func = np.cos(kA)/kA
        elif c == 2:
            func = np.sin(kA)/kA
        elif c == 3:
            func = np.exp(1j*kA)/kA
        else:
            func = np.exp(-1j*kA)/kA
        if mu == 1:
            C = np.sqrt((2*n + 1) * (2*nu + 1)) / 2 * 1j**(nu - n- 1) * func
        elif mu == -1:
            C = np.sqrt((2*n + 1) * (2*nu + 1)) / 2 * 1j**(nu - n- 1) * (-1)**(2 + sigma) * func
        else:
            C = 0*kA
    return C

def RotateCoefficients(Q, phi0, theta0, chi0):
    """Compute new expansion coefficients in a rotated frame (x', y', z').

The original coordinate system is (x, y, z), the final is (x′ , y ′ , z ′ ). Three consecutive rotations (Euler angles) are described as follows, where (x1 , y1 , z1 ) is the coordinate system after the first rotation, and (x2 , y2 , z2 ) is the coordinate system after the second rotation.

1. A rotation about the z-axis (through an angle phi0).
2. A rotation about the y1-axis (through an angle theta0).
3. A rotation about the z2-axis (through an angle chi0)."""
    J = len(Q)
    Qnew = np.zeros(Q.shape, dtype=complex)
    for j in range(J): # Note that range(J) = 0, 1, 2, ..., J-1
        s, mu, n = smn_from_j(j)
        for m in range(-n, n + 1):
            jold = j_from_smn(s, m, n)
            Qnew[j] = Qnew[j] + Q[jold] * np.exp(1j*m*phi0) * dnmum(n, mu, m, theta0) * np.exp(1j*mu*chi0)
    return Qnew

def TranslateCoefficients(Q, kA, c=3):
    """Compute new expansion coefficients in a frame translated kA in the positive direction of the z-axis."""
    J = len(Q)
    S, M, N = smn_from_j(J - 1)
    Nnew = N + int(kA) + 1 # Estimate of required Nnew
    Jnew = 2*Nnew*(Nnew + 2)
    Qnew = np.zeros(Jnew, dtype=complex)
    for jnew in range(0, Jnew):
        sigma, mu, nu = smn_from_j(jnew)
        for s in [1, 2]:
            for n in range(np.max([1, np.abs(mu)]), N + 1):
                jold = j_from_smn(s, mu, n)
                Qnew[jnew] = Qnew[jnew] + Q[jold]*Csncsigmamunu(s, n, c, sigma, mu, nu, kA)
    return Qnew
        
if __name__ == '__main__':
    if True: 
        print('Test of orthogonality relations')
        Ntheta = 100
        Nphi = 200
        thetavec = np.linspace(0, np.pi, Ntheta, endpoint=True)
        phivec = np.linspace(0, 2*np.pi, Nphi, endpoint=False)
        dtheta = thetavec[1] - thetavec[0]
        dphi = phivec[1] - phivec[0]
        theta, phi = np.meshgrid(thetavec, phivec, indexing='ij')
    
        kr = 1.5

        s1 = 2
        m1 = -2
        n1 = 5
        c1 = 1

        s2 = s1
        m2 = -m1
        n2 = n1
        c2 = 2

        F1_r, F1_theta, F1_phi = Fsmnc(s1, m1, n1, c1, kr, theta, phi)
        F2_r, F2_theta, F2_phi = Fsmnc(s2, m2, n2, c2, kr, theta, phi)

        np.set_printoptions(precision=2, suppress=True)

        print('\nThe following numbers should be = 1')
        print(np.sum(F1_r*F2_r*np.sin(theta)*dtheta*dphi)/((-1)**m1*n1*(n1 + 1)*znc(n1, c1, kr)*znc(n2, c2, kr)/kr**2))
    
        print(np.sum((F1_theta*F2_theta + F1_phi*F2_phi)*np.sin(theta)*dtheta*dphi)/((-1)**m1*Rsnc(s1, n1, c1, kr)*Rsnc(s2, n2, c2, kr)))
    
        print('\nTesting reciprocity integral')
        cvec = [1, 2, 3, 4]
        A = np.zeros((4, 4), dtype=complex)
        for m, c1 in enumerate(cvec):
            for n, c2 in enumerate(cvec):
                E1_r, E1_theta, E1_phi = Fsmnc(s1, m1, n1, c1, kr, theta, phi)
                H1_r, H1_theta, H1_phi = Fsmnc(3-s1, m1, n1, c1, kr, theta, phi)
                E2_r, E2_theta, E2_phi = Fsmnc(s2, m2, n2, c2, kr, theta, phi)
                H2_r, H2_theta, H2_phi = Fsmnc(3-s2, m2, n2, c2, kr, theta, phi)
                A[m,n] = -1j*kr**2*np.sum((E1_theta*H2_phi - E1_phi*H2_theta - E2_theta*H1_phi + E2_phi*H1_theta)*np.sin(theta)*dtheta*dphi)/(-1j*(-1)**m1)
        print('Numerical result:')
        print(A)
        print('Should be equal to analytical result:')
        print(np.array([[0, 1, 1j, -1j],
                        [-1, 0, -1, -1],
                        [-1j, 1, 0, -2j],
                        [1j,1,2j,0]]), '\n')

    if True:
        print('Testing rotation and translation')
        
        from scipy.spatial.transform import Rotation as R

        def CartesianCoordFromSphericalCoord(r, theta, phi):
            x = r*np.sin(theta)*np.cos(phi)
            y = r*np.sin(theta)*np.sin(phi)
            z = r*np.cos(theta)
            return x, y, z
        def SphericalCoordFromCartesianCoord(x, y=None, z=None):
            if y == None: # Enable calling this function with one argument x = [x, y, z]; ugly but it works
                x, y, z = x[0], x[1], x[2]
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arccos(z/r)
            phi = np.arctan2(y, x)
            return r, theta, phi
        def CartesianVecFromSphericalVec(Er, Etheta, Ephi, theta, phi):
            Ex = Er*np.sin(theta)*np.cos(phi) + Etheta*np.cos(theta)*np.cos(phi) + Ephi*(-np.sin(phi))
            Ey = Er*np.sin(theta)*np.sin(phi) + Etheta*np.cos(theta)*np.sin(phi) + Ephi*np.cos(phi)
            Ez = Er*np.cos(theta) + Etheta*(-np.sin(theta))
            return Ex, Ey, Ez
        
        Ei = np.array([0, 1, 0])  # Plane wave cartesian polarization
        thetai = np.pi/2          # Propagation direction theta
        phii = 0                  # Propagation direction phi

        rotphi1 = np.pi/3         # First set of rotations: around z
        rottheta1 = np.pi/4       # First set of rotations: around y
        rotchi1 = np.pi/5         # First set of rotations: around z
        kA = 2                    # Translation along z axis
        rotphi2 = np.pi/6         # Second set of rotations: around z
        rottheta2 = np.pi/7       # Second set of rotations: around y
        rotchi2 = np.pi/8         # Second set of rotations: around z
        r1 = R.from_euler('zyz', [rotchi1, rottheta1, rotphi1])
        r2 = R.from_euler('zyz', [rotchi2, rottheta2, rotphi2])
        t = np.array([0, 0, kA])

        kr0 = 1                   # Field point coordinate system 0: radius
        theta0 = np.pi/2          # Field point coordinate system 0: theta
        phi0 = np.pi/2            # Field point coordinate system 0: phi
        x0, y0, z0 = CartesianCoordFromSphericalCoord(kr0, theta0, phi0)

        x1, y1, z1 = r1.as_matrix().T.dot([x0, y0, z0])
        kr1, theta1, phi1 = SphericalCoordFromCartesianCoord(x1, y1, z1)

        x2, y2, z2 = x1 - t[0], y1 - t[1], z1 - t[2]
        kr2, theta2, phi2 = SphericalCoordFromCartesianCoord(x2, y2, z2)

        x3, y3, z3 = r2.as_matrix().T.dot([x2, y2, z2])
        kr3, theta3, phi3 = SphericalCoordFromCartesianCoord(x3, y3, z3)
            
        N = 10
        Q0 = PlaneWave(Ei, thetai, phii, N)
        Q1 = RotateCoefficients(Q0, rotphi1, rottheta1, rotchi1)
        Q2 = TranslateCoefficients(Q1, kA, c=1)
        Q3 = RotateCoefficients(Q2, rotphi2, rottheta2, rotchi2)

        Er0, Etheta0, Ephi0 = FieldValue(Q0, kr0, theta0, phi0)
        Er1, Etheta1, Ephi1 = FieldValue(Q1, kr1, theta1, phi1)
        Er2, Etheta2, Ephi2 = FieldValue(Q2, kr2, theta2, phi2)
        Er3, Etheta3, Ephi3 = FieldValue(Q3, kr3, theta3, phi3)

        Ex0, Ey0, Ez0 = CartesianVecFromSphericalVec(Er0, Etheta0, Ephi0, theta0, phi0)
        Ex1, Ey1, Ez1 = CartesianVecFromSphericalVec(Er1, Etheta1, Ephi1, theta1, phi1)
        Ex2, Ey2, Ez2 = CartesianVecFromSphericalVec(Er2, Etheta2, Ephi2, theta2, phi2)
        Ex3, Ey3, Ez3 = CartesianVecFromSphericalVec(Er3, Etheta3, Ephi3, theta3, phi3)

        print('The following vectors should be equal:')
        print(np.array([Ex0, Ey0, Ez0]))
        print(r1.as_matrix().dot([Ex1, Ey1, Ez1]))
        print(r1.as_matrix().dot([Ex2, Ey2, Ez2]))
        print(r1.as_matrix().dot(r2.as_matrix().dot([Ex3, Ey3, Ez3])))
