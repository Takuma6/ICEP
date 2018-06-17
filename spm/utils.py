import numpy as np

def pbc(r, lbox):
    """Enforce periodic boundary conditions on position vector r

    Args:
        r : position vector
        lbox : box dimensions 

    Returns:
        float : wrapped positions to lie within center box"""
    return np.fmod(r + lbox, lbox)

def distance(r1, r2, lbox):
    """Compute distance vector between two positions assuming PBC

    Args:
        r1 : center position, origin
        r2 : second position
        lbox : box dimensions

    Returns:
        float : distance vector r2 - r1 wrapped around box"""
    r12 = r2 - r1
    return r12 - np.around(r12 / lbox) * lbox

def phiSine(r, a, xi):
    """Compute smooth profile funciton using sines

    Args:
        r : radial distance
        a : particle radius
        xi : particle interface thickness

    Retuns:
        float : phi(r) value of phase field at position r"""
    x  = a-r
    hx = (np.sin(np.pi*x/xi)+1.0)/2.0
    hx[np.nonzero(x < -xi/2.0)] = 0.0
    hx[np.nonzero(x >  xi/2.0)] = 1.0
    return hx

def phiGauss(x, a, xi, delta):
    """Compute smooth profile funciton using gaussians

    Args:
        x : radial distance
        a : particle radius
        xi : particle interface thickness
        delta : grid spacing

    Returns:
        float : phi(r) value of phase field at position r"""
    def h(x):
        ans = np.zeros_like(x)
        sel = (x > 0.0)
        ans[sel] = np.exp(-(delta/x[sel])**2)
        return ans
    hxi = xi/2.0
    return h((a + hxi)-x)/(h((a+hxi)-x) + h(x-(a-hxi)))

def etdPhi(hL):
    """First two phi functions used for ETDRK integrators \phi_n(h L), with L the linear operator and h the time step

    Returns:
        [phi_0(hL), phi_0(hL)]
        phi_0(z) = exp(z)
        phi_1(z) = (epx(z) - 1) / (z)
    """
    def calculphi1(z):
        alpha = 1.0/np.array([1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0])
        ans   = np.zeros_like(z)
        large = np.abs(z) > 4.0e-2
        small = np.logical_not(large)
        ans[large] = (np.exp(z[large]) - 1.0)/z[large]

        ans[small] = alpha[0] + alpha[1]*z[small] + alpha[2]*z[small]**2 + \
                     alpha[3]*z[small]**3 + alpha[4]*z[small]**4
        
        large = np.logical_and(small, z > 4.0e-4)
        ans[large] = ans[large] + alpha[5]*z[large]**5

        large = np.logical_and(small, z > 4.0e-3)
        ans[large] = ans[large] + alpha[6]*z[large]**6

        return ans
    phi0 = np.exp(hL)
    phi1 = calculphi1(hL)
    return np.stack([phi0,phi1])
