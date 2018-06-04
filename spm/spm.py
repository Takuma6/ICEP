import numpy as np
import functools
from . import utils

class Fluid:
    def __init__(self, mu, rho):
        """Initialize fluid structure with dynamic viscosity mu and density rho"""
        self.mu  = mu
        self.rho = rho
        self.nu = self.mu / self.rho
        
class Particle:
    def __init__(self, radius, xi, rho):
        """Initialize particle structure with radius a, interface xi, and density rho"""        
        self.radius = radius
        self.xi     = xi
        self.rho    = rho

class Particle2D(Particle):
    def __init__(self, radius, xi, rho):
        """Initialize 2D particle structure with radius a, interface xi, and density rho"""        
        super().__init__(radius, xi, rho)
        self.volume = np.pi*self.radius**2
        mass = self.rho * self.volume
        self.imass  = 1 / mass
        self.imoment= 2 / (mass*self.radius**2)

class Particle3D(Particle):
    def __init__(self, radius, xi, rho):
        """Initialize 3D particle structure with a radius a, interface xi, and density rho"""
        super().__init__(radius, xi, rho)
        self.volume = 4/3*np.pi*self.radius**3
        mass = self.rho * self.volume
        self.imass  = 1 / mass
        self.imoment= 5 / (2*mass*self.radius**2)


class Grid:
    def __init__(self, powers, dx):
        """Initialize rectangular grid  of size 2**powers, with grid spacing dx"""                
        self.ns  = np.array([2**n for n in powers], dtype=np.int)
        self.dx  = dx
        self.dv  = self.dx**self.dim
        self.length = self.ns*self.dx

        # init grid
        lattice = list(map(lambda li, ni: np.linspace(0.0, li, ni, endpoint=False), self.length, self.ns))
        self.X  = np.array(np.meshgrid(*lattice, indexing = 'ij'))
        
        lattice = list(map(lambda ni: np.fft.fftfreq(ni, d=self.dx)*2*np.pi, self.ns[:-1]))
        lattice.append(np.fft.rfftfreq(self.ns[-1], d=self.dx)*2*np.pi)
        self.K  = np.array(np.meshgrid(*lattice, indexing='ij'))
        self.K2 = np.einsum('i...,i...->...', self.K, self.K)

    def maxK2(self):
        """Compute maximum K2 for pseudo-spectral method"""                        
        return self.K2.max()

    def shiftK(self):
        """Return phase factors for staggered grid calculations"""
        return np.exp(1j*self.K*self.dx/2)

class Grid2D(Grid):
    dim = 2
    def _solenoidalProjectorK(self):
        """Initialize solenoidal projectors in K-space"""
        iK2 = 1 / np.where(self.K2 == 0, 1, self.K2).astype(float)
        Pxx = np.ones_like(self.K[0]) - iK2*self.K[0]**2
        Pyy = np.ones_like(self.K[1]) - iK2*self.K[1]**2        
        Pxy = -iK2*self.K[0]*self.K[1]
        return np.stack([Pxx, Pxy, Pxy, Pyy]).reshape((2,2)+Pxx.shape)
    def x2scalar(self, vx):
        return 0.5*(vx + np.roll(vx,1,axis=0))
    def y2scalar(self, vy):
        return 0.5*(vy + np.roll(vy,1,axis=1))

class Grid3D(Grid):
    dim = 3
    def _solenoidalProjectorK(self):
        """Initialize solenoidal projectors in K-space"""
        iK2 = 1 / np.where(self.K2 == 0, 1, self.K2).astype(float)
        Pxx = np.ones_like(self.K[0]) - iK2*self.K[0]**2
        Pyy = np.ones_like(self.K[1]) - iK2*self.K[1]**2
        Pzz = np.ones_like(self.K[2]) - iK2*self.K[2]**2
        Pxy = -iK2*self.K[0]*self.K[1]
        Pxz = -iK2*self.K[0]*self.K[2]
        Pyz = -iK2*self.K[1]*self.K[2]
        return np.stack([Pxx,Pxy,Pxz,Pxy,Pyy,Pyz,Pxz,Pyz,Pzz]).reshape((3,3)+Pxx.shape)
    
class SPM:
    def _particleGridDistance(self, Ri):
        """Compute (pbc) distance from particle center to grid points
        
        Args:
            Ri : particle position vector
        Returns:
            r - Ri for all r grid points"""
        return np.linalg.norm(self._particleGridDisplacement(Ri), axis=0)
    
    def makePhi(self, phi, R):
        """Compute phi field for given particle configuration
        
        Args:
            phi : phi(r) function
            R   : particle position vectors
        Returns:
            phi(r) = \sum_i phi_i(r)"""
        return functools.reduce(lambda a, b: a + b, map(lambda Ri: phi(self._particleGridDistance(Ri)), R))

    def makeUp(self, phi, R, V, O):
        """Compute total particle velocity field for given particle configuration
        
        Args:
            phi : phi(r) function
            R : particle position vectors
            V : particle velocity vectors
            O : particle angular velocities
        Returns:
            up(r) = \sum_i up_i(r)"""
        def up(Ri, Vi, Oi):
            dRi = self._particleGridDisplacement(Ri)
            return phi(np.linalg.norm(dRi, axis=0))*self._particleGridVelocity(dRi, Vi, Oi)
        
        return functools.reduce(lambda a, b: a + b, map(lambda Ri,Vi,Oi: up(Ri,Vi,Oi), R, V, O))

    def ffta(self, a):
        """Fourier transform of scalar field a(r)"""
        return np.fft.rfftn(a)
    def iffta(self, a):
        """Inverse Fourier transform of scalar field a(k)"""
        return np.fft.irfftn(a, self.grid.ns)
    def fftu(self, u):
        """Fourier transform for vector field u(r) = [u_1(r), u_2(r), ...]"""
        return np.stack([np.fft.rfftn(ui) for ui in u])
    def ifftu(self, u):
        """Inverse Fourier transform for vector field u(k) = [u_1(k), u_2(k), ...]"""
        return np.stack([np.fft.irfftn(ui, self.grid.ns) for ui in u])

class SPM2D(SPM):
    def __init__(self, params):
        """Instantiate 2D SPM object"""
        if len(params['grid']['powers']) != 2:
            print('expected dim = 2')
            return
        self.grid     = Grid2D(params['grid']['powers'], params['grid']['dx'])
        self.fluid    = Fluid(params['fluid']['mu'], params['fluid']['rho'])
        self.particle = Particle2D(params['particle']['a']*self.grid.dx, \
                                   params['particle']['a_xi']*self.grid.dx, \
                                   params['particle']['mass_ratio']*self.fluid.rho)
    
    def _particleGridDisplacement(self, Ri):
        """Compute (pbc) displacement vector from particle center to grid points"""
        return utils.distance(Ri[...,None,None], self.grid.X, self.grid.length[...,None,None])

    def _particleGridVelocity(self, dRi, Vi, Oi):
        """Compute single particle velocity field (WITHOUT phi_i(r) factor)
        
        Args:
            dRi : Displacement vectors from particle center to grid points
            Vi  : Particle velocity vector
            Oi  : Particle angular velocity
        Returns:
            up_i(r) = [V_i + O_i \cross r_i]"""
#        return Vi[...,None,None] + np.einsum('i...,i->i...', np.roll(dRi,1,axis=0), np.array([-Oi,Oi]))
        Wi = np.array([-Oi,Oi])
        return Vi[...,None,None] + np.roll(dRi,1,axis=0)*Wi[...,None,None]

    def makeForceHydro(self, phi, u, R, V, O):
        """Compute momentum change (F_i*dt) on all particles given updated flow field and particle configuration
        
        Args:
            phi : phi(r) function
            u : updated velocity field u^*
            R : updated particle positions at t = t_{n+1}
            V : old particle velocities at t_n
            O : old particle angular velocities at t_n"""
        def particleForce(phi, u, Ri, Vi, Oi):
            dRi = self._particleGridDisplacement(Ri)
            fp  = -phi(np.linalg.norm(dRi, axis=0))*(self._particleGridVelocity(dRi, Vi, Oi) - u)
            return np.stack([np.sum(fp, axis=(1,2)), np.array([np.sum(dRi[0,...]*fp[1,...] - dRi[1,...]*fp[0,...]), 0.0])])
        dvrho = self.grid.dv*self.fluid.rho
        fp    = np.array([particleForce(phi, u, Ri, Vi, Oi) for Ri,Vi,Oi in zip(R,V,O)])
        return dvrho*fp[:,0,:], dvrho*fp[:,1,0] # forces & torques

    def makeAdvectionK(self, uk):
        """Compute non-linear advection terms div(uu)
        
        Args:
            uk : updated total velocity field in k-space
        Returns:
            FT[div(uu)](k)"""
        u  = self.ifftu(uk)
        UK = self.fftu([u[0]**2, u[0]*u[1], u[1]**2])
        return np.stack([UK[0], UK[1], UK[1], UK[2]]).reshape((2,2)+UK[0].shape)

    def makeDivAdvectionK(self, uk):
        u  = self.ifftu(uk)
        UK = self.fftu([u[0]**2, u[0]*u[1], u[1]**2])
        return np.stack([self.grid.K[0]*UK[0] + self.grid.K[1]*UK[1], \
                         self.grid.K[0]*UK[1] + self.grid.K[1]*UK[2]])

class SPM3D(SPM):
    def __init__(self, params):
        """Instantiate 3D SPM object"""
        if len(params['grid']['powers']) != 3:
            print('expected dim = 3')
            return
        self.grid = Grid3D(params['grid']['powers'], params['grid']['dx'])
        self.fluid    = Fluid(params['fluid']['mu'], params['fluid']['rho'])
        self.particle = Particle3D(params['particle']['a']*self.grid.dx, \
                                   params['particle']['a_xi']*self.grid.dx, \
                                   params['particle']['mass_ratio']*self.fluid.rho)
    
    def _particleGridDisplacement(self, Ri):
        """Compute (pbc) distance vector from particle center to grid points"""
        return utils.distance(Ri[...,None,None,None], self.grid.X, self.grid.length[...,None,None,None])

    def _particleGridVelocity(self, dRi, Vi, Oi):
        """Compute single particle velocity field (WITHOUT phi_i(r) factor)

        Args:
            dRi : Displacement vectors from particle center to grid points
            Vi  : Particle velocity vector
            Oi  : Particle angular velocity vector
        Returns:
            up_i(r) = [V_i + O_i\cross r_i]"""
        return Vi[...,None,None,None] + np.cross(Oi[...,None,None,None], dRi, axis=0)

    def makeForceHydro(self, phi, u, R, V, O):
        """Compute momentum change (F_i*dt) on all particles given updated flow field and particle configuration

        Args:
            phi : phi(r) fucntion
            u : updated velocity field u^*
            R : updated particle positiosn at t = t_{n+1}
            V : old particle velocities at t_n
            O : old particle angular velocities at t_n"""
        def particleForce(phi, u, Ri, Vi, Oi):
            dRi = self._particleGridDisplacement(Ri)
            fpi = phi(np.linalg.norm(dRi, axis=0)) * (self._particleGridVelocity(dRi, Vi, Oi) - u)
            dmy = np.stack([np.sum(fpi, axis=(1,2,3)), np.sum(np.cross(dRi, fpi, axis=0), axis=(1,2,3))])
            return dmy
        dvrho = self.grid.dv*self.fluid.rho
        fp    = np.array([particleForce(phi, u, Ri, Vi, Oi) for Ri, Vi, Oi in zip(R, V, O)])
        return dvrho*fp[:,0,:], dvrho*fp[:,1,:]

    def makeAdvectionK(self, uk):
        """Compute non-linear advection terms div(uu)
        
        Args:
            uk : updated total velocit field in k-space
        Returns:
            FT[div(uu)](k)"""
        u = self.ifftu(uk)
        UK= self.fftu([u[0]**2, u[0]*u[1], u[0]*u[2], u[1]**2, u[1]*u[2], u[2]**2])
        return np.stack([UK[0], UK[1], UK[2], UK[1], UK[3], UK[4], UK[2], UK[4], UK[5]]).reshape((3,3)+UK[0].shape)
    
