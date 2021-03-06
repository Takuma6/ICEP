import numpy as np
import functools
import os
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

        # new term for a.c.
        lattice  = list(map(lambda ni: np.fft.fftfreq(ni, d=self.dx)*2*np.pi, self.ns))
        self.K_c = np.array(np.meshgrid(*lattice, indexing='ij'))

    def maxK2(self):
        """Compute maximum K2 for pseudo-spectral method"""                        
        return self.K2.max()

    def shiftK(self):
        """Return phase factors for staggered grid calculations in rfft"""
        return np.exp(1j*self.K*self.dx/2)

    def shiftK_c(self):
        """Return phase factors for staggered grid calculations for all k"""
        return np.exp(1j*self.K_c*self.dx/2)

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
    def xyzScalar(self, v):
        return self.x2scalar(v[0]), self.y2scalar(v[1])

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
    def x2scalar(self, vx):
        return 0.5*(vx + np.roll(vx,1,axis=0))
    def y2scalar(self, vy):
        return 0.5*(vy + np.roll(vy,1,axis=1))
    def z2scalar(self, vz):
        return 0.5*(vz + np.roll(vz,1,axis=2))
    def xyzScalar(self, v):
        return self.x2scalar(v[0]), self.y2scalar(v[1]), self.z2scalar(v[2])
    
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

    def makePhiWall(self, width_wall, axis='y'):
        """Compute phi field for walls
        
        Args:
            width_wall : int value of the width of walls
        Returns:
            phi_wall(r)"""
        xi=self.particle.xi
        if axis=='x':
            index_wall = 0
        elif axis=='y':
            index_wall = 1
        elif axis=='x':
            index_wall = 2
        else :
            print('invalid wall axis', flush=True)
            os._exit()
        X           = np.array(self.grid.X[index_wall])
        top_wall    = self.grid.length[index_wall] - width_wall*self.grid.dx
        bottom_wall = width_wall*self.grid.dx
        phir_wall   = (lambda x, width_wall, xi_wall : utils.phiSine(x, width_wall, xi_wall)) 
        phi_top     = 1-phir_wall(X, top_wall,    xi)
        phi_bottom  =   phir_wall(X, bottom_wall, xi)
        return phi_top+phi_bottom

    def makePhi_janus(self, phi, R, N):
        """Compute phi field with janus parameter for given particle configuration
        
        Args:
            phi : phi(r) function
            R   : particle translational position vectors
            N   : particle rotational position vectors
        Returns:
            phi(r) = \sum_i phi_i(r)"""
        return functools.reduce(lambda a, b: a + b, map(lambda Ri, ni: phi(self._particleGridDistance(Ri))*self._janusmap_tanh(Ri, ni), R, N))

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

    def normalize(self, x):
        return x / np.linalg.norm(x, axis=-1)[...,None]

    def _janusmap(self, Ri, ni):
        r    = self._particleGridDisplacement(Ri)
        norm = np.linalg.norm(r, axis=0)
        idx  = norm > 0
        r[:,idx] = r[:,idx] / norm[idx]
        r[:,np.logical_not(idx)] = 0
        return np.einsum('i,i...->...', ni, r)

    def _janus(self, p, Ri, ni, func):
        avg,delta = (p['head'] + p['tail'])/2, (p['head'] - p['tail'])
        return avg, delta, avg + self._janusmap(Ri, ni)*(delta/2)*func(self._particleGridDistance(Ri)/self.particle.radius)

    def makeDielectricField(self, electric_property, position, rotation, phi_, particle_id=0):
        avg,delta,test = self._janus(electric_property['epsilon'], position[particle_id], rotation[particle_id], (lambda x: x))
        epsilon = test*phi_+(1-phi_)*electric_property['epsilon']['fluid']
        d_epsilon       = self.ifftu(1j*self.grid.K*self.grid.shiftK()*self.ffta(epsilon))
        return epsilon, d_epsilon

    def _janusmap_tanh(self, Ri, ni):
        r    = self._particleGridDisplacement(Ri)
        r    = np.einsum('i,i...->...', ni, r)
        norm = np.linalg.norm(r, axis=0)
        idx  = norm > 0
        r[:,idx] = r[:,idx] / norm[idx]
        r[:,np.logical_not(idx)] = 0
        return r

    def _janus_tanh(self, p, R, N, sharpness):
        avg,delta = (p['head'] + p['tail'])/2, (p['head'] - p['tail'])
        phi_sine  = (lambda x : utils.phiSine(x, self.particle.radius, self.particle.xi))
        dmy       = self.makePhi_janus(phi_sine, R, N)
        return avg, delta, avg + (delta/2)*np.tanh(sharpness*dmy)

    def makeDielectricField_tanh(self, electric_property, position, rotation, phi_, sharpness=200):
        avg,delta,test = self._janus_tanh(electric_property['epsilon'], position, rotation, sharpness)
        epsilon        = test*phi_+(1-phi_)*electric_property['epsilon']['fluid']
        d_epsilon      = self.ifftu(1j*self.grid.K*self.grid.shiftK()*self.ffta(epsilon))
        return epsilon, d_epsilon

    # complex permittivity
    def _complex_permittivity(self, _epsilon, _sigma, _frequency): 
        return _epsilon -1j*_sigma/_frequency

    def _janus_tanh_complex(self, p, R, N, frequency, sharpness):
        p_head    = self._complex_permittivity(p['epsilon']['head'], p['sigma']['head'], frequency)
        p_tail    = self._complex_permittivity(p['epsilon']['tail'], p['sigma']['tail'], frequency)
        avg,delta = (p_head + p_tail)/2, (p_head - p_tail)
        phi_sine  = (lambda x : utils.phiSine(x, self.particle.radius, self.particle.xi))
        dmy       = self.makePhi_janus(phi_sine, R, N)
        return avg, delta, avg + (delta/2)*np.tanh(sharpness*dmy)

    def makeDielectricField_tanh_complex(self, electric_property, position, rotation, phi_, frequency, sharpness=200):
        avg,delta,test = self._janus_tanh_complex(electric_property, position, rotation, frequency, sharpness)
        p_fluid        = self._complex_permittivity(electric_property['epsilon']['fluid'], electric_property['sigma']['fluid'], frequency)
        epsilon        = test*phi_+(1-phi_)*p_fluid
        d_epsilon      = self.icfftu(1j*self.grid.K_c*self.grid.shiftK_c()*self.cffta(epsilon))
        return epsilon, d_epsilon

    def makeDielectricField_wall_tanh_complex(self, electric_property, position, rotation, _phi, _phi_wall, frequency, wall_prop='head', sharpness=200):
        avg,delta,test = self._janus_tanh_complex(electric_property, position, rotation, frequency, sharpness)
        p_fluid        = self._complex_permittivity(electric_property['epsilon']['fluid'], electric_property['sigma']['fluid'], frequency)
        if wall_prop=='head' or wall_prop=='tail':
            p_wall = self._complex_permittivity(electric_property['epsilon'][wall_prop], electric_property['sigma'][wall_prop], frequency)
        else :
            print('invalid wall property', flush=True)
            os._exit
        epsilon        = test*_phi + (1-_phi-_phi_wall)*p_fluid + p_wall*_phi_wall
        d_epsilon      = self.icfftu(1j*self.grid.K_c*self.grid.shiftK_c()*self.cffta(epsilon))
        return epsilon, d_epsilon

    def makeDielectricField_tanh_complex2(self, electric_property, position, rotation, phi_, frequency, charge, ze=1, D=1, Phi_0=1, sharpness=200):
        def _makeRhoe_complex_include_particle(c, ze):
            dmy = functools.reduce(lambda a, b: a + b, map(lambda zei,ci : zei*ci, ze, c))
            dmy = self.cffta(dmy); dmy[0, 0] = 0
            return self.icffta(dmy)
        free_charge_density = _makeRhoe_complex_include_particle(charge, np.array([ze,-ze])[...,None])
        avg,delta,test = self._janus_tanh_complex(electric_property, position, rotation, frequency, sharpness)
        sigma_f        = 2*ze*np.abs(free_charge_density)*D/Phi_0
        eps_f          = self._complex_permittivity(electric_property['epsilon']['fluid'], sigma_f, frequency)
        ind_0          = np.logical_not(sigma_f > electric_property['sigma']['fluid'])
        eps_f[ind_0]   = self._complex_permittivity(electric_property['epsilon']['fluid'], electric_property['sigma']['fluid'], frequency)
        epsilon        = test*phi_+(1-phi_)*eps_f
        d_epsilon      = self.icfftu(1j*self.grid.K_c*self.grid.shiftK_c()*self.cffta(epsilon))
        return epsilon, d_epsilon
 
    # fft for variables have only real values
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

    # normal fft
    def cffta(self, a):
        """Fourier transform of scalar field a(r)"""
        return np.fft.fftn(a)
    def icffta(self, a):
        """Inverse Fourier transform of scalar field a(k)"""
        return np.fft.ifftn(a, self.grid.ns)
    def cfftu(self, u):
        """Fourier transform for vector field u(r) = [u_1(r), u_2(r), ...]"""
        return np.stack([np.fft.fftn(ui) for ui in u])
    def icfftu(self, u):
        """Inverse Fourier transform for vector field u(k) = [u_1(k), u_2(k), ...]"""
        return np.stack([np.fft.ifftn(ui, self.grid.ns) for ui in u])

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

    def sloverRotation(self, omega, rotation):
        return omega*(np.dstack([-rotation[:,-1], rotation[:,0]]).reshape(rotation.shape))

    def makeTanOp(self, phi_dmy):
        gradPhi = self.ifftu(1j*np.array(self.grid.K)*self.ffta(phi_dmy)[None,...])
        norm = np.linalg.norm(gradPhi, axis=0)
        n = gradPhi/np.where(norm == 0, 1, norm).astype(float)
        iid0 = phi_dmy==0
        iid1 = phi_dmy==1
        for i in range(2):
            n[i][iid0] = 0
            n[i][iid1] = 0
        return np.stack([np.ones_like(n[0])-n[0]*n[0], -n[0]*n[1]
                                    ,-n[1]*n[0], np.ones_like(n[1])-n[1]*n[1]]).reshape((2,2)+n[0].shape)

    def makeSurfaceNormalShift(self, phi_dmy):
        def _interpolateNorm(vec, axis):
            _dmy = np.zeros_like(vec)
            for i in range(2):
                if i==axis:
                    _dmy[i] = vec[i]**2
                else:
                    _dmy[i] = (vec[i]**2 + np.roll(vec[i], -1, axis=axis)**2)/2
            return (_dmy[0]+_dmy[1])**(1/2)    
        gradPhi = self.ifftu(1j*np.array(self.grid.K)*self.grid.shiftK()*self.ffta(phi_dmy)[None,...])
        for i in range(2):
            norm = _interpolateNorm(gradPhi, axis=i)
            n[i] = gradPhi[i]/np.where(norm == 0, 1, norm).astype(float)
        iid0 = phi_dmy==0
        iid1 = phi_dmy==1
        for i in range(2):
            n[i][iid0] = 0
            n[i][iid1] = 0
        return n

    def makeRhoe(self, c, ze, phi_dmy):
        dmy = (1 - phi_dmy)*functools.reduce(lambda a, b: a + b, map(lambda zei,ci : zei*ci, ze, c))
        dmy = self.ffta(dmy); dmy[0, 0] = 0
        return self.iffta(dmy)

    def makeRhoe_complex(self, c, ze, phi_dmy):
        dmy = (1 - phi_dmy)*functools.reduce(lambda a, b: a + b, map(lambda zei,ci : zei*ci, ze, c))
        dmy = self.cffta(dmy); dmy[0, 0] = 0
        return self.icffta(dmy)

    def momentumConservation(self, uk):
        uk[:,0,0] = 0
        return

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
            fpi = -phi(np.linalg.norm(dRi, axis=0)) * (self._particleGridVelocity(dRi, Vi, Oi) - u)
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

    def makeTanOp(self, phi_dmy):
        gradPhi = self.ifftu(1j*np.array(self.grid.K)*self.ffta(phi_dmy)[None,...])
        norm = np.linalg.norm(gradPhi, axis=0)
        n = gradPhi/np.where(norm == 0, 1, norm).astype(float)
        iid0 = phi_dmy==0
        iid1 = phi_dmy==1
        for i in range(3):
            n[i][iid0] = 0
            n[i][iid1] = 0
        return np.stack([np.ones_like(n[0])-n[0]*n[0], -n[0]*n[1], -n[0]*n[2]
                        ,-n[1]*n[0], np.ones_like(n[1])-n[1]*n[1], -n[1]*n[2]
                        ,-n[2]*n[0], -n[2]*n[1], np.ones_like(n[2])-n[2]*n[2]]).reshape((3,3)+n[0].shape)
    
    def sloverRotation(self, omega, rotation):
        return np.cross(omega, rotation)

    def makeRhoe(self, c, ze, phi_dmy):
        dmy = (1 - phi_dmy)*functools.reduce(lambda a, b: a + b, map(lambda zei,ci : zei*ci, ze, c))
        dmy = self.ffta(dmy); dmy[0, 0, 0] = 0
        return self.iffta(dmy)

    def makeRhoe_complex(self, c, ze, phi_dmy):
        dmy = (1 - phi_dmy)*functools.reduce(lambda a, b: a + b, map(lambda zei,ci : zei*ci, ze, c))
        dmy = self.cffta(dmy); dmy[0, 0, 0] = 0
        return self.icffta(dmy)

    def momentumConservation(self, uk):
        uk[:,0,0,0] = 0
        return
