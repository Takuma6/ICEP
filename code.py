#!/opt/anaconda/bin/python3

print("ok", flush=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
import h5py 
import functools
import importlib

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import lgmres

import spm.utils as utils
import spm.spm   as spm
def reload():
    importlib.reload(utils)
    importlib.reload(spm)

# main function
def solver(phi, uk, position, rotation, velocity, omega, \
                   charge, electricfield, electric_potential, \
                   phiFunc, fluidSolver, posSolver, velSolver, potentialSolver):
    # 1 - solute concentration
    phi_s               =   sys.makePhi(phi_sine, position) 
    charge              =   np.array([solverC(ci, sys.ifftu(uk) , position, electricfield, gi, zi, phi_s) for ci,gi,zi in zip(charge, gamma, ze)])
    rho_e               =   makeRhoe(charge, ze, phi_s)
    
    # 2 - advection / diffusion
    uk                  =   fluidSolver(uk); uk[:,0,0] = 0
    position, rotation  =   posSolver(position, velocity, rotation, omega)
    phi                 =   sys.makePhi(phiFunc, position)
    phi_s               =   sys.makePhi(phi_sine, position) 
    eps,deps            =   sys.makeDielectricField(em, position, rotation, phi_s)
    
    # 3 - electrostatic field
    # Ext. pot_ext are local variables
    potential, electricfield, rho_b, f_maxwell   =   potentialSolver(eps, Ext, rho_e, deps, electric_potential-potential_ext)
    potential          +=   potential_ext
    electricfield      +=   Ext 
    uk                  =   uk + dt*np.einsum('ij...,j...->i...', PKsole, sys.fftu(rho_e*electricfield+f_maxwell)); uk[:,0,0] = 0
    #uk                  =   solverEHD(uk, rho_e, electricfield, deps); uk[:,0,0] = 0
    
    # 4 - hydrodynamic forces
    u                   =   sys.ifftu(uk)
    force_h, torque_h   =   sys.makeForceHydro(phiFunc, u, position, velocity, omega)
    #force_g, torque_g = sys.makeForceGravity(phiFunc, np.array([0.0, -1e-2, 0.0])*(sys.particle.volume*(sys.particle.rho - sys.fluid.rho)), position)
    
    # 5 - update velocities
    velocity, omega     =   velSolver(velocity, omega, force_h/dt, torque_h/dt)
    u                   =   sys.makeUp(phiFunc, position, velocity, omega) - phi[None,...]*u   
    
    # 6 - particle constraint force
    uk                  = uk + np.einsum('ij...,j...->i...', PKsole, sys.fftu(u)); uk[:,0,0] = 0
    
    return phi, uk, position, rotation, velocity, omega, force_h/dt, torque_h/dt, \
    	   charge, potential, electricfield, rho_e, rho_b, f_maxwell, eps

# fluid & particle dynamics
def solverNS(uk):
    gnl    = -1j*np.einsum('ij...,k...,kj...->i...', PKsole, sys.grid.K, sys.makeAdvectionK(uk))
    ukstar = np.stack([phihL[0]*uk_d + dt*phihL[1]*gnl_d for uk_d,gnl_d in zip(uk,gnl)])
    return ukstar

def solverEHD(uk, free_charge_density, electricfield, deps):
    def _from_staggered_to_normal(vector):
        dmy = np.zeros_like(vector)
        for i in range(len(vector)):
            dmy[i][...] = 0.5*(vector[i] + np.roll(vector[i], 1, axis=i))
        return dmy
    E_on_normal = _from_staggered_to_normal(electricfield)
    A = free_charge_density[None,...]*E_on_normal
    B = -np.einsum("i...,i...->...", E_on_normal, E_on_normal)*_from_staggered_to_normal(deps)/2
    return uk + dt*np.einsum('ij...,j...->i...', PKsole, sys.fftu(A+B))

def solverParticlePos(position, velocity, rotation, omega):
    position_new = utils.pbc(position + velocity * dt, sys.grid.length)
    rotation_new = rotation + dt*sys.sloverRotation(omega, rotation)
    return position_new, sys.normalize(rotation_new)

def constantPosition(position, velocity, rotation, omega):
    return position, rotation

def constantRotation(position, velocity, rotation, omega):
    return utils.pbc(position + velocity * dt, sys.grid.length), rotation

def solverParticleVel(velocity, omega, force, torque):
    return velocity + sys.particle.imass*force*dt, omega + sys.particle.imoment*torque*dt

def constantVelocity(velocity, omega, force, torque):
    return velocity, omega

# electrodynamics
def makeRhoe(c, ze, phi_dmy):
    dmy = (1 - phi_dmy)*functools.reduce(lambda a, b: a + b, map(lambda zei,ci : zei*ci, ze, c))
    dmy = sys.ffta(dmy); dmy[0, 0] = 0
    return sys.iffta(dmy)

def solverC(charge, u, position, electric_field, gamma, ze, phi_dmy):
    nnsole  = sys.makeTanOp(phi_dmy)
    chargek = sys.ffta(charge)            
    A = sys.fftu(u*charge[None,...])
    B = sys.fftu(gamma*kbT*np.einsum("ij..., j...->i...", nnsole, sys.ifftu(1j*np.array(sys.grid.K)*chargek[None,...])))
    C = sys.fftu(gamma*ze*charge*np.einsum("ij..., j...->i...", nnsole, -electric_field))
    return sys.iffta(chargek - dt*1j*np.einsum("i..., i...->...", sys.grid.K, (A-B-C)))

def solverPoisson(eps, Ext, rho_e, deps):  
    def mvps(v):
        w = v.view()
        w.shape = eps.shape
        dmy = sys.ifftu(1j*sys.grid.K*sys.grid.shiftK()*sys.ffta(w))
        for i in range(len(dmy)):
            dmy[i][...] *= 0.5*(eps + np.roll(eps, -1, axis=i))
        dmy = sys.iffta(np.sum(1j*sys.grid.K*np.conj(sys.grid.shiftK())*sys.fftu(dmy), axis=0))
        dmy.shape = (NN)
        return dmy
    def rhs():
        dmy = Ext.copy()
        for i in range(len(dmy)):
            dmy[i][...] *= 0.5*(eps + np.roll(eps, -1, axis=i))
        dmy = sys.iffta(np.sum(1j*sys.grid.K*np.conj(sys.grid.shiftK())*sys.fftu(dmy), axis=0))
        dmy.shape = (NN)
        return dmy
    class gmres_counter(object):
        def __init__(self, disp=True):
            self._disp = disp
            self.niter = 0
        def __call__(self, rk=None):
            self.niter += 1
            if self._disp:
                print('iter %3i\t error = %.3e / %.3e' % (self.niter, np.max(np.abs(mvps(rk)-b)), np.max(np.abs(A*rk -b))))
    
    NN            = np.prod(eps.shape)
    A             = LinearOperator((NN,NN), matvec=mvps)
    b             = rhs() - rho_e.reshape(NN)
    counter       = gmres_counter()
    pot, exitcode = sp.sparse.linalg.lgmres(A, b, tol=1e-5)#, callback=counter)
    pot.shape     = eps.shape
    E             = -sys.ifftu(1j*sys.grid.K*sys.grid.shiftK()*sys.ffta(pot)) 

    def bound_charge_solver(E_total, epsilon0):
        dmy = E_total.copy()
        eps_minus_eps0 = eps - epsilon0
        for i in range(len(dmy)):
            dmy[i][...] *= 0.5*(eps_minus_eps0 + np.roll(eps_minus_eps0, -1, axis=i))
        dmy = sys.iffta(np.sum(1j*sys.grid.K*np.conj(sys.grid.shiftK())*sys.fftu(dmy), axis=0))
        return dmy
    rho_b   = -bound_charge_solver(E+Ext, 1)
    
    def _solve_maxwell_force(E_, deps_, free_charge):
        def _from_staggered_to_normal(vector):
            dmy = np.zeros_like(vector)
            for i in range(len(vector)):
                dmy[i][...] = 0.5*(vector[i] + np.roll(vector[i], 1, axis=i))
            return dmy
        def _from_normal_to_staggered(scalar, dimention):
            dmy = np.zeros((dimention,)+ scalar.shape)
            for i in range(dimention):
                dmy[i][...] = 0.5*(scalar + np.roll(scalar, -1, axis=i))
            return dmy
        dmy = _from_staggered_to_normal(E_)
        E_2 = np.linalg.norm(dmy, axis=0)
        dmy_stag = deps_.copy()
        dmy_stag *= -0.5*_from_normal_to_staggered(E_2, len(E_))
        dmy_stag += _from_normal_to_staggered(free_charge, len(E_))*E_
        dmy_normal = _from_staggered_to_normal(dmy_stag)
        return dmy_stag, dmy_normal
    f_maxwell_staggered, f_maxwell_normal = _solve_maxwell_force(E+Ext, deps, rho_e)
    
    E[...]  = sys.grid.xyzScalar(E)
    return pot, E, rho_b, f_maxwell_normal

def solverPoisson2(eps, Ext, rho_e, deps, potential_in):  
    def mvps(v):
        w = v.view()
        w.shape = eps.shape
        dmy = sys.ifftu(1j*sys.grid.K*sys.grid.shiftK()*sys.ffta(w))
        for i in range(len(dmy)):
            dmy[i][...] *= 0.5*(eps + np.roll(eps, -1, axis=i))
        dmy = sys.iffta(np.sum(1j*sys.grid.K*np.conj(sys.grid.shiftK())*sys.fftu(dmy), axis=0))
        dmy.shape = (NN)
        return dmy
    def rhs():
        dmy = Ext.copy()
        for i in range(len(dmy)):
            dmy[i][...] *= 0.5*(eps + np.roll(eps, -1, axis=i))
        dmy = sys.iffta(np.sum(1j*sys.grid.K*np.conj(sys.grid.shiftK())*sys.fftu(dmy), axis=0))
        dmy.shape = (NN)
        return dmy
    class gmres_counter(object):
        def __init__(self, disp=True):
            self._disp = disp
            self.niter = 0
        def __call__(self, rk=None):
            self.niter += 1
            if self._disp:
                print('iter %3i\t error = %.3e / %.3e' % (self.niter, np.max(np.abs(mvps(rk)-b)), np.max(np.abs(A*rk -b))))
    
    NN            = np.prod(eps.shape)
    A             = LinearOperator((NN,NN), matvec=mvps)
    b             = rhs() - rho_e.reshape(NN)
    counter       = gmres_counter()
    pot, exitcode = sp.sparse.linalg.lgmres(A, b, x0=potential_in.reshape(NN), tol=1e-5)#, callback=counter)
    pot.shape     = eps.shape
    E             = -sys.ifftu(1j*sys.grid.K*sys.grid.shiftK()*sys.ffta(pot)) 

    def bound_charge_solver(E_total, epsilon0):
        dmy = E_total.copy()
        eps_minus_eps0 = eps - epsilon0
        for i in range(len(dmy)):
            dmy[i][...] *= 0.5*(eps_minus_eps0 + np.roll(eps_minus_eps0, -1, axis=i))
        dmy = sys.iffta(np.sum(1j*sys.grid.K*np.conj(sys.grid.shiftK())*sys.fftu(dmy), axis=0))
        return dmy
    rho_b   = -bound_charge_solver(E+Ext, 1)
    
    def _solve_maxwell_force(E_, deps_, free_charge):
        def _from_staggered_to_normal(vector):
            dmy = np.zeros_like(vector)
            for i in range(len(vector)):
                dmy[i][...] = 0.5*(vector[i] + np.roll(vector[i], 1, axis=i))
            return dmy
        def _from_normal_to_staggered(scalar, dimention):
            dmy = np.zeros((dimention,)+ scalar.shape)
            for i in range(dimention):
                dmy[i][...] = 0.5*(scalar + np.roll(scalar, -1, axis=i))
            return dmy
        dmy = _from_staggered_to_normal(E_)
        E_2 = np.linalg.norm(dmy, axis=0)
        dmy_stag = deps_.copy()
        dmy_stag *= -0.5*_from_normal_to_staggered(E_2, len(E_))
        #dmy_stag += _from_normal_to_staggered(free_charge, len(E_))*E_
        dmy_normal = _from_staggered_to_normal(dmy_stag)
        return dmy_stag, dmy_normal
    f_maxwell_staggered, f_maxwell_normal = _solve_maxwell_force(E+Ext, deps, rho_e)
    
    E[...]  = sys.grid.xyzScalar(E)
    return pot, E, rho_b, f_maxwell_normal

def uniform_ElectricField_x(coef_E = .1):
    Ext = np.zeros_like(sys.ifftu(uk)); Ext[0] = coef_E
    potential_ext = np.array(np.max(sys.grid.X[0]) - sys.grid.X[0])*coef_E
    return Ext, potential_ext

def uniform_ElectricField_y(coef_E = .1):
    Ext = np.zeros_like(sys.ifftu(uk)); Ext[1] = coef_E
    potential_ext = np.array(np.max(sys.grid.X[1]) - sys.grid.X[1])*coef_E
    return Ext, potential_ext

setder = lambda i : "trajectory/frame_" + str(np.int(i))
def saveh5(i, output, u, phi, position, rotation, velocity, omega, force, torque, \
           concentration, free_charge_density, bound_charge_density, electric_potential, electric_field, eps, f_maxwell, time):
    output.create_group(setder(i))
    output.create_dataset(setder(i)+'/Time', data = time*i)
    output.create_dataset(setder(i)+'/u', data = u)
    output.create_dataset(setder(i)+'/phi', data = phi)
    output.create_dataset(setder(i)+'/epsilon', data = eps)
    output.flush()
    output.create_dataset(setder(i)+'/R', data = position)
    output.create_dataset(setder(i)+'/Q', data = rotation)
    output.create_dataset(setder(i)+'/V', data = velocity)
    output.create_dataset(setder(i)+'/O', data = omega)
    output.create_dataset(setder(i)+'/Force_h', data = force)
    output.create_dataset(setder(i)+'/Torque_h', data = torque)
    output.flush()
    output.create_dataset(setder(i)+'/concentration', data = concentration)
    output.create_dataset(setder(i)+'/c_sum', data = np.sum(concentration, axis=(0,1,2)))
    output.create_dataset(setder(i)+'/free_charge_density', data = free_charge_density)
    output.create_dataset(setder(i)+'/bound_charge_density', data = bound_charge_density)
    output.create_dataset(setder(i)+'/electric_potential', data = electric_potential)
    output.create_dataset(setder(i)+'/electric_field', data = electric_field)
    output.create_dataset(setder(i)+'/maxwell_force', data = f_maxwell)
    output.flush()


# set parameters
print("SPM simulatin starts!", flush=True)
# system 
Np   = 6
dim  = 2
if dim==2:
    sys  = spm.SPM2D({'grid':{'powers':[Np,Np], 'dx':0.5},\
                      'particle':{'a':5, 'a_xi':2, 'mass_ratio':1.2},\
                      'fluid':{'rho':1.0, 'mu':1.0}})
elif dim==3:
    sys  = spm.SPM3D({'grid':{'powers':[Np,Np,Np], 'dx':0.5},\
                      'particle':{'a':5, 'a_xi':2, 'mass_ratio':1.2},\
                      'fluid':{'rho':1.0, 'mu':1.0}})

dt        = 1 / (sys.fluid.nu*sys.grid.maxK2())
phihL     = utils.etdPhi(-sys.fluid.nu*sys.grid.K2*dt)
phir      = (lambda x : utils.phiGauss(x, sys.particle.radius, sys.particle.xi, sys.grid.dx))
phi_sine  = (lambda x : utils.phiSine(x, sys.particle.radius, sys.particle.xi))

# electro-property
ze       = np.array([1,-1])[...,None]
gamma    = np.ones(2)[...,None]
kbT      = 1
species  = 2
epsilon0 = 1
em       = {'epsilon':{'head':10, 'tail':0.1, 'fluid':1}, \
			'sigma':{'head':0, 'tail':0, 'fluid':0}}

# particle property
R     = np.ones((1,dim))*sys.grid.length/2
Q     = sys.normalize([[1,0]]) 
V     = np.zeros_like(R)
O     = np.zeros(len(R)) #2d
#O     = np.zeros_like(R) #3d

# field property
phi                =   sys.makePhi(phir, R)
PKsole             =   sys.grid._solenoidalProjectorK()
uk                 =   np.einsum('ij...,j...->i...', PKsole, sys.fftu(sys.makeUp(phir, R, V, O)))
charge             =   np.ones((species, sys.grid.ns[0], sys.grid.ns[1])) #2d
#charge             =   np.ones((species, sys.grid.ns[0], sys.grid.ns[1], sys.grid.ns[2])) #3d
rho_e              =   makeRhoe(charge, ze, phi)

Ext, potential_ext    =   uniform_ElectricField_x()
phi_s                 =   sys.makePhi(phi_sine, R) 
eps, deps             =   sys.makeDielectricField(em, R, Q, phi_s)
potential, E, rho_b, f_maxwell   =   solverPoisson(eps, Ext, rho_e, deps)
E                    +=   Ext 
potential            +=   potential_ext     

nframes = 100
ngts    = 100
output_file = "output.hdf5"
outfh       = h5py.File(output_file, 'w')
saveh5(0, outfh, sys.ifftu(uk), phi, R, Q, V, O, O, O, charge, rho_e, rho_b, potential, E, eps, f_maxwell, dt*ngts)

for frame in range(nframes):
    print("now at loop:",frame, flush=True)
    for gts in range(ngts):
        phi, uk, R, Q, V, O, Fh, Nh, charge, potential, E, rho_e, rho_b, f_maxwell, eps \
            = solver(phi, uk, R, Q, V, O, charge, E, potential, phir, solverNS, constantRotation, solverParticleVel, solverPoisson2)
    saveh5(frame+1, outfh, sys.ifftu(uk), phi, R, Q, V, O, Fh, Nh, charge, rho_e, rho_b, potential, E, eps, f_maxwell, dt*ngts)
    outfh.flush()

outfh.flush()
outfh.close()

print("SPM Simulation Ended", flush=True)
