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
def solver(phi, uk, position, velocity, omega, \
                   charge, electricfield, \
                   phiFunc, fluidSolver, posSolver, velSolver, potentialSolver):
    # 1 - solute concentration
    charge = np.array([solverC(ci, sys.ifftu(uk) , position, electricfield, gi, zi) for ci,gi,zi in zip(charge, gamma, ze)])
    rho_e = makeRhoe2(charge, ze, R)
    
    # 2 - advection / diffusion
    uk            = fluidSolver(uk); uk[:,0,0] = 0
    position  = posSolver(position, velocity)
    phi           = sys.makePhi(phiFunc, position)
    eps,deps = dielectric_field(em, position)
    
    # 3 - electrostatic field
    # Ext. pot_ext are local variables
    potential, electricfield, rho_b   = potentialSolver(eps, Ext, rho_e)
    potential   += potential_ext
    electricfield                 += Ext 
    uk                 = uk + dt*np.einsum('ij...,j...->i...', PKsole, sys.fftu((rho_e+rho_b)[None,...]*electricfield))
    
    # 4 - hydrodynamic forces
    u      = sys.ifftu(uk)
    force_h, torque_h = sys.makeForceHydro(phiFunc, u, position, velocity, omega)
    #force_g, torque_g = sys.makeForceGravity(phiFunc, np.array([0.0, -1e-2, 0.0])*(sys.particle.volume*(sys.particle.rho - sys.fluid.rho)), position)
    
    # 5 - update velocities
    velocity, omega  = velSolver(velocity, omega, force_h/dt, torque_h/dt)
    u      = sys.makeUp(phiFunc, position, velocity, omega) - phi[None,...]*u   
    
    # 6 - particle constraint force
    uk     = uk + np.einsum('ij...,j...->i...', PKsole, sys.fftu(u)); uk[:,0,0] = 0
    
    return phi, uk, position, velocity, omega, force_h/dt, torque_h/dt, charge, potential, electricfield, rho_e, rho_b


# fluid & particle dynamics
def solverNS(uk):
    gnl    = -1j*np.einsum('ij...,k...,kj...->i...', PKsole, sys.grid.K, sys.makeAdvectionK(uk))
    ukstar = np.stack([phihL[0]*uk_d + dt*phihL[1]*gnl_d for uk_d,gnl_d in zip(uk,gnl)])
    return ukstar

def solverParticlePos(position, velocity):
    return utils.pbc(position + velocity * dt, sys.grid.length)

def solverParticleVel(velocity, omega, force, torque):
    return velocity + sys.particle.imass*force*dt, omega + sys.particle.imoment*torque*dt

def constantVelocity(velocity, omega, force, torque):
    return velocity, omega

# electrodynamics
grad = lambda a : sys.ifftu(1j*np.array(sys.grid.K)*a[None,...])
def makeTanOp(phiFunc, position):
    phi_sine  = (lambda x : utils.phiSine(x, sys.particle.radius, sys.particle.xi))
    phi = sys.makePhi(phi_sine, position)
    gradPhi = grad(sys.ffta(phi))
    norm = np.linalg.norm(gradPhi, axis=0)
    n = gradPhi/np.where(norm == 0, 1, norm).astype(float)
    iid0 = phi==0
    iid1 = phi==1
    for i in range(2):
        n[i][iid0] = 0
        n[i][iid1] = 0
    return np.stack([np.ones_like(n[0])-n[0]*n[0], -n[0]*n[1]
                                ,-n[1]*n[0], np.ones_like(n[1])-n[1]*n[1]]).reshape((2,2)+n[0].shape)

def makeRhoe(phi, c, ze):
    return (1 - phi)*functools.reduce(lambda a, b: a + b, map(lambda zei,ci : zei*ci, ze, c))

def makeRhoe2(c, ze, position):
    phi_sine  = (lambda x : utils.phiSine(x, sys.particle.radius, sys.particle.xi))
    phi_dmy = sys.makePhi(phi_sine, position)
    dmy = (1 - phi_dmy)*functools.reduce(lambda a, b: a + b, map(lambda zei,ci : zei*ci, ze, c))
    dmy = sys.ffta(dmy); dmy[0, 0] = 0
    return sys.iffta(dmy)

def makeEPotential(rho_e_epsilon_k):
    rho_e_epsilon_k[0,0,0] = 0
    return sys.iffta(rho_e_epsilon_k/np.where(sys.grid.K2 == 0, 1, sys.grid.K2).astype(float))

def makeGradPotential(rho_e_epsilon_k):
    rho_e_epsilon_k[0,0,0] = 0
    phik = rho_e_epsilon_k/np.where(sys.grid.K2 == 0, 1, sys.grid.K2).astype(float)
    return sys.iffta(phik), sys.ifftu(-1j*np.einsum("i..., i...->i...", sys.grid.K, phik[None,...]))

def solverC(charge, u, position, E, gamma, ze):
    nnsole = makeTanOp(phi_sine, position)                # choosing phi_sine
    # advection term
    A = sys.fftu(u*charge[None,...])
    # concentration gradient term
    chargek = sys.ffta(charge)
    B = sys.fftu(gamma*kbT*np.einsum("ij..., j...->i...", nnsole, grad(chargek)))
    # electric force term
    C = sys.fftu(gamma*ze*charge*np.einsum("ij..., j...->i...", nnsole, -E))
    return sys.iffta(chargek - dt*1j*np.einsum("i..., i...->...", sys.grid.K, (A-B-C)))

def dielectric_field(em, position):
    phi_sine  = (lambda x : utils.phiSine(x, sys.particle.radius, sys.particle.xi))
    dmy = sys.makePhi(phi_sine, position)
    eps  = em['epsilon']['particle']*dmy+(1-dmy)*em['epsilon']['fluid']
    deps       = sys.ifftu(1j*sys.grid.K*sys.grid.shiftK()*sys.ffta(eps))
    return eps, deps

# dielectric field
def potential_solver_2(eps, Ext, rho_e):  
    def mvps(v):
        w = v.view()
        w.shape = eps.shape
        dmy = sys.ifftu(1j*sys.grid.K*sys.grid.shiftK()*sys.ffta(w))
        dmy[0][...] *= 0.5*(eps + np.roll(eps, -1, axis=0))
        dmy[1][...] *= 0.5*(eps + np.roll(eps, -1, axis=1))
        dmy = sys.iffta(np.sum(1j*sys.grid.K*np.conj(sys.grid.shiftK())*sys.fftu(dmy), axis=0))
        dmy.shape = (NN)
        return dmy
    def rhs():
        dmy = Ext.copy()
        dmy[0][...] *= 0.5*(eps + np.roll(eps, -1, axis=0))
        dmy[1][...] *= 0.5*(eps + np.roll(eps, -1, axis=1))
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
    NN                   = np.prod(eps.shape)
    A                      = LinearOperator((NN,NN), matvec=mvps)
    b                      = rhs() - rho_e.reshape(NN)
    counter           = gmres_counter()
    pot, exitcode = sp.sparse.linalg.lgmres(A, b, tol=1e-5)#, callback=counter)
    pot.shape       = eps.shape
    E                      = -sys.ifftu(1j*sys.grid.K*sys.grid.shiftK()*sys.ffta(pot)) 
    def bound_charge_solver(E_total, epsilon0):
        dmy = E_total.copy()
        eps_minus_eps0 = eps - epsilon0
        dmy[0][...] *= 0.5*(eps_minus_eps0 + np.roll(eps_minus_eps0, -1, axis=0))
        dmy[1][...] *= 0.5*(eps_minus_eps0 + np.roll(eps_minus_eps0, -1, axis=1))
        dmy = sys.iffta(np.sum(1j*sys.grid.K*np.conj(sys.grid.shiftK())*sys.fftu(dmy), axis=0))
        return dmy
    rho_b = - bound_charge_solver(E + Ext, 1)
    E[...]                = sys.grid.x2scalar(E[0]),sys.grid.y2scalar(E[1])
    return pot, E, rho_b

def dielectric_field(em, position):
    phi_sine  = (lambda x : utils.phiSine(x, sys.particle.radius, sys.particle.xi))
    dmy = sys.makePhi(phi_sine, position)
    eps  = em['epsilon']['particle']*dmy+(1-dmy)*em['epsilon']['fluid']
    deps       = sys.ifftu(1j*sys.grid.K*sys.grid.shiftK()*sys.ffta(eps))
    return eps, deps

def uniform_ElectricField_x(coef_E = 1):
    Ext = np.zeros_like(sys.ifftu(uk)); Ext[0] = coef_E
    potential_ext = np.array(np.max(sys.grid.X[0]) - sys.grid.X[0])*coef_E
    return Ext, potential_ext


setder = lambda i : "trajectory/frame_" + str(np.int(i))
def saveh5(i, output, u, phi, position, velocity, omega, force, torque, \
           concentration, free_charge_density, bound_charge_density, electric_potential, electric_field, time):
    output.create_group(setder(i))
    output.create_dataset(setder(i)+'/Time', data = time*i)
    output.create_dataset(setder(i)+'/u_x', data = u[0])
    output.create_dataset(setder(i)+'/u_y', data = u[1])
    output.create_dataset(setder(i)+'/phi', data = phi)
    output.flush()
    output.create_dataset(setder(i)+'/R', data = position)
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
    output.flush()


# set parameters
print("SPM simulatin starts!", flush=True)
# system 
Np = 6
sys     = spm.SPM2D({'grid':{'powers':[Np,Np], 'dx':0.5},\
                     'particle':{'a':10, 'a_xi':4, 'mass_ratio':1.2},\
                     'fluid':{'rho':1.0, 'mu':1.0}})
dt = 1 / (sys.fluid.nu*sys.grid.maxK2())
phihL   = utils.etdPhi(-sys.fluid.nu*sys.grid.K2*dt)
phir = (lambda x : utils.phiGauss(x, sys.particle.radius, sys.particle.xi, sys.grid.dx))
phi_sine  = (lambda x : utils.phiSine(x, sys.particle.radius, sys.particle.xi))

# electro-property
ze = np.array([1,-1])[...,None]
gamma = np.ones(2)[...,None]
kbT = 1
species = 2
epsilon0 = 1
em = {'epsilon':{'particle':100, 'fluid':1}, 'sigma':{'particle':0, 'fluid':0}}

# particle property
R     = np.ones((1,2))*sys.grid.length/2
V     = np.zeros_like(R)
O     = np.zeros(len(R))

# field property
phi                   =   sys.makePhi(phir, R)
PKsole             =   sys.grid._solenoidalProjectorK()
uk                    =   np.einsum('ij...,j...->i...', PKsole, sys.fftu(sys.makeUp(phir, R, V, O)))
charge            =   np.ones((species, sys.grid.ns[0], sys.grid.ns[1]))
rho_e              =   makeRhoe2(phi, charge, ze)

Ext, potential_ext = uniform_ElectricField_x()
E                      = Ext.copy() 
potential         =   potential_ext.copy()     

nframes = 100
output_file = "output.hdf5"
outfh = h5py.File(output_file, 'w')
saveh5(0, outfh, sys.ifftu(uk), phi, R, V, O, O, O, charge, rho_e, np.zeros_like(rho_e), potential, E, dt*100)

for frame in range(nframes):
    print("now at loop:",frame, flush=True)
    for gts in range(100):
        phi, uk, R, V, O, Fh, Nh, charge, potential, E, rho_e, rho_b \
            = solver(phi, uk, R, V, O, charge, E, phir, solverNS, solverParticlePos, solverParticleVel, potential_solver_2)
    saveh5(frame+1, outfh, sys.ifftu(uk), phi, R, V, O, Fh, Nh, charge, rho_e, rho_b, potential, E, dt*100)
    outfh.flush()

outfh.flush()
outfh.close()

print("SPM Simulation Ended", flush=True)
