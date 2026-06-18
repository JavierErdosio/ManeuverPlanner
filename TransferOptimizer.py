import numpy as  np
import pyvista as pv
from pyvista import examples
from stateVector import stateVector
from TwoBodySolver import SatPoints

#General data
mu = 398600 #[km^3/s^2]
Re = 6378 #[km]


#Problem specific data
Rpref,Raref = 870+Re, 1287.58+Re #[km]
Rp1,Ra1 = 870+Re,1279.2+Re #[km]
Rp2,Ra2 = 870+Re,1296+Re #[km]

eref = 0.028
i = np.deg2rad(58)
RAAN = np.deg2rad(0)
omega = np.deg2rad(90)
theta = np.deg2rad(0)

e1 = (Ra1-Rp1)/(Ra1+Rp1)
e2 = (Ra2-Rp2)/(Ra2+Rp2)

aref = (Rpref+Raref)/2
a1 = (Rp1+Ra1)/2
a2 = (Rp2+Ra2)/2


P = lambda a: 2*np.pi*((a)**3/mu)**0.5 #[s]
P1 = P(a1)/3600 #[hours]
P2 = P(a2)/3600 #[hours]

#Initial state vector
y0ref = stateVector(eref,870,i,RAAN,omega,theta)
y01 = stateVector(e1,870,i,RAAN,omega,theta)
y02 = stateVector(e2,870,i,RAAN,omega,theta)

#DeltaV
vRef = np.linalg.norm(y0ref[3:6]) #[km/s]
v01 = np.linalg.norm(y01[3:6])    #[km/s]
v02 = np.linalg.norm(y02[3:6])    #[km/s]

print("Required deltaV1 = %.3f [m/s]" %abs((v01-vRef)*1000))
print("Required deltaV2 = %.3f [m/s]" %abs((v02-vRef)*1000))

#Propagation
Orbits = 69
Positionref,vref = SatPoints(P1*Orbits,10000,y0ref) #Reference satellite

PositionSat1,vsat1 = SatPoints(P1*Orbits,10000,y01) #Smaller orbit than reference
PositionSat2,vsat2 = SatPoints(P1*Orbits,10000,y01) #Smaller orbit than reference

PositionSat3,vsat3 = SatPoints(P2*Orbits,10000,y02) #Bigger orbit than reference
PositionSat4,vsat4 = SatPoints(P2*Orbits,10000,y02) #Bigger orbit than reference


Positionref,vref   = SatPoints(P1*Orbits,10000,np.concatenate((Positionref[-1],vref[-1])))
PositionSat1,vsat1 = SatPoints(P1*Orbits,10000,np.concatenate((PositionSat1[-1],y0ref[3:6])))
PositionSat2,vsat2 = SatPoints(P1*Orbits,10000,np.concatenate((PositionSat2[-1],vsat2[-1])))
PositionSat3,vsat3 = SatPoints(P1*Orbits*2-P2*Orbits,10000,np.concatenate((PositionSat3[-1],y0ref[3:6])))
PositionSat4,vsat4 = SatPoints(P1*Orbits*2-P2*Orbits,10000,np.concatenate((PositionSat4[-1],vsat4[-1])))


Rfref = Positionref[-1]
Rf1 = PositionSat1[-1]
Rf2 = PositionSat2[-1]
Rf3 = PositionSat3[-1]
Rf4 = PositionSat4[-1]

th = lambda R,e,a: np.acos((a*(1-e**2)-R)/(R*e))

thref = np.rad2deg(th(np.linalg.norm(Rfref),eref,aref))

th1 = np.rad2deg(th(np.linalg.norm(Rf1),e1,a1))

th2 = np.rad2deg(th(np.linalg.norm(Rf2),e1,a1))

th3 = np.rad2deg(th(np.linalg.norm(Rf3),e2,a2))

th4 = np.rad2deg(th(np.linalg.norm(Rf4),e2,a2))

print(thref,th1,th2,th3,th4)


#Graph
plotter = pv.Plotter()

plotter = pv.Plotter()
    
#Earth
earth = pv.Sphere(radius=Re, theta_resolution=120, phi_resolution=120)
earth = earth.texture_map_to_sphere(inplace=True, prevent_seam=True)
xyz = earth.points
x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
u = 0.5 + np.arctan2(y, x) / (2 * np.pi)
v = 0.5 + np.arcsin(z / Re) / np.pi
earth.active_texture_coordinates = np.column_stack((u, v))
texture = pv.read_texture("earth_texture.jpg")
earth_actor = plotter.add_mesh(earth, texture=texture)

#Stars
cubemap = examples.download_cubemap_space_16k()
_ = plotter.add_actor(cubemap.to_skybox())
plotter.set_environment_texture(cubemap, is_srgb=True,resample=1 / 64)

#Orbits
orbit = pv.lines_from_points(Positionref)
plotter.add_mesh(orbit, color="red", line_width=1)

orbit = pv.lines_from_points(PositionSat1)
plotter.add_mesh(orbit, color="green", line_width=1)

orbit = pv.lines_from_points(PositionSat3)
plotter.add_mesh(orbit, color="yellow", line_width=1)

#Sats
sphereref = pv.Sphere(radius=100,center=Rfref)
plotter.add_mesh(sphereref,color="red")

sphereSat1 = pv.Sphere(radius=100,center=Rf1)
plotter.add_mesh(sphereSat1,color="green")

sphereSat2 = pv.Sphere(radius=100,center=Rf2)
plotter.add_mesh(sphereSat2,color="green")

sphereSat3 = pv.Sphere(radius=100,center=Rf3)
plotter.add_mesh(sphereSat3,color="yellow")

sphereSat4 = pv.Sphere(radius=100,center=Rf4)
plotter.add_mesh(sphereSat4,color="yellow")

#plotter.camera_position = "xy"

plotter.show(window_size=[1920,1080],interactive_update=False,title="Maneuver visualizer")