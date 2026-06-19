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
steps = 10000

Positionref,vref,tref    = SatPoints(P1*Orbits,steps,y0ref) #Reference satellite
PositionSat1,vsat1,tsat1 = SatPoints(P1*Orbits,steps,y01) #Smaller orbit than reference
PositionSat2,vsat2,tsat2 = SatPoints(P1*Orbits,steps,y01) #Smaller orbit than reference
PositionSat3,vsat3,tsat3 = SatPoints(P2*Orbits,steps,y02) #Bigger orbit than reference
PositionSat4,vsat4,tsat4 = SatPoints(P2*Orbits,steps,y02) #Bigger orbit than reference


Positionref1,vref1,tref1    = SatPoints(P1*Orbits,steps,np.concatenate((Positionref[-1],vref[-1])))
PositionSat11,vsat11,tsat11 = SatPoints(P1*Orbits,steps,np.concatenate((PositionSat1[-1],y0ref[3:6])))
PositionSat21,vsat21,tsat21 = SatPoints(P1*Orbits,steps,np.concatenate((PositionSat2[-1],vsat2[-1])))
PositionSat31,vsat31,tsat31 = SatPoints(P1*Orbits*2-P2*Orbits,steps,np.concatenate((PositionSat3[-1],y0ref[3:6])))
PositionSat41,vsat41,tsat41 = SatPoints(P1*Orbits*2-P2*Orbits,steps,np.concatenate((PositionSat4[-1],vsat4[-1])))

Positionref  = np.concatenate((Positionref,Positionref1))
PositionSat1 = np.concatenate((PositionSat1,PositionSat11))
PositionSat2 = np.concatenate((PositionSat2,PositionSat21))
PositionSat3 = np.concatenate((PositionSat3,PositionSat31))
PositionSat4 = np.concatenate((PositionSat4,PositionSat41))

tref  = np.concatenate((tref,tref1+tref[-1]+0.00000001))
tsat1 = np.concatenate((tsat1,tsat11+tsat1[-1]+0.00000001))
tsat2 = np.concatenate((tsat2,tsat21+tsat2[-1]+0.00000001))
tsat3 = np.concatenate((tsat3,tsat31+tsat3[-1]+0.00000001))
tsat4 = np.concatenate((tsat4,tsat41+tsat4[-1]+0.00000001))

newP3 = np.zeros_like(PositionSat3)
newP4 = np.zeros_like(PositionSat3)

for i in range(len(tref)):
    if tref[i] == tsat3[i]:
        newP3[i] = PositionSat3[i]
        print("equal")
    else:
        for j in range(len(tsat3)):
            if tref[i] > tsat3[j] and tref[i] < tsat3[j+1]:
                newP3[i] = (PositionSat3[j]+PositionSat3[j+1])/2
                break

for i in range(len(tref)):
    if tref[i] == tsat4[i]:
        newP4[i] = PositionSat4[i]
        print("equal")
    else:
        for j in range(len(tsat4)):
            if tref[i] > tsat3[j] and tref[i] < tsat4[j+1]:
                newP4[i] = (PositionSat4[j]+PositionSat4[j+1])/2
                break

PositionSat3 = newP3
PositionSat4 = newP4

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
sphereref = pv.Sphere(radius=100,center=Positionref[0])
SatRef = plotter.add_mesh(sphereref,color="red",name="SatRef")

sphereSat1 = pv.Sphere(radius=100,center=PositionSat1[0])
Sat1 = plotter.add_mesh(sphereSat1,color="green",name="Sat1")

sphereSat2 = pv.Sphere(radius=100,center=PositionSat2[0])
Sat2 = plotter.add_mesh(sphereSat2,color="green",name="Sat2")

sphereSat3 = pv.Sphere(radius=100,center=PositionSat3[0])
Sat3 = plotter.add_mesh(sphereSat3,color="yellow",name="Sat3")

sphereSat4 = pv.Sphere(radius=100,center=PositionSat4[0])
Sat4 = plotter.add_mesh(sphereSat4,color="yellow",name="Sat4")

plotter.camera_position = [(-335.2078914223734, -26617.65720402787, 15780.412349809743),(1.3878246591048082e-05, 0.0, -64.95921660585418),(-0.995974737497762, -0.036256256172517445, -0.08197442378337853)]

#plotter.open_movie("phasing.mp4",framerate=60,quality=10)
plotter.show(window_size=[1920,1080],interactive_update=True,title="Maneuver visualizer")

for i in range(1,20000):
    sphereref = pv.Sphere(radius=100,center=Positionref[i])
    SatRef = plotter.add_mesh(sphereref,color="red",name="SatRef")

    sphereSat1 = pv.Sphere(radius=100,center=PositionSat1[i])
    Sat1 = plotter.add_mesh(sphereSat1,color="green",name="Sat1")

    sphereSat2 = pv.Sphere(radius=100,center=PositionSat2[i])
    Sat2 = plotter.add_mesh(sphereSat2,color="green",name="Sat2")

    sphereSat3 = pv.Sphere(radius=100,center=PositionSat3[i])
    Sat3 = plotter.add_mesh(sphereSat3,color="yellow",name="Sat3")

    sphereSat4 = pv.Sphere(radius=100,center=PositionSat4[i])
    Sat4 = plotter.add_mesh(sphereSat4,color="yellow",name="Sat4")

    #plotter.write_frame()
    plotter.update()
    