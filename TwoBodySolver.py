import numpy as np
from numpy import cos,sin
from scipy.integrate import odeint
from scipy.integrate import solve_ivp


#Datos generales
muTierra = 398600 #[km^3/s^2]
R_e = 6378 #[km]
f = 0.003353 



def SatPoints(hours,steps,rogvog):
    tf = hours*3600 #Tiempo [s] 
    pasos = steps

    #Resolucion
    y0 = rogvog #Vector de estado
    t = np.linspace(0, tf, pasos)

    def ec(t,y):
        X,Y,Z,Vx,Vy,Vz = y
        
        pos = np.linalg.norm([X,Y,Z])
        # Ecuaciones de estado
        Ax = -muTierra*(X)/pos**3
        Ay = -muTierra*(Y)/pos**3
        Az = -muTierra*(Z)/pos**3


        return([Vx,Vy,Vz,Ax,Ay,Az])

    sol = solve_ivp(ec,
                    [0,tf],
                    y0,t_eval=t,
                    method='DOP853',
                    rtol=1e-10,
                    atol=1e-14)

    
    x = sol.y[0]
    y = sol.y[1]
    z = sol.y[2]

    vx = sol.y[3]
    vy = sol.y[4]
    vz = sol.y[5]

    points = np.column_stack((x,y,z))
    velocity = np.column_stack((vx,vy,vz))

    return points,velocity
