import numpy as np
from numpy import cos,sin


#Datos generales
muTierra = 398600 #[km^3/s^2]
R_e = 6378 #[km]


def stateVector(e,hp,i,Ohm,omega,theta):
    ############################PROBLEMA A########################
    h = np.sqrt((hp+R_e)*muTierra*(1+e)) #Despeje de la EC de la orbita para theta 0 (Curtis 2.50)

    rf = h**2/muTierra*(1/(1+e*cos(theta)))*np.array([cos(theta),sin(theta),0]) #Vector posicion en sistema perifocal
    vf = muTierra/h*np.array([-sin(theta),e+cos(theta),0]) #Vector velocidad en sistema perifocal

    ############################PROBLEMA B#######################

    Q = np.array([[-sin(Ohm)*cos(i)*sin(omega)+cos(Ohm)*cos(omega), cos(Ohm)*cos(i)*sin(omega)+sin(Ohm)*cos(omega), sin(i)*sin(omega)],
                [-sin(Ohm)*cos(i)*cos(omega)-cos(Ohm)*sin(omega), cos(Ohm)*cos(i)*cos(omega)-sin(Ohm)*sin(omega), sin(i)*cos(omega)], 
                [               sin(Ohm)*sin(i),                                   -cos(Ohm)*sin(i),                   cos(i)      ]]) #Matriz de geo ecu a peri

    QT = np.transpose(Q) #Matriz de transformacion de Peri a Geo

    rfX = np.matmul(QT,rf) #Vector posicion en el sistema geo
    vfX = np.matmul(QT,vf) #Vector velocidad en el sistema geo

    return([rfX[0],rfX[1],rfX[2],vfX[0],vfX[1],vfX[2]])