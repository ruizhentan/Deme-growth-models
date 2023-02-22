# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 13:32:45 2023

@author: A101004
"""
# when glandular structures grow by invading. A cell newly
# created by a division will immediately attempt to invade a neighboring deme. 
import numpy as np
import pickle
from scipy import ndimage
from time import time
import matplotlib.pyplot as plt

def therapy_AT(Time, cap, grid_len, r_s, r_R, d0, r_mix, fs, fr, 
               s_spread, r_spread, r_shift, d_r, d_D, filestring):
    del_t = 0.01
    T = int(Time/(del_t))
    Ns = np.zeros((grid_len, grid_len,Time)).astype(int)
    Nr = np.zeros((grid_len, grid_len,Time)).astype(int)

    # generate initial distribution of cells 
    Ns[int(0.5*grid_len-s_spread/2):int(0.5*grid_len+s_spread/2),int(0.5*grid_len-s_spread/2):int(0.5*grid_len+s_spread/2),0] = np.random.normal(loc=cap*fs, scale=cap*fs*0.2, size=(int(s_spread),int(s_spread))).astype(int)
    pr = fr*(s_spread/r_spread)**2
    Nr[int(r_shift + 0.5*grid_len-r_spread/2):int(r_shift + 0.5*grid_len+r_spread/2),int(r_shift + 0.5*grid_len-r_spread/2):int(r_shift + 0.5*grid_len+r_spread/2),0] = np.random.binomial(Ns[int(0.5*grid_len-r_spread/2):int(0.5*grid_len+r_spread/2),int(0.5*grid_len-r_spread/2):int(0.5*grid_len+r_spread/2),0], pr)
    Ns[:,:,0] = Ns[:,:,0] - Nr[:,:,0]
    N_initial = np.sum(Nr[:,:,0]+Ns[:,:,0])

    Ns_temp1 = np.zeros((grid_len, grid_len)).astype(int)
    Nr_temp1 = np.zeros((grid_len, grid_len)).astype(int)
    Ns_temp1[:,:] = Ns[:,:,0]
    Nr_temp1[:,:] = Nr[:,:,0]
    Drug = np.zeros(T).astype(int)
    Ns_num = np.zeros(T).astype(int)
    Nr_num = np.zeros(T).astype(int)
    Drug[0] = 1
    drug = 1
    
    for t in range(1,T): 
        # growth rates.
        dG = (1 - 0.5*(Ns_temp1[:,:]+Nr_temp1[:,:])/cap)
        dG[np.where(dG < 0)] = 0 # make sure growth doesn't get negative number.
        Ns_new = np.random.binomial(Ns_temp1, del_t*r_s*dG)
        Nr_new = np.random.binomial(Nr_temp1, del_t*r_R*dG)
        # killing of sensitive cells by drug. 
        if drug == 1:
            Ns_new_kill = np.random.binomial(Ns_new, d_D)
            Ns_new = Ns_new - Ns_new_kill
            Ns_temp1[:,:] = Ns_temp1[:,:] - Ns_new_kill #substract the cells that die during division.            
            
        # invasion          
        Ns_new_invade = np.random.binomial(Ns_new, r_mix)
        Nr_new_invade = np.random.binomial(Nr_new, r_mix)
        Ns_temp1[:,:] = Ns_temp1[:,:] + Ns_new - Ns_new_invade
        Nr_temp1[:,:] = Nr_temp1[:,:] + Nr_new - Nr_new_invade
        # # invasion.
        if (np.sum(Ns_new_invade) > 0): # there are invading cells. 
             I, J = np.where(Ns_new_invade > 0)
             ns = [Ns_new_invade[I[ii],J[ii]] for ii in range(0, len(I))]
             I = [I[ii] for ii in range(0, len(I)) for jj in range(0,ns[ii])] 
             J = [J[ii] for ii in range(0, len(J)) for jj in range(0,ns[ii])] 
             loc_new = np.random.choice([0,np.pi/2,np.pi,3*np.pi/2],np.sum(Ns_new_invade))
             I1 = I + np.cos(loc_new).astype(int)
             J1 = J + np.sin(loc_new).astype(int)
             Ns_temp1[I1,J1] = Ns_temp1[I1,J1]+ Ns_new_invade[I,J]
        if (np.sum(Nr_new_invade) > 0): # there are invading cells. 
             I, J = np.where(Nr_new_invade > 0)
             ns = [Nr_new_invade[I[ii],J[ii]] for ii in range(0, len(I))]
             I = [I[ii] for ii in range(0, len(I)) for jj in range(0,ns[ii])] 
             J = [J[ii] for ii in range(0, len(J)) for jj in range(0,ns[ii])] 
             loc_new = np.random.choice([0,np.pi/2,np.pi,3*np.pi/2],np.sum(Nr_new_invade))
             I1 = I + np.cos(loc_new).astype(int)
             J1 = J + np.sin(loc_new).astype(int)
             Nr_temp1[I1,J1] = Nr_temp1[I1,J1]+ Nr_new_invade[I,J]  
        
        # death rates
        Nr_dead = np.random.binomial(Nr_temp1[:,:], d0*del_t)
        Ns_dead = np.random.binomial(Ns_temp1[:,:], d0*del_t)
        Ns_temp1[:,:] = Ns_temp1[:,:] - Ns_dead
        Nr_temp1[:,:] = Nr_temp1[:,:] - Nr_dead
                               
        # death rates
        Nr_dead = np.random.binomial(Nr_temp1[:,:], d0*del_t)
        Ns_dead = np.random.binomial(Ns_temp1[:,:], d0*del_t)
        Ns_temp1[:,:] = Ns_temp1[:,:] - Ns_dead
        Nr_temp1[:,:] = Nr_temp1[:,:] - Nr_dead
    
        # update temporary arrays
        if (drug == 1):
            if((np.sum(Ns_temp1[:,:] + Nr_temp1[:,:]))< int(0.5*N_initial)):
                drug = 0
        else:
            if((np.sum(Ns_temp1[:,:] + Nr_temp1[:,:]))> int(N_initial)):
                drug = 1 
        Drug[t] = drug
        Ns_num[t] = np.sum(Ns_temp1)
        Nr_num[t] = np.sum(Nr_temp1)
        # Save distribution every cell cycle
        if (t%100 == 0):
            Ns[:,:,int(t/100)]= Ns_temp1
            Nr[:,:,int(t/100)]= Nr_temp1
            if (np.sum(Ns_temp1 + Nr_temp1) >= 1.25*N_initial): #end the simulations if growth to 1.25 of original size before treatment. 
                break

    f1 = open(filestring, 'wb')
    pickle.dump([Ns[:,:,0:int(t/100)],Nr[:,:,0:int(t/100)], t,Ns_num[0:t], Nr_num[0:t], Drug[0:t],Time, cap, grid_len, r_s, r_R, d0, r_mix, fs, fr, 
                   s_spread, r_spread, r_shift, d_r, d_D], f1)
    f1.close() 
    return Ns[:,:,0:t], Nr[:,:,0:t], Drug[0:t], t


def therapy_CT(Time, cap, grid_len, r_s, r_R, d0, r_mix, fs, fr, 
               s_spread, r_spread, r_shift, d_r, d_D, filestring):
    del_t = 0.01
    T = int(Time/(del_t))
    Ns = np.zeros((grid_len, grid_len,Time)).astype(int)
    Nr = np.zeros((grid_len, grid_len,Time)).astype(int)

    # generate initial distribution of cells 
    Ns[int(0.5*grid_len-s_spread/2):int(0.5*grid_len+s_spread/2),int(0.5*grid_len-s_spread/2):int(0.5*grid_len+s_spread/2),0] = np.random.normal(loc=cap*fs, scale=cap*fs*0.2, size=(int(s_spread),int(s_spread))).astype(int)
    pr = fr*(s_spread/r_spread)**2
    Nr[int(r_shift + 0.5*grid_len-r_spread/2):int(r_shift + 0.5*grid_len+r_spread/2),int(r_shift + 0.5*grid_len-r_spread/2):int(r_shift + 0.5*grid_len+r_spread/2),0] = np.random.binomial(Ns[int(0.5*grid_len-r_spread/2):int(0.5*grid_len+r_spread/2),int(0.5*grid_len-r_spread/2):int(0.5*grid_len+r_spread/2),0], pr)
    Ns[:,:,0] = Ns[:,:,0] - Nr[:,:,0]
    N_initial = np.sum(Nr[:,:,0]+Ns[:,:,0])

    Ns_temp1 = np.zeros((grid_len, grid_len)).astype(int)
    Nr_temp1 = np.zeros((grid_len, grid_len)).astype(int)
    Ns_temp1[:,:] = Ns[:,:,0]
    Nr_temp1[:,:] = Nr[:,:,0]
    Drug = 1 # dummy
    Ns_num = np.zeros(T).astype(int)
    Nr_num = np.zeros(T).astype(int)
    
    for t in range(1,T): 
        # growth rates.
        dG = (1 - 0.5*(Ns_temp1[:,:]+Nr_temp1[:,:])/cap)
        dG[np.where(dG < 0)] = 0
        Ns_new = np.random.binomial(Ns_temp1, del_t*r_s*dG)
        Nr_new = np.random.binomial(Nr_temp1, del_t*r_R*dG)
        # killing of sensitive cells by drug. 

        Ns_new_kill = np.random.binomial(Ns_new, d_D)
        Ns_new = Ns_new - Ns_new_kill
        Ns_temp1[:,:] = Ns_temp1[:,:] - Ns_new_kill #substract the cells that die during division.            

        # invasion          
        Ns_new_invade = np.random.binomial(Ns_new, r_mix)
        Nr_new_invade = np.random.binomial(Nr_new, r_mix)
        Ns_temp1[:,:] = Ns_temp1[:,:] + Ns_new - Ns_new_invade
        Nr_temp1[:,:] = Nr_temp1[:,:] + Nr_new - Nr_new_invade
        # # invasion.
        if (np.sum(Ns_new_invade) > 0): # there are invading cells. 
             I, J = np.where(Ns_new_invade > 0)
             ns = [Ns_new_invade[I[ii],J[ii]] for ii in range(0, len(I))]
             I = [I[ii] for ii in range(0, len(I)) for jj in range(0,ns[ii])] 
             J = [J[ii] for ii in range(0, len(J)) for jj in range(0,ns[ii])] 
             loc_new = np.random.choice([0,np.pi/2,np.pi,3*np.pi/2],np.sum(Ns_new_invade))
             I1 = I + np.cos(loc_new).astype(int)
             J1 = J + np.sin(loc_new).astype(int)
             Ns_temp1[I1,J1] = Ns_temp1[I1,J1]+ Ns_new_invade[I,J]
        if (np.sum(Nr_new_invade) > 0): # there are invading cells. 
             I, J = np.where(Nr_new_invade > 0)
             ns = [Nr_new_invade[I[ii],J[ii]] for ii in range(0, len(I))]
             I = [I[ii] for ii in range(0, len(I)) for jj in range(0,ns[ii])] 
             J = [J[ii] for ii in range(0, len(J)) for jj in range(0,ns[ii])] 
             loc_new = np.random.choice([0,np.pi/2,np.pi,3*np.pi/2],np.sum(Nr_new_invade))
             I1 = I + np.cos(loc_new).astype(int)
             J1 = J + np.sin(loc_new).astype(int)
             Nr_temp1[I1,J1] = Nr_temp1[I1,J1]+ Nr_new_invade[I,J]           
        # death rates
        Nr_dead = np.random.binomial(Nr_temp1[:,:], d0*del_t)
        Ns_dead = np.random.binomial(Ns_temp1[:,:], d0*del_t)
        Ns_temp1[:,:] = Ns_temp1[:,:] - Ns_dead
        Nr_temp1[:,:] = Nr_temp1[:,:] - Nr_dead
    
        # update temporary arrays
        Ns_num[t] = np.sum(Ns_temp1)
        Nr_num[t] = np.sum(Nr_temp1)
        # Save distribution every cell cycle
        if (t%100 == 0):
            Ns[:,:,int(t/100)]= Ns_temp1
            Nr[:,:,int(t/100)]= Nr_temp1
            if (np.sum(Ns_temp1 + Nr_temp1) >= 1.25*N_initial): #end the simulations if growth to 1.25 of original size before treatment. 
                break

    f1 = open(filestring, 'wb')
    pickle.dump([Ns[:,:,0:int(t/100)],Nr[:,:,0:int(t/100)], t,Ns_num[0:t], Nr_num[0:t], Drug,Time, cap, grid_len, r_s, r_R, d0, r_mix, fs, fr, 
                   s_spread, r_spread, r_shift, d_r, d_D], f1)
    f1.close() 
    return Ns[:,:,0:t], Nr[:,:,0:t], Drug, t


def therapy_NT(Time, cap, cap_t, grid_len, r_s, r_R, d0, ft, fr, 
               s_spread, r_spread, d_r, d_D, filestring): # no treatment at all. Control. 
    del_t = 0.01
    steps = 10
    T = int(Time/(steps*del_t))
    Ns = np.zeros((grid_len, grid_len,T)).astype(int)
    Nr = np.zeros((grid_len, grid_len,T)).astype(int)

    # generate initial distribution of cells 
    fs = ft*cap_t/(s_spread**2*cap)
    Ns[int(0.5*grid_len-s_spread/2):int(0.5*grid_len+s_spread/2),int(0.5*grid_len-s_spread/2):int(0.5*grid_len+s_spread/2),0] = np.random.normal(loc=cap*fs, scale=cap*fs*0.2, size=(int(s_spread),int(s_spread))).astype(int)
    pr = fr*(s_spread/r_spread)**2
    Nr[int(0.5*grid_len-r_spread/2):int(0.5*grid_len+r_spread/2),int(0.5*grid_len-r_spread/2):int(0.5*grid_len+r_spread/2),0] = np.random.binomial(Ns[int(0.5*grid_len-r_spread/2):int(0.5*grid_len+r_spread/2),int(0.5*grid_len-r_spread/2):int(0.5*grid_len+r_spread/2),0], pr)
    Ns[:,:,0] = Ns[:,:,0] - Nr[:,:,0]
    N_total = np.sum(Nr[:,:,0]+Ns[:,:,0])

    Ns_temp1 = np.zeros((grid_len, grid_len)).astype(int)
    Nr_temp1 = np.zeros((grid_len, grid_len)).astype(int)
    Ns_temp1[:,:] = Ns[:,:,0]
    Nr_temp1[:,:] = Nr[:,:,0]
    
    for t in range(1,T): # 10 cycles of division.
        # update into main array
        Ns[:,:,t] = Ns_temp1[:,:]
        Nr[:,:,t] = Nr_temp1[:,:]  
        for tt in range(10):
            # growth rates.
            dG = 1 - 0.5*(Ns_temp1[:,:]+Nr_temp1[:,:])/cap
            dG[np.where(dG < 0)] = 0
            Ns_new = np.random.binomial(Ns_temp1, del_t*r_s*dG)
            Nr_new = np.random.binomial(Nr_temp1, del_t*r_s*dG)

            Ns_temp1[:,:] = Ns_temp1[:,:] + Ns_new 
            Nr_temp1[:,:] = Nr_temp1[:,:] + Nr_new 
          
            # death rates
            Nr_dead = np.random.binomial(Nr_temp1[:,:], d0*del_t)
            Ns_dead = np.random.binomial(Ns_temp1[:,:], d0*del_t)
            Ns_temp1[:,:] = Ns_temp1[:,:] - Ns_dead
            Nr_temp1[:,:] = Nr_temp1[:,:] - Nr_dead
    
            # deme division (only for deme with greater than carrying capacity.)
            N = Ns_temp1[:,:] + Nr_temp1[:,:]
            I, J = np.where(N > cap) #identify demes that has potential to divide. 
            thres = d_r*(N[I,J]-cap)/cap  
            prob_div = np.random.uniform(low=0.0, high= 1.0, size= len(I))
            loc_new = np.random.choice([0,np.pi/2,np.pi,3*np.pi/2],len(I)) # direction to shift, if it divides. 
            div = np.where(prob_div < thres)[0]
            for ii in div:
                I0, J0 = I[ii], J[ii]
                I1, J1 = I0 + np.cos(loc_new[ii]).astype(int), J0 + np.sin(loc_new[ii]).astype(int)
                a = Ns_temp1[I0,J0]
                b = Nr_temp1[I0,J0]
                Ns_temp1[I0,J0] = np.ceil(0.5*a)
                Nr_temp1[I0,J0] = np.ceil(0.5*b)
                if ((I1 - I0) ==0): # shift is in y-direction
                    if ((J1 - J0)==1): #shift in positive y.
                        Ns_temp1[I1,(J1+1):grid_len-1] = Ns_temp1[I1,J1:grid_len-2]
                        Nr_temp1[I1,(J1+1):grid_len-1] = Nr_temp1[I1,J1:grid_len-2]
                    else: 
                        Ns_temp1[I1,0:J1-1] = Ns_temp1[I1,1:J1] 
                        Nr_temp1[I1,0:J1-1] = Nr_temp1[I1,1:J1]
                else: # shift is in x-direction
                    if ((I1 - I0)==1): #shift in positive x.
                        Ns_temp1[(I1+1):grid_len-1,J1] = Ns_temp1[I1:grid_len-2,J1]
                        Nr_temp1[(I1+1):grid_len-1,J1] = Nr_temp1[I1:grid_len-2,J1]
                    else: 
                        Ns_temp1[0:I1-1,J1] = Ns_temp1[1:I1,J1]  
                        Nr_temp1[0:I1-1,J1] = Nr_temp1[1:I1,J1]  
                Ns_temp1[I1,J1] = np.floor(0.5*a) 
                Nr_temp1[I1,J1] = np.floor(0.5*b)       

        if (np.sum(Ns[:,:,t] + Nr[:,:,t]) > 1.25*N_total): #end the simulations if growth to 1.25 of original size before treatment. 
            Ns[:,:,t+1] = Ns_temp1[:,:]
            Nr[:,:,t+1] = Nr_temp1[:,:]      
            break
    f1 = open(filestring, 'wb')
    pickle.dump([Ns,Nr,t, Time, cap, grid_len, r_s, r_R, d0, fs, fr, 
                   s_spread, r_spread, d_r, d_D], f1)
    f1.close() 
    return Ns[:,:,t], Nr[:,:,t], t


###################################################################
### when resistant cell is more invasive than sensitive cells.  ###
###################################################################

def therapy_ATmol(Time, cap, grid_len, r_s, r_R, d0, r_mix_s, r_mix_r, fs, fr, 
               s_spread, r_spread, r_shift, d_r, d_D, filestring):
    del_t = 0.01
    T = int(Time/(del_t))
    Ns = np.zeros((grid_len, grid_len,Time)).astype(int)
    Nr = np.zeros((grid_len, grid_len,Time)).astype(int)

    # generate initial distribution of cells 
    Ns[int(0.5*grid_len-s_spread/2):int(0.5*grid_len+s_spread/2),int(0.5*grid_len-s_spread/2):int(0.5*grid_len+s_spread/2),0] = np.random.normal(loc=cap*fs, scale=cap*fs*0.2, size=(int(s_spread),int(s_spread))).astype(int)
    pr = fr*(s_spread/r_spread)**2
    Nr[int(r_shift + 0.5*grid_len-r_spread/2):int(r_shift + 0.5*grid_len+r_spread/2),int(r_shift + 0.5*grid_len-r_spread/2):int(r_shift + 0.5*grid_len+r_spread/2),0] = np.random.binomial(Ns[int(0.5*grid_len-r_spread/2):int(0.5*grid_len+r_spread/2),int(0.5*grid_len-r_spread/2):int(0.5*grid_len+r_spread/2),0], pr)
    Ns[:,:,0] = Ns[:,:,0] - Nr[:,:,0]
    N_initial = np.sum(Nr[:,:,0]+Ns[:,:,0])

    Ns_temp1 = np.zeros((grid_len, grid_len)).astype(int)
    Nr_temp1 = np.zeros((grid_len, grid_len)).astype(int)
    Ns_temp1[:,:] = Ns[:,:,0]
    Nr_temp1[:,:] = Nr[:,:,0]
    Drug = np.zeros(T).astype(int)
    Ns_num = np.zeros(T).astype(int)
    Nr_num = np.zeros(T).astype(int)
    Drug[0] = 1
    drug = 1
    
    for t in range(1,T): 
        # growth rates.
        dG = (1 - 0.5*(Ns_temp1[:,:]+Nr_temp1[:,:])/cap)
        dG[np.where(dG < 0)] = 0 # make sure growth doesn't get negative number.
        Ns_new = np.random.binomial(Ns_temp1, del_t*r_s*dG)
        Nr_new = np.random.binomial(Nr_temp1, del_t*r_R*dG)
        # killing of sensitive cells by drug. 
        if drug == 1:
            Ns_new_kill = np.random.binomial(Ns_new, d_D)
            Ns_new = Ns_new - Ns_new_kill
            Ns_temp1[:,:] = Ns_temp1[:,:] - Ns_new_kill #substract the cells that die during division.            
            
        # invasion          
        Ns_new_invade = np.random.binomial(Ns_new, r_mix_s)
        Nr_new_invade = np.random.binomial(Nr_new, r_mix_r)
        Ns_temp1[:,:] = Ns_temp1[:,:] + Ns_new - Ns_new_invade
        Nr_temp1[:,:] = Nr_temp1[:,:] + Nr_new - Nr_new_invade
        # # invasion.
        if (np.sum(Ns_new_invade) > 0): # there are invading cells. 
             I, J = np.where(Ns_new_invade > 0)
             ns = [Ns_new_invade[I[ii],J[ii]] for ii in range(0, len(I))]
             I = [I[ii] for ii in range(0, len(I)) for jj in range(0,ns[ii])] 
             J = [J[ii] for ii in range(0, len(J)) for jj in range(0,ns[ii])] 
             loc_new = np.random.choice([0,np.pi/2,np.pi,3*np.pi/2],np.sum(Ns_new_invade))
             I1 = I + np.cos(loc_new).astype(int)
             J1 = J + np.sin(loc_new).astype(int)
             Ns_temp1[I1,J1] = Ns_temp1[I1,J1]+ Ns_new_invade[I,J]
        if (np.sum(Nr_new_invade) > 0): # there are invading cells. 
             I, J = np.where(Nr_new_invade > 0)
             ns = [Nr_new_invade[I[ii],J[ii]] for ii in range(0, len(I))]
             I = [I[ii] for ii in range(0, len(I)) for jj in range(0,ns[ii])] 
             J = [J[ii] for ii in range(0, len(J)) for jj in range(0,ns[ii])] 
             loc_new = np.random.choice([0,np.pi/2,np.pi,3*np.pi/2],np.sum(Nr_new_invade))
             I1 = I + np.cos(loc_new).astype(int)
             J1 = J + np.sin(loc_new).astype(int)
             Nr_temp1[I1,J1] = Nr_temp1[I1,J1]+ Nr_new_invade[I,J]  
        
        # death rates
        Nr_dead = np.random.binomial(Nr_temp1[:,:], d0*del_t)
        Ns_dead = np.random.binomial(Ns_temp1[:,:], d0*del_t)
        Ns_temp1[:,:] = Ns_temp1[:,:] - Ns_dead
        Nr_temp1[:,:] = Nr_temp1[:,:] - Nr_dead
                               
        # death rates
        Nr_dead = np.random.binomial(Nr_temp1[:,:], d0*del_t)
        Ns_dead = np.random.binomial(Ns_temp1[:,:], d0*del_t)
        Ns_temp1[:,:] = Ns_temp1[:,:] - Ns_dead
        Nr_temp1[:,:] = Nr_temp1[:,:] - Nr_dead
    
        # update temporary arrays
        if (drug == 1):
            if((np.sum(Ns_temp1[:,:] + Nr_temp1[:,:]))< int(0.5*N_initial)):
                drug = 0
        else:
            if((np.sum(Ns_temp1[:,:] + Nr_temp1[:,:]))> int(N_initial)):
                drug = 1 
        Drug[t] = drug
        Ns_num[t] = np.sum(Ns_temp1)
        Nr_num[t] = np.sum(Nr_temp1)
        # Save distribution every cell cycle
        if (t%100 == 0):
            Ns[:,:,int(t/100)]= Ns_temp1
            Nr[:,:,int(t/100)]= Nr_temp1
            if (np.sum(Ns_temp1 + Nr_temp1) >= 1.25*N_initial): #end the simulations if growth to 1.25 of original size before treatment. 
                break

    f1 = open(filestring, 'wb')
    pickle.dump([Ns[:,:,0:int(t/100)],Nr[:,:,0:int(t/100)], t,Ns_num[0:t], Nr_num[0:t], Drug[0:t],Time, cap, grid_len, r_s, r_R, d0, r_mix_s, r_mix_r,fs, fr, 
                   s_spread, r_spread, r_shift, d_r, d_D], f1)
    f1.close() 
    return Ns[:,:,0:t], Nr[:,:,0:t], Drug[0:t], t


def therapy_CTmol(Time, cap, grid_len, r_s, r_R, d0, r_mix_s, r_mix_r, fs, fr, 
               s_spread, r_spread, r_shift, d_r, d_D, filestring):
    del_t = 0.01
    T = int(Time/(del_t))
    Ns = np.zeros((grid_len, grid_len,Time)).astype(int)
    Nr = np.zeros((grid_len, grid_len,Time)).astype(int)

    # generate initial distribution of cells 
    Ns[int(0.5*grid_len-s_spread/2):int(0.5*grid_len+s_spread/2),int(0.5*grid_len-s_spread/2):int(0.5*grid_len+s_spread/2),0] = np.random.normal(loc=cap*fs, scale=cap*fs*0.2, size=(int(s_spread),int(s_spread))).astype(int)
    pr = fr*(s_spread/r_spread)**2
    Nr[int(r_shift + 0.5*grid_len-r_spread/2):int(r_shift + 0.5*grid_len+r_spread/2),int(r_shift + 0.5*grid_len-r_spread/2):int(r_shift + 0.5*grid_len+r_spread/2),0] = np.random.binomial(Ns[int(0.5*grid_len-r_spread/2):int(0.5*grid_len+r_spread/2),int(0.5*grid_len-r_spread/2):int(0.5*grid_len+r_spread/2),0], pr)
    Ns[:,:,0] = Ns[:,:,0] - Nr[:,:,0]
    N_initial = np.sum(Nr[:,:,0]+Ns[:,:,0])

    Ns_temp1 = np.zeros((grid_len, grid_len)).astype(int)
    Nr_temp1 = np.zeros((grid_len, grid_len)).astype(int)
    Ns_temp1[:,:] = Ns[:,:,0]
    Nr_temp1[:,:] = Nr[:,:,0]
    Drug = 1 # dummy
    Ns_num = np.zeros(T).astype(int)
    Nr_num = np.zeros(T).astype(int)
    
    for t in range(1,T): 
        # growth rates.
        dG = (1 - 0.5*(Ns_temp1[:,:]+Nr_temp1[:,:])/cap)
        dG[np.where(dG < 0)] = 0
        Ns_new = np.random.binomial(Ns_temp1, del_t*r_s*dG)
        Nr_new = np.random.binomial(Nr_temp1, del_t*r_R*dG)
        # killing of sensitive cells by drug. 

        Ns_new_kill = np.random.binomial(Ns_new, d_D)
        Ns_new = Ns_new - Ns_new_kill
        Ns_temp1[:,:] = Ns_temp1[:,:] - Ns_new_kill #substract the cells that die during division.            

        # invasion          
        Ns_new_invade = np.random.binomial(Ns_new, r_mix_s)
        Nr_new_invade = np.random.binomial(Nr_new, r_mix_r)
        Ns_temp1[:,:] = Ns_temp1[:,:] + Ns_new - Ns_new_invade
        Nr_temp1[:,:] = Nr_temp1[:,:] + Nr_new - Nr_new_invade
        # # invasion.
        if (np.sum(Ns_new_invade) > 0): # there are invading cells. 
             I, J = np.where(Ns_new_invade > 0)
             ns = [Ns_new_invade[I[ii],J[ii]] for ii in range(0, len(I))]
             I = [I[ii] for ii in range(0, len(I)) for jj in range(0,ns[ii])] 
             J = [J[ii] for ii in range(0, len(J)) for jj in range(0,ns[ii])] 
             loc_new = np.random.choice([0,np.pi/2,np.pi,3*np.pi/2],np.sum(Ns_new_invade))
             I1 = I + np.cos(loc_new).astype(int)
             J1 = J + np.sin(loc_new).astype(int)
             Ns_temp1[I1,J1] = Ns_temp1[I1,J1]+ Ns_new_invade[I,J]
        if (np.sum(Nr_new_invade) > 0): # there are invading cells. 
             I, J = np.where(Nr_new_invade > 0)
             ns = [Nr_new_invade[I[ii],J[ii]] for ii in range(0, len(I))]
             I = [I[ii] for ii in range(0, len(I)) for jj in range(0,ns[ii])] 
             J = [J[ii] for ii in range(0, len(J)) for jj in range(0,ns[ii])] 
             loc_new = np.random.choice([0,np.pi/2,np.pi,3*np.pi/2],np.sum(Nr_new_invade))
             I1 = I + np.cos(loc_new).astype(int)
             J1 = J + np.sin(loc_new).astype(int)
             Nr_temp1[I1,J1] = Nr_temp1[I1,J1]+ Nr_new_invade[I,J]           
        # death rates
        Nr_dead = np.random.binomial(Nr_temp1[:,:], d0*del_t)
        Ns_dead = np.random.binomial(Ns_temp1[:,:], d0*del_t)
        Ns_temp1[:,:] = Ns_temp1[:,:] - Ns_dead
        Nr_temp1[:,:] = Nr_temp1[:,:] - Nr_dead
    
        # update temporary arrays
        Ns_num[t] = np.sum(Ns_temp1)
        Nr_num[t] = np.sum(Nr_temp1)
        # Save distribution every cell cycle
        if (t%100 == 0):
            Ns[:,:,int(t/100)]= Ns_temp1
            Nr[:,:,int(t/100)]= Nr_temp1
            if (np.sum(Ns_temp1 + Nr_temp1) >= 1.25*N_initial): #end the simulations if growth to 1.25 of original size before treatment. 
                break

    f1 = open(filestring, 'wb')
    pickle.dump([Ns[:,:,0:int(t/100)],Nr[:,:,0:int(t/100)], t,Ns_num[0:t], Nr_num[0:t], Drug,Time, cap, grid_len, r_s, r_R, d0, r_mix_s, r_mix_r, fs, fr, 
                   s_spread, r_spread, r_shift, d_r, d_D], f1)
    f1.close() 
    return Ns[:,:,0:t], Nr[:,:,0:t], Drug, t

#######################################################
### Number of clusters at different location apart. ###
#######################################################

def therapy_AT_2loc(Time, cap, grid_len, r_s, r_R, d0, r_mix, fs, fr, 
               s_spread, r_spread, r_shift, d_r, d_D, filestring):
    del_t = 0.01
    T = int(Time/(del_t))
    Ns = np.zeros((grid_len, grid_len,Time)).astype(int)
    Nr = np.zeros((grid_len, grid_len,Time)).astype(int)

    # generate initial distribution of cells 
    Ns[int(0.5*grid_len-s_spread/2):int(0.5*grid_len+s_spread/2),int(0.5*grid_len-s_spread/2):int(0.5*grid_len+s_spread/2),0] = np.random.normal(loc=cap*fs, scale=cap*fs*0.2, size=(int(s_spread),int(s_spread))).astype(int)
    pr = fr*(s_spread/r_spread)**2
    Nr[int(-r_shift + 0.5*grid_len-r_spread/4):int(-r_shift + 0.5*grid_len+r_spread/4),int(0.5*grid_len-r_spread/2):int(0.5*grid_len+r_spread/2),0] = np.random.binomial(Ns[int(0.5*grid_len-r_spread/4):int(0.5*grid_len+r_spread/4),int(0.5*grid_len-r_spread/2):int(0.5*grid_len+r_spread/2),0], pr)
    Nr[int(r_shift + 0.5*grid_len-r_spread/4):int(r_shift + 0.5*grid_len+r_spread/4),int(0.5*grid_len-r_spread/2):int(0.5*grid_len+r_spread/2),0] = np.random.binomial(Ns[int(0.5*grid_len-r_spread/4):int(0.5*grid_len+r_spread/4),int(0.5*grid_len-r_spread/2):int(0.5*grid_len+r_spread/2),0], pr)

    Ns[:,:,0] = Ns[:,:,0] - Nr[:,:,0]
    N_initial = np.sum(Nr[:,:,0]+Ns[:,:,0])

    Ns_temp1 = np.zeros((grid_len, grid_len)).astype(int)
    Nr_temp1 = np.zeros((grid_len, grid_len)).astype(int)
    Ns_temp1[:,:] = Ns[:,:,0]
    Nr_temp1[:,:] = Nr[:,:,0]
    Drug = np.zeros(T).astype(int)
    Ns_num = np.zeros(T).astype(int)
    Nr_num = np.zeros(T).astype(int)
    Drug[0] = 1
    drug = 1
    
    for t in range(1,T): 
        # growth rates.
        dG = (1 - 0.5*(Ns_temp1[:,:]+Nr_temp1[:,:])/cap)
        dG[np.where(dG < 0)] = 0 # make sure growth doesn't get negative number.
        Ns_new = np.random.binomial(Ns_temp1, del_t*r_s*dG)
        Nr_new = np.random.binomial(Nr_temp1, del_t*r_R*dG)
        # killing of sensitive cells by drug. 
        if drug == 1:
            Ns_new_kill = np.random.binomial(Ns_new, d_D)
            Ns_new = Ns_new - Ns_new_kill
            Ns_temp1[:,:] = Ns_temp1[:,:] - Ns_new_kill #substract the cells that die during division.            
            
        # invasion          
        Ns_new_invade = np.random.binomial(Ns_new, r_mix)
        Nr_new_invade = np.random.binomial(Nr_new, r_mix)
        Ns_temp1[:,:] = Ns_temp1[:,:] + Ns_new - Ns_new_invade
        Nr_temp1[:,:] = Nr_temp1[:,:] + Nr_new - Nr_new_invade
        # # invasion.
        if (np.sum(Ns_new_invade) > 0): # there are invading cells. 
             I, J = np.where(Ns_new_invade > 0)
             ns = [Ns_new_invade[I[ii],J[ii]] for ii in range(0, len(I))]
             I = [I[ii] for ii in range(0, len(I)) for jj in range(0,ns[ii])] 
             J = [J[ii] for ii in range(0, len(J)) for jj in range(0,ns[ii])] 
             loc_new = np.random.choice([0,np.pi/2,np.pi,3*np.pi/2],np.sum(Ns_new_invade))
             I1 = I + np.cos(loc_new).astype(int)
             J1 = J + np.sin(loc_new).astype(int)
             Ns_temp1[I1,J1] = Ns_temp1[I1,J1]+ Ns_new_invade[I,J]
        if (np.sum(Nr_new_invade) > 0): # there are invading cells. 
             I, J = np.where(Nr_new_invade > 0)
             ns = [Nr_new_invade[I[ii],J[ii]] for ii in range(0, len(I))]
             I = [I[ii] for ii in range(0, len(I)) for jj in range(0,ns[ii])] 
             J = [J[ii] for ii in range(0, len(J)) for jj in range(0,ns[ii])] 
             loc_new = np.random.choice([0,np.pi/2,np.pi,3*np.pi/2],np.sum(Nr_new_invade))
             I1 = I + np.cos(loc_new).astype(int)
             J1 = J + np.sin(loc_new).astype(int)
             Nr_temp1[I1,J1] = Nr_temp1[I1,J1]+ Nr_new_invade[I,J]  
        
        # death rates
        Nr_dead = np.random.binomial(Nr_temp1[:,:], d0*del_t)
        Ns_dead = np.random.binomial(Ns_temp1[:,:], d0*del_t)
        Ns_temp1[:,:] = Ns_temp1[:,:] - Ns_dead
        Nr_temp1[:,:] = Nr_temp1[:,:] - Nr_dead
                               
        # death rates
        Nr_dead = np.random.binomial(Nr_temp1[:,:], d0*del_t)
        Ns_dead = np.random.binomial(Ns_temp1[:,:], d0*del_t)
        Ns_temp1[:,:] = Ns_temp1[:,:] - Ns_dead
        Nr_temp1[:,:] = Nr_temp1[:,:] - Nr_dead
    
        # update temporary arrays
        if (drug == 1):
            if((np.sum(Ns_temp1[:,:] + Nr_temp1[:,:]))< int(0.5*N_initial)):
                drug = 0
        else:
            if((np.sum(Ns_temp1[:,:] + Nr_temp1[:,:]))> int(N_initial)):
                drug = 1 
        Drug[t] = drug
        Ns_num[t] = np.sum(Ns_temp1)
        Nr_num[t] = np.sum(Nr_temp1)
        # Save distribution every cell cycle
        if (t%100 == 0):
            Ns[:,:,int(t/100)]= Ns_temp1
            Nr[:,:,int(t/100)]= Nr_temp1
            if (np.sum(Ns_temp1 + Nr_temp1) >= 1.25*N_initial): #end the simulations if growth to 1.25 of original size before treatment. 
                break

    f1 = open(filestring, 'wb')
    pickle.dump([Ns[:,:,0:int(t/100)],Nr[:,:,0:int(t/100)], t,Ns_num[0:t], Nr_num[0:t], Drug[0:t],Time, cap, grid_len, r_s, r_R, d0, r_mix, fs, fr, 
                   s_spread, r_spread, r_shift, d_r, d_D], f1)
    f1.close() 
    return Ns[:,:,0:t], Nr[:,:,0:t], Drug[0:t], t


def therapy_CT_2loc(Time, cap, grid_len, r_s, r_R, d0, r_mix, fs, fr, 
               s_spread, r_spread, r_shift, d_r, d_D, filestring):
    del_t = 0.01
    T = int(Time/(del_t))
    Ns = np.zeros((grid_len, grid_len,Time)).astype(int)
    Nr = np.zeros((grid_len, grid_len,Time)).astype(int)

    # generate initial distribution of cells 
    Ns[int(0.5*grid_len-s_spread/2):int(0.5*grid_len+s_spread/2),int(0.5*grid_len-s_spread/2):int(0.5*grid_len+s_spread/2),0] = np.random.normal(loc=cap*fs, scale=cap*fs*0.2, size=(int(s_spread),int(s_spread))).astype(int)
    pr = fr*(s_spread/r_spread)**2
    Nr[int(-r_shift + 0.5*grid_len-r_spread/4):int(-r_shift + 0.5*grid_len+r_spread/4),int(0.5*grid_len-r_spread/2):int(0.5*grid_len+r_spread/2),0] = np.random.binomial(Ns[int(0.5*grid_len-r_spread/4):int(0.5*grid_len+r_spread/4),int(0.5*grid_len-r_spread/2):int(0.5*grid_len+r_spread/2),0], pr)
    Nr[int(r_shift + 0.5*grid_len-r_spread/4):int(r_shift + 0.5*grid_len+r_spread/4),int(0.5*grid_len-r_spread/2):int(0.5*grid_len+r_spread/2),0] = np.random.binomial(Ns[int(0.5*grid_len-r_spread/4):int(0.5*grid_len+r_spread/4),int(0.5*grid_len-r_spread/2):int(0.5*grid_len+r_spread/2),0], pr)

    Ns[:,:,0] = Ns[:,:,0] - Nr[:,:,0]
    N_initial = np.sum(Nr[:,:,0]+Ns[:,:,0])

    Ns_temp1 = np.zeros((grid_len, grid_len)).astype(int)
    Nr_temp1 = np.zeros((grid_len, grid_len)).astype(int)
    Ns_temp1[:,:] = Ns[:,:,0]
    Nr_temp1[:,:] = Nr[:,:,0]
    Drug = 1 # dummy
    Ns_num = np.zeros(T).astype(int)
    Nr_num = np.zeros(T).astype(int)
    
    for t in range(1,T): 
        # growth rates.
        dG = (1 - 0.5*(Ns_temp1[:,:]+Nr_temp1[:,:])/cap)
        dG[np.where(dG < 0)] = 0
        Ns_new = np.random.binomial(Ns_temp1, del_t*r_s*dG)
        Nr_new = np.random.binomial(Nr_temp1, del_t*r_R*dG)
        # killing of sensitive cells by drug. 

        Ns_new_kill = np.random.binomial(Ns_new, d_D)
        Ns_new = Ns_new - Ns_new_kill
        Ns_temp1[:,:] = Ns_temp1[:,:] - Ns_new_kill #substract the cells that die during division.            

        # invasion          
        Ns_new_invade = np.random.binomial(Ns_new, r_mix)
        Nr_new_invade = np.random.binomial(Nr_new, r_mix)
        Ns_temp1[:,:] = Ns_temp1[:,:] + Ns_new - Ns_new_invade
        Nr_temp1[:,:] = Nr_temp1[:,:] + Nr_new - Nr_new_invade
        # # invasion.
        if (np.sum(Ns_new_invade) > 0): # there are invading cells. 
             I, J = np.where(Ns_new_invade > 0)
             ns = [Ns_new_invade[I[ii],J[ii]] for ii in range(0, len(I))]
             I = [I[ii] for ii in range(0, len(I)) for jj in range(0,ns[ii])] 
             J = [J[ii] for ii in range(0, len(J)) for jj in range(0,ns[ii])] 
             loc_new = np.random.choice([0,np.pi/2,np.pi,3*np.pi/2],np.sum(Ns_new_invade))
             I1 = I + np.cos(loc_new).astype(int)
             J1 = J + np.sin(loc_new).astype(int)
             Ns_temp1[I1,J1] = Ns_temp1[I1,J1]+ Ns_new_invade[I,J]
        if (np.sum(Nr_new_invade) > 0): # there are invading cells. 
             I, J = np.where(Nr_new_invade > 0)
             ns = [Nr_new_invade[I[ii],J[ii]] for ii in range(0, len(I))]
             I = [I[ii] for ii in range(0, len(I)) for jj in range(0,ns[ii])] 
             J = [J[ii] for ii in range(0, len(J)) for jj in range(0,ns[ii])] 
             loc_new = np.random.choice([0,np.pi/2,np.pi,3*np.pi/2],np.sum(Nr_new_invade))
             I1 = I + np.cos(loc_new).astype(int)
             J1 = J + np.sin(loc_new).astype(int)
             Nr_temp1[I1,J1] = Nr_temp1[I1,J1]+ Nr_new_invade[I,J]           
        # death rates
        Nr_dead = np.random.binomial(Nr_temp1[:,:], d0*del_t)
        Ns_dead = np.random.binomial(Ns_temp1[:,:], d0*del_t)
        Ns_temp1[:,:] = Ns_temp1[:,:] - Ns_dead
        Nr_temp1[:,:] = Nr_temp1[:,:] - Nr_dead
    
        # update temporary arrays
        Ns_num[t] = np.sum(Ns_temp1)
        Nr_num[t] = np.sum(Nr_temp1)
        # Save distribution every cell cycle
        if (t%100 == 0):
            Ns[:,:,int(t/100)]= Ns_temp1
            Nr[:,:,int(t/100)]= Nr_temp1
            if (np.sum(Ns_temp1 + Nr_temp1) >= 1.25*N_initial): #end the simulations if growth to 1.25 of original size before treatment. 
                break

    f1 = open(filestring, 'wb')
    pickle.dump([Ns[:,:,0:int(t/100)],Nr[:,:,0:int(t/100)], t,Ns_num[0:t], Nr_num[0:t], Drug,Time, cap, grid_len, r_s, r_R, d0, r_mix, fs, fr, 
                   s_spread, r_spread, r_shift, d_r, d_D], f1)
    f1.close() 
    return Ns[:,:,0:t], Nr[:,:,0:t], Drug, t

# 4 location
def therapy_AT_4loc(Time, cap, grid_len, r_s, r_R, d0, r_mix, fs, fr, 
               s_spread, r_spread, r_shift, d_r, d_D, filestring):
    del_t = 0.01
    T = int(Time/(del_t))
    Ns = np.zeros((grid_len, grid_len,Time)).astype(int)
    Nr = np.zeros((grid_len, grid_len,Time)).astype(int)

    # generate initial distribution of cells 
    Ns[int(0.5*grid_len-s_spread/2):int(0.5*grid_len+s_spread/2),int(0.5*grid_len-s_spread/2):int(0.5*grid_len+s_spread/2),0] = np.random.normal(loc=cap*fs, scale=cap*fs*0.2, size=(int(s_spread),int(s_spread))).astype(int)
    pr = fr*(s_spread/r_spread)**2
    Nr[int(-r_shift + 0.5*grid_len-r_spread/4):int(-r_shift + 0.5*grid_len+r_spread/4),int(-r_shift + 0.5*grid_len-r_spread/4):int(-r_shift + 0.5*grid_len+r_spread/4),0] = np.random.binomial(Ns[int(0.5*grid_len-r_spread/4):int(0.5*grid_len+r_spread/4),int(0.5*grid_len-r_spread/4):int(0.5*grid_len+r_spread/4),0], pr)
    Nr[int(r_shift + 0.5*grid_len-r_spread/4):int(r_shift + 0.5*grid_len+r_spread/4),int(-r_shift + 0.5*grid_len-r_spread/4):int(-r_shift + 0.5*grid_len+r_spread/4),0] = np.random.binomial(Ns[int(0.5*grid_len-r_spread/4):int(0.5*grid_len+r_spread/4),int(0.5*grid_len-r_spread/4):int(0.5*grid_len+r_spread/4),0], pr)
    Nr[int(-r_shift + 0.5*grid_len-r_spread/4):int(-r_shift + 0.5*grid_len+r_spread/4),int(r_shift + 0.5*grid_len-r_spread/4):int(r_shift + 0.5*grid_len+r_spread/4),0] = np.random.binomial(Ns[int(0.5*grid_len-r_spread/4):int(0.5*grid_len+r_spread/4),int(0.5*grid_len-r_spread/4):int(0.5*grid_len+r_spread/4),0], pr)
    Nr[int(r_shift + 0.5*grid_len-r_spread/4):int(r_shift + 0.5*grid_len+r_spread/4),int(r_shift + 0.5*grid_len-r_spread/4):int(r_shift + 0.5*grid_len+r_spread/4),0] = np.random.binomial(Ns[int(0.5*grid_len-r_spread/4):int(0.5*grid_len+r_spread/4),int(0.5*grid_len-r_spread/4):int(0.5*grid_len+r_spread/4),0], pr)

    Ns[:,:,0] = Ns[:,:,0] - Nr[:,:,0]
    N_initial = np.sum(Nr[:,:,0]+Ns[:,:,0])

    Ns_temp1 = np.zeros((grid_len, grid_len)).astype(int)
    Nr_temp1 = np.zeros((grid_len, grid_len)).astype(int)
    Ns_temp1[:,:] = Ns[:,:,0]
    Nr_temp1[:,:] = Nr[:,:,0]
    Drug = np.zeros(T).astype(int)
    Ns_num = np.zeros(T).astype(int)
    Nr_num = np.zeros(T).astype(int)
    Drug[0] = 1
    drug = 1
    
    for t in range(1,T): 
        # growth rates.
        dG = (1 - 0.5*(Ns_temp1[:,:]+Nr_temp1[:,:])/cap)
        dG[np.where(dG < 0)] = 0 # make sure growth doesn't get negative number.
        Ns_new = np.random.binomial(Ns_temp1, del_t*r_s*dG)
        Nr_new = np.random.binomial(Nr_temp1, del_t*r_R*dG)
        # killing of sensitive cells by drug. 
        if drug == 1:
            Ns_new_kill = np.random.binomial(Ns_new, d_D)
            Ns_new = Ns_new - Ns_new_kill
            Ns_temp1[:,:] = Ns_temp1[:,:] - Ns_new_kill #substract the cells that die during division.            
            
        # invasion          
        Ns_new_invade = np.random.binomial(Ns_new, r_mix)
        Nr_new_invade = np.random.binomial(Nr_new, r_mix)
        Ns_temp1[:,:] = Ns_temp1[:,:] + Ns_new - Ns_new_invade
        Nr_temp1[:,:] = Nr_temp1[:,:] + Nr_new - Nr_new_invade
        # # invasion.
        if (np.sum(Ns_new_invade) > 0): # there are invading cells. 
             I, J = np.where(Ns_new_invade > 0)
             ns = [Ns_new_invade[I[ii],J[ii]] for ii in range(0, len(I))]
             I = [I[ii] for ii in range(0, len(I)) for jj in range(0,ns[ii])] 
             J = [J[ii] for ii in range(0, len(J)) for jj in range(0,ns[ii])] 
             loc_new = np.random.choice([0,np.pi/2,np.pi,3*np.pi/2],np.sum(Ns_new_invade))
             I1 = I + np.cos(loc_new).astype(int)
             J1 = J + np.sin(loc_new).astype(int)
             Ns_temp1[I1,J1] = Ns_temp1[I1,J1]+ Ns_new_invade[I,J]
        if (np.sum(Nr_new_invade) > 0): # there are invading cells. 
             I, J = np.where(Nr_new_invade > 0)
             ns = [Nr_new_invade[I[ii],J[ii]] for ii in range(0, len(I))]
             I = [I[ii] for ii in range(0, len(I)) for jj in range(0,ns[ii])] 
             J = [J[ii] for ii in range(0, len(J)) for jj in range(0,ns[ii])] 
             loc_new = np.random.choice([0,np.pi/2,np.pi,3*np.pi/2],np.sum(Nr_new_invade))
             I1 = I + np.cos(loc_new).astype(int)
             J1 = J + np.sin(loc_new).astype(int)
             Nr_temp1[I1,J1] = Nr_temp1[I1,J1]+ Nr_new_invade[I,J]  
        
        # death rates
        Nr_dead = np.random.binomial(Nr_temp1[:,:], d0*del_t)
        Ns_dead = np.random.binomial(Ns_temp1[:,:], d0*del_t)
        Ns_temp1[:,:] = Ns_temp1[:,:] - Ns_dead
        Nr_temp1[:,:] = Nr_temp1[:,:] - Nr_dead
                               
        # death rates
        Nr_dead = np.random.binomial(Nr_temp1[:,:], d0*del_t)
        Ns_dead = np.random.binomial(Ns_temp1[:,:], d0*del_t)
        Ns_temp1[:,:] = Ns_temp1[:,:] - Ns_dead
        Nr_temp1[:,:] = Nr_temp1[:,:] - Nr_dead
    
        # update temporary arrays
        if (drug == 1):
            if((np.sum(Ns_temp1[:,:] + Nr_temp1[:,:]))< int(0.5*N_initial)):
                drug = 0
        else:
            if((np.sum(Ns_temp1[:,:] + Nr_temp1[:,:]))> int(N_initial)):
                drug = 1 
        Drug[t] = drug
        Ns_num[t] = np.sum(Ns_temp1)
        Nr_num[t] = np.sum(Nr_temp1)
        # Save distribution every cell cycle
        if (t%100 == 0):
            Ns[:,:,int(t/100)]= Ns_temp1
            Nr[:,:,int(t/100)]= Nr_temp1
            if (np.sum(Ns_temp1 + Nr_temp1) >= 1.25*N_initial): #end the simulations if growth to 1.25 of original size before treatment. 
                break

    f1 = open(filestring, 'wb')
    pickle.dump([Ns[:,:,0:int(t/100)],Nr[:,:,0:int(t/100)], t,Ns_num[0:t], Nr_num[0:t], Drug[0:t],Time, cap, grid_len, r_s, r_R, d0, r_mix, fs, fr, 
                   s_spread, r_spread, r_shift, d_r, d_D], f1)
    f1.close() 
    return Ns[:,:,0:t], Nr[:,:,0:t], Drug[0:t], t


def therapy_CT_4loc(Time, cap, grid_len, r_s, r_R, d0, r_mix, fs, fr, 
               s_spread, r_spread, r_shift, d_r, d_D, filestring):
    del_t = 0.01
    T = int(Time/(del_t))
    Ns = np.zeros((grid_len, grid_len,Time)).astype(int)
    Nr = np.zeros((grid_len, grid_len,Time)).astype(int)

    # generate initial distribution of cells 
    Ns[int(0.5*grid_len-s_spread/2):int(0.5*grid_len+s_spread/2),int(0.5*grid_len-s_spread/2):int(0.5*grid_len+s_spread/2),0] = np.random.normal(loc=cap*fs, scale=cap*fs*0.2, size=(int(s_spread),int(s_spread))).astype(int)
    pr = fr*(s_spread/r_spread)**2
    Nr[int(-r_shift + 0.5*grid_len-r_spread/4):int(-r_shift + 0.5*grid_len+r_spread/4),int(-r_shift + 0.5*grid_len-r_spread/4):int(-r_shift + 0.5*grid_len+r_spread/4),0] = np.random.binomial(Ns[int(0.5*grid_len-r_spread/4):int(0.5*grid_len+r_spread/4),int(0.5*grid_len-r_spread/4):int(0.5*grid_len+r_spread/4),0], pr)
    Nr[int(r_shift + 0.5*grid_len-r_spread/4):int(r_shift + 0.5*grid_len+r_spread/4),int(-r_shift + 0.5*grid_len-r_spread/4):int(-r_shift + 0.5*grid_len+r_spread/4),0] = np.random.binomial(Ns[int(0.5*grid_len-r_spread/4):int(0.5*grid_len+r_spread/4),int(0.5*grid_len-r_spread/4):int(0.5*grid_len+r_spread/4),0], pr)
    Nr[int(-r_shift + 0.5*grid_len-r_spread/4):int(-r_shift + 0.5*grid_len+r_spread/4),int(r_shift + 0.5*grid_len-r_spread/4):int(r_shift + 0.5*grid_len+r_spread/4),0] = np.random.binomial(Ns[int(0.5*grid_len-r_spread/4):int(0.5*grid_len+r_spread/4),int(0.5*grid_len-r_spread/4):int(0.5*grid_len+r_spread/4),0], pr)
    Nr[int(r_shift + 0.5*grid_len-r_spread/4):int(r_shift + 0.5*grid_len+r_spread/4),int(r_shift + 0.5*grid_len-r_spread/4):int(r_shift + 0.5*grid_len+r_spread/4),0] = np.random.binomial(Ns[int(0.5*grid_len-r_spread/4):int(0.5*grid_len+r_spread/4),int(0.5*grid_len-r_spread/4):int(0.5*grid_len+r_spread/4),0], pr)

    Ns[:,:,0] = Ns[:,:,0] - Nr[:,:,0]
    N_initial = np.sum(Nr[:,:,0]+Ns[:,:,0])

    Ns_temp1 = np.zeros((grid_len, grid_len)).astype(int)
    Nr_temp1 = np.zeros((grid_len, grid_len)).astype(int)
    Ns_temp1[:,:] = Ns[:,:,0]
    Nr_temp1[:,:] = Nr[:,:,0]
    Drug = 1 # dummy
    Ns_num = np.zeros(T).astype(int)
    Nr_num = np.zeros(T).astype(int)
    
    for t in range(1,T): 
        # growth rates.
        dG = (1 - 0.5*(Ns_temp1[:,:]+Nr_temp1[:,:])/cap)
        dG[np.where(dG < 0)] = 0
        Ns_new = np.random.binomial(Ns_temp1, del_t*r_s*dG)
        Nr_new = np.random.binomial(Nr_temp1, del_t*r_R*dG)
        # killing of sensitive cells by drug. 

        Ns_new_kill = np.random.binomial(Ns_new, d_D)
        Ns_new = Ns_new - Ns_new_kill
        Ns_temp1[:,:] = Ns_temp1[:,:] - Ns_new_kill #substract the cells that die during division.            

        # invasion          
        Ns_new_invade = np.random.binomial(Ns_new, r_mix)
        Nr_new_invade = np.random.binomial(Nr_new, r_mix)
        Ns_temp1[:,:] = Ns_temp1[:,:] + Ns_new - Ns_new_invade
        Nr_temp1[:,:] = Nr_temp1[:,:] + Nr_new - Nr_new_invade
        # # invasion.
        if (np.sum(Ns_new_invade) > 0): # there are invading cells. 
             I, J = np.where(Ns_new_invade > 0)
             ns = [Ns_new_invade[I[ii],J[ii]] for ii in range(0, len(I))]
             I = [I[ii] for ii in range(0, len(I)) for jj in range(0,ns[ii])] 
             J = [J[ii] for ii in range(0, len(J)) for jj in range(0,ns[ii])] 
             loc_new = np.random.choice([0,np.pi/2,np.pi,3*np.pi/2],np.sum(Ns_new_invade))
             I1 = I + np.cos(loc_new).astype(int)
             J1 = J + np.sin(loc_new).astype(int)
             Ns_temp1[I1,J1] = Ns_temp1[I1,J1]+ Ns_new_invade[I,J]
        if (np.sum(Nr_new_invade) > 0): # there are invading cells. 
             I, J = np.where(Nr_new_invade > 0)
             ns = [Nr_new_invade[I[ii],J[ii]] for ii in range(0, len(I))]
             I = [I[ii] for ii in range(0, len(I)) for jj in range(0,ns[ii])] 
             J = [J[ii] for ii in range(0, len(J)) for jj in range(0,ns[ii])] 
             loc_new = np.random.choice([0,np.pi/2,np.pi,3*np.pi/2],np.sum(Nr_new_invade))
             I1 = I + np.cos(loc_new).astype(int)
             J1 = J + np.sin(loc_new).astype(int)
             Nr_temp1[I1,J1] = Nr_temp1[I1,J1]+ Nr_new_invade[I,J]           
        # death rates
        Nr_dead = np.random.binomial(Nr_temp1[:,:], d0*del_t)
        Ns_dead = np.random.binomial(Ns_temp1[:,:], d0*del_t)
        Ns_temp1[:,:] = Ns_temp1[:,:] - Ns_dead
        Nr_temp1[:,:] = Nr_temp1[:,:] - Nr_dead
    
        # update temporary arrays
        Ns_num[t] = np.sum(Ns_temp1)
        Nr_num[t] = np.sum(Nr_temp1)
        # Save distribution every cell cycle
        if (t%100 == 0):
            Ns[:,:,int(t/100)]= Ns_temp1
            Nr[:,:,int(t/100)]= Nr_temp1
            if (np.sum(Ns_temp1 + Nr_temp1) >= 1.25*N_initial): #end the simulations if growth to 1.25 of original size before treatment. 
                break

    f1 = open(filestring, 'wb')
    pickle.dump([Ns[:,:,0:int(t/100)],Nr[:,:,0:int(t/100)], t,Ns_num[0:t], Nr_num[0:t], Drug,Time, cap, grid_len, r_s, r_R, d0, r_mix, fs, fr, 
                   s_spread, r_spread, r_shift, d_r, d_D], f1)
    f1.close() 
    return Ns[:,:,0:t], Nr[:,:,0:t], Drug, t