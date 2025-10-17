import numpy as np
from scipy.optimize import fsolve

def charnock(which_stress,U_10N,swh,ustar,dcp):
    """
    Charnock variable.  Several methods and versions are available.
    Parameters:
        which_stress - how to calculate Charnock variable.  Options are:
            'C3.6_Wi' - COARE 3.6, windspeed dependent model
            'C3.6_SS' - COARE 3.6, seastate/wave-age dependent model
        U_10N - 10m neutral windspeed [m s-1]
        swh - significant wave height [m]
        ustar - friction velocity [m s-1]
        dcp - dominant phase speed [m s-1]
    Return:
        alpha - Charnock variable [-]
    """
    if which_stress == 'C3.6_Wi':    # Uses COARE 3.5 values
        umax = 19    # [m s-1]
        a1 = 0.0017    # [m-1 s]
        a2 = -0.005    # [-]
        alpha = a1*U_10N + a2
        alpha[U_10N > umax] = a1*umax + a2
    elif which_stress == 'C3.6_SS':    # Uses COARE 3.5 values
        Ad = 0.091    # [-]
        Bd = 2.0    # [-]
        g = 9.81    # Acceleration due to gravity [m s-2]
        z0rough = swh*Ad*(ustar/dcp)**Bd    # Momentum roughness contributed by ocean waves [m]
        alpha = z0rough*g/ustar**2
    return alpha

def fall_velocity_PK97(r):
    """
    Fall velocity of spherical droplets, based on Pruppacher and Klett (1997) section 10.3.6.
    Parameters:
        r - droplet radius [m]
    Return:
        v_fall - settling velocity [m s-1]
    """
    nu_a = 1.5e-5    # Kinematic viscosity of air [m2 s-1]
    rho_a = 1.25    # Density of air [kg m-3]
    rho_w = 1030    # Density of seawater [kg m-3]
    g = 9.81    # Acceleration due to gravity [m s-2]
    sigma_aw = 7.4e-2    # Surface tension of air-water interface [N m-1]
    v_stokes = 2*r**2*g*(rho_w - rho_a)/9/(rho_a*nu_a)    # Stokes velocity [m s-1]
    
    # r < 10 micrometers
    lamda_a0 = 6.6e-8    # Mean free path at 1013.25 mb, 293.15 K [m]
    f_slip = 1 + 1.26*lamda_a0/r    # Slip flow Cunningham correction factor [-]
    v_small = f_slip*v_stokes    # Settling velocity for r < 10 micrometers [m s-1]
    
    # 10 <= r <= 535 micrometers
    CdNRe2 = 32*r**3*(rho_w - rho_a)/rho_a/nu_a**2*g/3    # Product of drag coefficient and square of Reynolds number [-]
    X = np.log(CdNRe2)    # X in polynomial curve fit [-]
    B = np.array([-0.318657e1, 0.992696, -0.153193e-2, -0.987059e-3, -0.578878e-3, 0.855176e-4, -0.327815e-5])    # Polynomial coefficients [-]
    Y = B[0] + B[1]*X + B[2]*X**2 + B[3]*X**3 + B[4]*X**4 + B[5]*X**5 + B[6]*X**6    # Y in polynomial curve fit [-]
    NRe = np.exp(Y)    # Reynolds number [-]
    v_med = nu_a*NRe/2/r    # Settling velocity for 10 <= r <= 535 micrometers [m s-1]
    
    # r > 535 micrometers
    NBo = g*(rho_w - rho_a)*r**2/sigma_aw    # Bond number [-]
    NP = sigma_aw**3/rho_a**2/nu_a**4/g/(rho_w - rho_a)    # Physical property number [-]
    NBoNP16 = NBo*NP**(1/6)    # Product of Bond number and physical property number to the 1/6 power [-]
    X = np.log(16/3*NBoNP16)    # X in polynomial curve fit [-]
    B = np.array([-0.500015e1, 0.523778e1, -0.204914e1, 0.475294, -0.542819e-1, 0.238449e-2])    # Polynomial coefficients [-]
    Y = B[0] + B[1]*X + B[2]*X**2 + B[3]*X**3 + B[4]*X**4 + B[5]*X**5    # Y in polynomial curve fit [-]
    NRe = NP**(1/6)*np.exp(Y)    # Reynolds number [-]
    v_large = nu_a*NRe/2/r    # Settling velocity for r > 535 micrometers [m s-1]
    
    # Connect regimes
    v_fall = v_med
    v_fall[r < 10e-6] = v_small[r < 10e-6]
    v_fall[r > 535e-6] = v_large[r > 535e-6]
    return v_fall

def qsat0(T_K,P):
    """
    Saturation specific humidity over a plane surface of pure water, with e_sat0 per Buck (1981) Eq 8.
    Parameters:
        T_K - temperature [K]
        P - pressure [Pa]
    Return:
        q_sat0 - saturation specific humidity [kg kg-1]
    """
    e_sat0 = 6.1121*np.exp(17.502*(T_K - 273.15)/(T_K - 273.15 + 240.97))*(1.0007 + 3.46e-8*P)*1e2    # Sat vap press [Pa]
    q_sat0 = e_sat0*0.622/(P - 0.378*e_sat0)    # Saturation specific humidity [kg kg-1]
    return q_sat0

def satratio(T_K,P,q,max_satratio):
    """
    Saturation ratio, with e_sat0 per Buck (1981) Eq 8.  We assume that s = q/qsat, rather than w/wsat, 
    so that air with q = qsat will be exactly saturated.
    Parameters:
        T_K - temperature [K]
        P - pressure [Pa]
        q - specific humidity [kg kg-1]
        max_satratio - maximum allowable saturation ratio [-]
    Return:
        s - saturation ratio [-]
    """
    q_sat0 = qsat0(T_K,P)    # Saturation specific humidity [kg kg-1]
    s = np.minimum(q/q_sat0,max_satratio)    # Saturation ratio [-]
    return s   

def stabIntM(zeta):
    """
    Integrated stability function for momentum, evaluated at zeta.
    Per COARE 3.6 Matlab code.
    Parameters:
        zeta - stability parameter [-]
    Return:
        psiM - integrated stability function at zeta [-]
    """
    psiM = np.full_like(zeta,np.nan)
    # Neutral
    psiM[zeta == 0] = 0
    # Unstable
    Xk = (1 - 16*zeta[zeta < 0])**0.25    # For small negative zeta
    psik = 2*np.log((1 + Xk)/2) + np.log((1 + Xk**2)/2) - 2*np.arctan(Xk) + 2*np.arctan(1)
    Xc = (1 - 10.15*zeta[zeta < 0])**(1/3)    # For large negative zeta
    psic = 1.5*np.log((1 + Xc + Xc**2)/3) - np.sqrt(3)*np.arctan((1 + 2*Xc)/np.sqrt(3)) + 4*np.arctan(1)/np.sqrt(3)
    f = zeta[zeta < 0]**2/(1 + zeta[zeta < 0]**2)
    psiM[zeta < 0] = (1 - f)*psik + f*psic
    # Stable (Grachev 2007, Eq. 12)
    am = 5
    bm = am/6.5
    Bm = ((1-bm)/bm)**(1/3)
    X = (1 + zeta[zeta > 0])**(1/3)
    psiM[zeta > 0] = -(3*am/bm)*(X-1) + ((am*Bm)/(2*bm))*(2*np.log((Bm+X)/(Bm+1)) - \
        np.log((Bm**2-Bm*X+X**2)/(Bm**2-Bm+1)) + 2*np.sqrt(3)*np.arctan((2*X-Bm)/(Bm*np.sqrt(3))) - \
        2*np.sqrt(3)*np.arctan((2-Bm)/(Bm*np.sqrt(3))))
    return psiM

def stabIntH(zeta):
    """
    Integrated stability function for heat, evaluated at zeta.
    Per COARE 3.6 Matlab code.
    Parameters:
        zeta - stability parameter [-]
    Return:
        psiH - integrated stability function at zeta [-]
    """
    psiH = np.full_like(zeta,np.nan)
    # Neutral
    psiH[zeta == 0] = 0
    # Unstable
    Yk = (1 - 16*zeta[zeta < 0])**0.5    # For small negative zeta
    psik = 2*np.log((1 + Yk)/2)
    Yc = (1 - 34.15*zeta[zeta < 0])**(1/3)    # For large negative zeta
    psic = 1.5*np.log((1 + Yc + Yc**2)/3) - np.sqrt(3)*np.arctan((1 + 2*Yc)/np.sqrt(3)) + 4*np.arctan(1)/np.sqrt(3)
    f = zeta[zeta < 0]**2/(1 + zeta[zeta < 0]**2)
    psiH[zeta < 0] = (1 - f)*psik + f*psic
    # Stable (Grachev 2007, Eq. 13)
    a = 5
    b = 5
    c = 3
    B = np.sqrt(c**2 - 4)
    psiH[zeta > 0] = -(b/2)*np.log(1+c*zeta[zeta > 0]+zeta[zeta > 0]**2) + \
        (((b*c)/(2*B))-(a/B))*(np.log((2*zeta[zeta > 0]+c-B)/(2*zeta[zeta > 0]+c+B))-np.log((c-B)/(c+B)))
    return psiH

def stabIntSprayH(zeta):
    """
    Integrated stability function for heat within spray layer, evaluated at zeta.
    Uses classical values for stability functions (gamma = 16, beta = 5, e.g. Dyer 1974).
    Parameters:
        zeta - stability parameter [-]
    Return:
        phisprH - integrated stability function within spray layer at zeta [-]
    """
    phisprH = np.full_like(zeta,np.nan)
    phisprH[zeta == 0] = 0
    phisprH[zeta > 0] = -2.5*zeta[zeta > 0]
    Y = (1 - 16*zeta[zeta < 0])**0.5
    phisprH[zeta < 0] = -(Y - 1)**2/16/zeta[zeta < 0]
    return phisprH

def thermo_HTspr(zT,tauf,tauT,p_0,gammaWB,y0,t_0,rho_a,delspr,z0t,z0q,L,\
        th_0,q_0,G_S,G_L,H_S0,H_L0,H_Sspr,H_Rspr,H_Lspr,Lv,cp_a,g):
    """
    Droplet thermodynamic calculations for spray heat flux due to temp change.
    Performs calculations for the entire droplet size vector.
    """
    p_zT = p_0 - rho_a*g*zT    # Pressure at zT [Pa]
    t_zT = (th_0 - 1/G_S*(H_S0*(np.log((z0t+zT)/z0t) - stabIntH((z0t+zT)/L)) \
            + zT/delspr*(1 - stabIntSprayH((z0t+zT)/L))*(H_Sspr - H_Rspr)))*(p_zT/1e5)**0.286    # Temp at zT [K]
    q_zT =   q_0 - 1/G_L*(H_L0*(np.log((z0q+zT)/z0q) - stabIntH((z0q+zT)/L)) \
            + zT/delspr*(1 - stabIntSprayH((z0q+zT)/L))*H_Lspr)    # Spec hum at zT [kg kg-1]
    qsat0_zT = qsat0(t_zT,p_zT)    # Saturation specific humidity at zT [kg kg-1]
    betaWB_zT = 1/(1 + Lv*gammaWB*(1 + y0)/cp_a*qsat0_zT)    # Wetbulb coefficient at zT [-]
    s_zT = satratio(t_zT,p_zT,q_zT,0.99999)    # Saturation ratio at zT [-]
    wetdep_zT = (1 - s_zT/(1 + y0))*(1 - betaWB_zT)/gammaWB    # Wetbulb depression at zT [-]
    tWB_zT = t_zT - wetdep_zT    # Wetbulb temperature at zT [K]
    tdropf = tWB_zT + (t_0 - tWB_zT)*np.exp(-tauf/tauT)    # Final droplet temperature [K]
    return (t_zT,wetdep_zT,tWB_zT,tdropf)

def thermo_HRspr(r0_rng,r0,v_g,t_zR_ig,q_zR_ig,Fp,tauf,p_0,gammaWB,y0,t_0,rho_a,Dv_a,xs,delspr,z0t,z0q,L,\
        th_0,q_0,G_S,G_L,H_S0,H_L0,H_Sspr,H_Rspr,H_Lspr,Lv,cp_a,rho_sw,nu,Phi_s,Mw,Ms,g,zRvaries):
    """
    Droplet thermodynamic calculations for spray heat flux due to size change.
    Performs calculations for the entire droplet size vector.
    """
    t_zR = np.full_like(Fp,np.nan)
    q_zR = np.full_like(Fp,np.nan)
    fsolveFailed = np.full_like(Fp,False,dtype=bool)    # True if fsolve failed to find a solution
    p_zR = p_0 - rho_a*g*delspr/2    # Pressure at zR assuming zR equals delspr/2 [Pa], assume this even if zR != delspr/2
    t_zR_const = (th_0 - 1/G_S*(H_S0*(np.log((z0t+delspr/2)/z0t) - stabIntH((z0t+delspr/2)/L)) \
            + 0.5*(1 - stabIntSprayH((z0t+delspr/2)/L))*(H_Sspr - H_Rspr)))*(p_zR/1e5)**0.286    # t_zR if zR equals delspr/2 [K]
    q_zR_const =   q_0 - 1/G_L*(H_L0*(np.log((z0q+delspr/2)/z0q) - stabIntH((z0q+delspr/2)/L)) \
            + 0.5*(1 - stabIntSprayH((z0q+delspr/2)/L))*H_Lspr)    # q_zR if zR equals delspr/2 [kg kg-1]
    for i in r0_rng:
        if r0[i] < 100*1e-6 and zRvaries == True:    # Calculate zR iteratively
            tq_zR_fsolve_params = (r0[i],v_g[i],Fp[i],p_zR,gammaWB,y0,rho_a,Dv_a,delspr,z0t,z0q,L,th_0,\
                    q_0,G_S,G_L,H_S0,H_L0,H_Sspr,H_Rspr,H_Lspr,Lv,cp_a,rho_sw)
            tq_zR_i,infodict,ier,mesg = fsolve(tq_zR_fsolve_residual,(t_zR_ig,q_zR_ig),\
                    args = tq_zR_fsolve_params,full_output = True)
            t_zR[i] = tq_zR_i[0]
            q_zR[i] = tq_zR_i[1]
            fsolveFailed[i] = False if ier == 1 else True
        else:
            t_zR[i] = t_zR_const
            q_zR[i] = q_zR_const
    qsat0_zR = qsat0(t_zR,p_zR)    # Saturation specific humidity at zR [kg kg-1]
    betaWB_zR = 1/(1 + Lv*gammaWB*(1 + y0)/cp_a*qsat0_zR)    # Wetbulb coefficient at zR [-]
    s_zR = satratio(t_zR,p_zR,q_zR,0.99999)    # Saturation ratio at zR [-]
    tauR = rho_sw*r0**2/(rho_a*Dv_a*Fp*qsat0_zR*betaWB_zR*np.abs(1 + y0 - s_zR))    # Char timescale for evap [s]
    req = r0*(xs*(1 + nu*Phi_s*Mw/Ms/(1 - s_zR)))**(1/3)    # Equilibrium radius at zR [m]
    delR = v_g*tauR    # Layer thickness governing H_Rspr [m]
    if zRvaries:
        zR = np.minimum(0.5*delspr,0.5*delR)    # H_Rspr height [m]
    else:
        zR = delspr/2
    rf = req + (r0 - req)*np.exp(-tauf/tauR)    # Final droplet radius [m]
    # Replace points where s_zR ~ 1 + y0, or fsolve fails
    szR_EQ_1Py0 = np.logical_and(abs(1 + y0 - s_zR) < 1e-3,~fsolveFailed)    # Points where s_zR ~ 1 + y0
    #szR_EQ_1Py0  = np.full_like(t_zR_ig,False,dtype=np.bool)    # Uncomment to keep these points; for debugging
    #fsolveFailed = np.full_like(t_zR_ig,False,dtype=np.bool)    # Uncomment to keep these points; for debugging
    s_zR = np.where(fsolveFailed,np.nan,\
           np.where(szR_EQ_1Py0, 1 + y0,s_zR))
    tauR = np.where(fsolveFailed,np.nan,\
           np.where(szR_EQ_1Py0, np.nan,tauR))
    zR   = np.where(fsolveFailed,np.nan,\
           np.where(szR_EQ_1Py0,\
               np.where(np.logical_and(abs(t_0 - t_zR) < 5e-1,abs(q_0 - q_zR) < 5e-4),0,np.nan),zR))
    rf   = np.where(fsolveFailed,r0,\
           np.where(szR_EQ_1Py0, r0,rf))
    return (s_zR,tauR,zR,rf)

def tq_zR_fsolve_residual(tq_zR,*params):
    """
    Residuals for profile equations for t_zR and q_zR, used by fsolve in thermo_HRspr.
    Performs calculations on one element of the droplet size vector.
    """
    t_zR,q_zR = tq_zR    # Temp and specific humidity at zR [K, kg kg-1]
    r0,v_g,Fp,p_zR,gammaWB,y0,rho_a,Dv_a,delspr,z0t,z0q,L,th_0,q_0,G_S,G_L,H_S0,H_L0,H_Sspr,\
            H_Lspr,H_Rspr,Lv,cp_a,rho_sw = params
    qsat0_zR = qsat0(t_zR,p_zR)    # Saturation specific humidity at zR [kg kg-1]
    betaWB_zR = 1/(1 + Lv*gammaWB*(1 + y0)/cp_a*qsat0_zR)    # Wetbulb coefficient at zR [-]
    s_zR = satratio(t_zR,p_zR,q_zR,0.99999)    # Saturation ratio at zR [-]
    tauR = rho_sw*r0**2/(rho_a*Dv_a*Fp*qsat0_zR*betaWB_zR*np.abs(1 + y0 - s_zR))    # Char timescale for evap [s]
    delR = v_g*tauR    # Layer thickness governing H_Rspr [m]
    zR = np.minimum(0.5*delspr,0.5*delR)    # H_Rspr height [m]
    Res_t_zR = (th_0 - 1/G_S*(H_S0*(np.log((z0t+zR)/z0t) - stabIntH((z0t+zR)/L)) \
            + zR/delspr*(1 - stabIntSprayH((z0t+zR)/L))*(H_Sspr - H_Rspr)))*(p_zR/1e5)**0.286 - t_zR    # Res for t [K], O(1)
    Res_q_zR =  (q_0 - 1/G_L*(H_L0*(np.log((z0q+zR)/z0q) - stabIntH((z0q+zR)/L)) \
            + zR/delspr*(1 - stabIntSprayH((z0q+zR)/L))*H_Lspr) - q_zR)*1000    # Res for q [g kg-1], O(1)
    return (Res_t_zR,Res_q_zR)

def swh_WEA17_Hack(U_10):
    """
    Hack to Wang et al. (2017) parameterization for significant wave
    height as a function of 10-m windspeed.
    Parameters:
        U_10 - 10-m windspeed [m s-1]
    Return:
        swh - significant wave height [m]
    """
    swh_WEA17 = 0.0143*U_10**2 + 0.9626    # SWH per Wang et al. 2017 [m]
    swh_tail = 5.5 + 0.14*U_10    # Hacked linear tail [m]
    smooth1 = 1 - 0.5*(np.tanh((U_10-24)/7) + 1)
    smooth2 =     0.5*(np.tanh((U_10-26)/7) + 1)
    swh = swh_WEA17*smooth1 + swh_tail*smooth2    # Combine WEA17 and hacked tail [m]
    return swh




