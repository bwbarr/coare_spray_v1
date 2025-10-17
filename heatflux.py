import numpy as np
from scipy.optimize import fsolve
from .ssgf import ssgf_dissejec_BCF23,ssgf_whitecap_F94
from .util import charnock,fall_velocity_PK97,qsat0,satratio,stabIntH,stabIntM,stabIntSprayH,\
        thermo_HRspr,thermo_HTspr,tq_zR_fsolve_residual,swh_WEA17_Hack

def sprayHFs(z_1,t_1,q_1,z_u,U_u,p_0,t_0,eps,dcp,swh,mss,fs,r0=None,delta_r0=None,SSGFname='dissejec_SS_BCF23',param_delspr_Wi=False,feedback=True,getprofiles=False,zRvaries=False,sprayLB=10.0,fdbksolve='iterIG',scaleSSGF=False,chi1=None,chi2=None,which_stress='C3.6_Wi',ustar_bulk_in=None,use_gf=True,which_z0tq='C3.6',z_ref=-1,print_counter=True,extravars=None):
    """
    Surface layer model for air-sea heat fluxes with spray.  Uses COARE framework for bulk 
    momentum and heat fluxes and uses BCF model to incorporate spray.  Set up to make 
    calculations using model output or in situ surface observations.  The model allows spray
    heat fluxes to feed back on bulk heat fluxes through both subgrid changes to the 
    spray layer temp/humidity profiles (the gamma feedback factors) and through changes to the 
    Obukhov stability length L.  The spray heat fluxes themselves also receive feedback from changes
    to the spray layer temp/humidity profiles and changes to L.    

    Parameters:
        z_1 - height of SL model top [m].  This is based on the thermodynamic variables rather
            than wind because spray heat fluxes and feedback are connected more strongly to SL
            thermodynamics than to wind.  z_1 is the height of the lowest atmospheric model 
            mass level if using model output, and it is the height of temp and humidity 
            measurements (assumed at the same height) if using observations.
        t_1 - temperature at z_1 [K]
        q_1 - specific humidity at z_1 [kg kg-1]
        z_u - height of given wind value [m].  This is the lowest atmospheric model mass level 
            (equal to z_1) if using model output, and it is the height of wind measurements 
            (probably not equal to z_1) if using observations.
        U_u - windspeed magnitude at z_u [m s-1].  Note that all windspeeds are current-relative.
        p_0 - surface pressure [Pa]
        t_0 - sea surface temperature [K]
        eps - wave energy dissipation flux [W m-2]
        dcp - dominant phase speed [m s-1]
        swh - significant wave height [m]
        mss - mean squared waveslope [-]
        fs - scale factor on droplet SSGF [-] ------- remove?
        r0 - SSGF radius vector [m]
        delta_r0 - SSGF bin width [m]
        SSGFname - name of SSGF to use.  Options are:
            Dissipation-ejection based: 
                'dissejec_SS_BCF23' - Uses original BCF23 spray model coefficients
                    (with calibration bug), i.e., fs = 2.2, C1 = 1.35, C2 = 0.1116, which 
                    reproduces the published BCF23 results.  This does not match the 
                    (corrected) F94+MOM80+fs0.4+30m/s datum.  Parameterized by seastate 
                    according to BCF23.
                'dissejec_Wi_BCF23' - Same as above, except parameterized by winds 
                    per BEA26.
                'dissejec_SS_BEA26_Cntl' - Uses BEA26 Control spray model coefficients, 
                    i.e., fs = 2.2, C1 = 0.969, C2 = 0.1116, which corrects the 
                    calibration bug, causing lower spray production than in the 
                    published BCF23.  Matches the F94+MOM80+fs0.4+30m/s datum.
                    Parameterized by seastate according to BCF23.
                'dissejec_Wi_BEA26_Cntl' - Same as above, except parameterized by winds
                    per BEA26.
                'dissejec_SS_BEA26_C3.6' - Uses BEA26 C3.6 Calibration spray model
                    coefficients, i.e., fs = 2.2, C1 = 0.969, C2 = 0.07, which are
                    optimized for use with COARE 3.6 scalar roughness lengths.  
                    Matches the F94+MOM80+fs0.4+30m/s datum.  Parameterized by 
                    seastate according to BCF23.
                'dissejec_Wi_BEA26_C3.6' - Same as above, except parameterized by winds
                    per BEA26.  (Also 'dissejec_Wi_BEA26_C3.6optim' for optimization work.)
                'dissejec_SS_BEA26_C4.X' - Uses BEA26 C4.X Calibration spray model
                    coefficients, i.e., fs = 2.2, C1 = 0.969, C2 = 0.15, which are
                    optimized for use with COARE 4.X scalar roughness lengths.
                    Matches the F94+MOM80+fs0.4+30m/s datum.  Parameterized by 
                    seastate according to BCF23.
                'dissejec_Wi_BEA26_C4.X' - Same as above, except parameterized by winds
                    per BEA26.  (Also 'dissejec_Wi_BEA26_C4.Xoptim' for optimization work.)
            Whitecap based:
                'whitecap_Wi_F94_MOM80' - parameterized by winds, using F94 size
                    distribution with MOM80 whitecap fraction.  The spray mass
                    flux from this model at U10 = 30 m/s with fs = 0.4 is currently
                    the datum for all other options.  The datum was calculated 
                    incorrectly in BCF23, leading to the 'published' vs 'fixed'
                    versions for some SSGFs.
                'whitecap_Wi_F94_BCF23_published' - parameterized by winds, using
                    F94 size distribution with BCF23 wind-based whitecap fraction.
                    Uses fs = 3.1, which does not match the F94+MOM80+fs0.4+30m/s
                    datum but reproduces the (incorrect) published BCF23 results.
                'whitecap_Wi_F94_BCF23_fixed' - parameterized by winds, using 
                    F94 size distribution with BCF23 wind-based whitecap fraction.
                    Uses fs = 2.2, which gives lower spray generation than the 
                    published BCF23 results, but matches the F94+MOM80+fs0.4+30m/s
                    datum.
        param_delspr_Wi - True to parameterize spray layer thickness from winds, False to use input swh.
        feedback - True to include subgrid-scale spray feedback -- Recommended
        getprofiles - True to calculate and output vertical thermodynamic profiles within surface layer
        zRvaries - True to iteratively determine zR -- Super slow and not recommended
        sprayLB - Lower bound on U_10 for calculating spray heat fluxes [m s-1]
        fdbksolve - how to solve for feedback.  Options are:
            'fsolve' - solves feedback using fsolve (slow, this is "the truth")
            'iterNoIG' - solves feedback with no initial guess physics with 5 iterations, this option
                tends to blow up in shallow water with large spray generation
            'iterIG' - solves feedback using initial guess physics with one iteration, this option
                is fast and usually gives good results, so it is the recommended option.
        scaleSSGF - True to scale SSGF to favor large or small droplets using chi1 and chi2 (rarely used)
        chi1 - factor scaling small droplet end of SSGF
        chi2 - factor scaling large droplet end of SSGF
        which_stress - how to calculate stress.  Options are:
            'C3.6_Wi' - COARE 3.6 algorithm, wind-dependent charnock parameter
            'C3.6_SS' - COARE 3.6 algorithm, seastate-dependent charnock parameter
            'ustar_bulk_given' - surface stress (represented by bulk ustar) is provided as an input.  This option
                allows stress to be provided by a wave model.
        ustar_bulk_in - if using which_stress option 'ustar_bulk_given', this is the input bulk ustar.  If 
            parameterizing stress internally (using COARE), set this to None.
        use_gf - True to use gust factor physics, False to set gf to 1.0.
        which_z0tq - how to calculate scalar roughness lengths:
            'C3.6' for COARE 3.6,
            'C4.0' for COARE 4.0,
            'C4.X' for COARE 4.X (also 'C4.Xoptim' for optimization work),
            'Hyp' for test hyperbola (in development, probably will remove),
            'ReyAn' for Reynolds analogy scaling (in development)
        z_ref - reference height for calculating spray changes to subgrid surface layer temperature,
            specific humidity, and saturation ratio [m].  z_ref should not be larger than z_1.  Pass
            z_ref = -1 to calculate changes at mid-spray-layer height.
        print_counter - True to print counter looping over spray points, False to not print.
        extravars - dictionary of variables for special options.
    """

    # 1. Define constants and non-varying parameters ================================================
    kappa = 0.4    # von Karman constant [-]
    g = 9.81    # Acceleration due to gravity [m s-2]
    Rdry = 287.1    # Dry air gas constant [J kg-1 K-1]
    rho_sw = 1030.    # Density of seawater [kg m-3]
    rho_dry = 2160.    # Density of chrystalline salt [kg m-3]
    cp_sw = 4200.    # Specific heat capacity of seawater [J kg-1 K-1]
    cp_a = 1004.67    # Specific heat capacity of air [J kg-1 K-1]
    nu = 2    # Number of ions into which NaCl dissociates [-]
    Phi_s = 0.924    # Practical osmotic coefficient at molality of 0.6 [-]
    Mw = 18.02    # Molecular weight of water [g mol-1]
    Ms = 58.44    # Molecular weight of salt [g mol-1]
    xs = 0.035    # Mass fraction of salt in seawater [-]
    Pr = 0.71    # Prandtl number for air [-]
    Sc = 0.60    # Schmidt number for water vapor diffusivity in air [-]
    Lv = (2.501-0.00237*(t_0-273.15))*1e6    # Latent heat of vap for water [J kg-1]
    rdryBYr0 = (xs*rho_sw/rho_dry)**(1/3)    # Ratio of rdry to r0 [-]
    y0 = -nu*Phi_s*Mw/Ms*rho_dry/rho_sw*rdryBYr0**3/(1 - rho_dry/rho_sw*rdryBYr0**3)    # y for surface seawater [-]
    q_0 = qsat0(t_0,p_0)*(1 + y0)    # Specific humidity at surface (accounting for salt) [kg kg-1]
    tv_1 = t_1*(1+0.608*q_1)    # Virtual temperature at z_1 [K]
    rho_a = (p_0 - 1.25*g*z_1)/(Rdry*tv_1)    # Air density [kg m-3], adjusting pressure using rho_a ~ 1.25 kg m-3
    p_1  = p_0 - rho_a*g*z_1    # Pressure at z_1 [Pa].  All pressure adjustments assume hydrostatic gradient.
    p_10 = p_0 - rho_a*g*10    # 10m pressure [Pa]
    th_0 = t_0*(1e5/p_0)**0.286    # Potential temperature at surface [K]
    th_1 = t_1*(1e5/p_1)**0.286    # Potential temperature at z_1 [K]
    thv_1 = th_1*(1+0.608*q_1)    # Virtual potential temperature at z_1 [K]
    tC_1 = t_1 - 273.15    # Convert to [C] for property calculations
    k_a = 2.411e-2*(1.+3.309e-3*tC_1-1.441e-6*tC_1**2)    # Thermal conductivity of air [W m-1 K-1]
    nu_a = 1.326e-5*(1.+6.542e-3*tC_1+8.301e-6*tC_1**2-4.84e-9*tC_1**3)    # Kin visc of air [m2 s-1]
    Dv_a = 2.11e-5*((tC_1+273.)/273.)**1.94    # Water vapor diffusivity in air [m2 s-1]
    gammaWB = 240.97*17.502/(tC_1+240.97)**2    # gamma = (dqsat/dT)/qsat [K-1], per Buck (1981)
    
    # 2. Define first guesses for MO-related quantities ====================================================
    # Set up first guess for gust factor or set dummy values to remove the physics.
    if use_gf:
        gust = np.full_like(U_u,0.5)    # Gust velocity [m s-1], first guess per COARE code
        S_u = np.sqrt(U_u**2 + gust**2)    # Windspeed at z_u adjusted for subgrid-scale gustiness [m s-1]
        gf = S_u/U_u    # Gust factor [-]
    else:
        S_u = np.copy(U_u)    # Gusty wind at z_u equals mean windspeed [m s-1]
        gf = np.full_like(U_u,1.0)    # Gust factor equal to 1.0 means no gustiness [-]
    # The following section determines initial values for interfacial flux-related items.
    # It comes directly from the COARE algorithm code.
    S_10 = S_u*np.log(10/1e-4)/np.log(z_u/1e-4)    # 10m windspeed including gustiness [m s-1], first guess using z0~1e-4, neutral
    ustar = 0.035*S_10    # Friction velocity including gustiness [m s-1], first guess
    z0 = 0.11*nu_a/ustar + 0.011*ustar**2/g    # Momentum roughness length [m], first guess using COARE 2.5 Charnock value of 0.011
    Cd10 = (kappa/np.log(10/z0))**2    # 10m drag coefficient [-], neglecting stability
    Ch10 = 0.00115    # 10m Stanton number [-]
    Ct10 = Ch10/np.sqrt(Cd10)    # 10m transfer coeff for temperature [-]
    z0t = 10/np.exp(kappa/Ct10)    # Thermal roughness length [m], first guess
    z0q = np.copy(z0t)    # Moisture roughness length [m], first guess
    Cd = (kappa/np.log(z_u/z0))**2    # Drag coefficient based on z_u [-], neglecting stability
    Ct = kappa/np.log(z_1/z0t)    # Transfer coeff for temperature based on z_1 [-]
    CC = kappa*Ct/Cd    # Transfer coefficient ratio [-]
    Ribcu = -z_u/600/0.004/1.2**3    # Critical bulk Ri?  Need reference.  Uses PBLH=600m and Beta=1.2 from C3.6 Matlab code
    Ribu = -g*z_u/t_1*(th_0-th_1 + 0.608*t_1*(q_0-q_1))/S_u**2    # Bulk Ri [-]
    zetu = np.where(Ribu<0,CC*Ribu/(1+Ribu/Ribcu),CC*Ribu*(1+27/9*Ribu/CC))    # Stability parameter derived from Ribu [-]
    L = z_u/zetu    # Obukhov stability length [m], first guess
    # Calculate first guess for interfacial fluxes
    if which_stress in ['C3.6_Wi','C3.6_SS']:    # If parameterizing stress, calculate ustar from z0
        ustar = S_u*kappa/(np.log(z_u/z0) - stabIntM(z_u/L))    # [m s-1]
        alpha_char = charnock(which_stress,S_10,swh,ustar,dcp)    # Charnock variable [-]
    elif which_stress in ['ustar_bulk_given']:    # If using input stress, calculate gusty ustar from ustar_bulk
        ustar = ustar_bulk_in*np.sqrt(gf)    # [m s-1]
    thstar_pr = -(th_0 - th_1)*kappa/(np.log(z_1/z0t) - stabIntH(z_1/L))    # Turbulent scale for theta without spray [K], first guess
    qstar_pr  = -( q_0 -  q_1)*kappa/(np.log(z_1/z0q) - stabIntH(z_1/L))    # Turbulent scale for q without spray [kg kg-1], first guess
    thvstar = thstar_pr*(1+0.61*q_1) + 0.61*t_1*qstar_pr    # Turb scale for thetav (includes spray in general but not here) [K], first guess

    # 3. Enter loop that iterates on L and heat fluxes ====================================================
    # Spray contributes to thvstar, so both interfacial and spray heat fluxes are calculated within loop.
    nits = 2#10    # Number of iterations for stability.  COARE 3.6 default is 10.
    for n in range(nits):

        print('Bulk iteration number: ' + str(n))

        # 4. Calculate L, roughness lengths, and turbulent scales without spray ====================================
        L = ustar**2/(kappa*g/thv_1*thvstar)
        if which_stress in ['C3.6_Wi','C3.6_SS']:    # Parameterize z0 using COARE
            z0 = 0.11*nu_a/ustar + alpha_char*ustar**2/g    # [m]
        elif which_stress in ['ustar_bulk_given']:    # Invert z0 using given stress
            z0 = z_u/np.exp(S_u*kappa/ustar + stabIntM(z_u/L))    # [m]
        Restar = ustar*z0/nu_a    # Roughness Reynolds number [-]
        if which_z0tq == 'C3.6':    # Close approximation to COARE 3.0
            z0q = np.minimum(1.6e-4,5.8e-5/Restar**0.72)    # [m]
            z0t = np.copy(z0q)    # [m]
        elif which_z0tq == 'C4.0':    # Jim's latest COARE 4.0
            z0t = np.minimum(5.5e-5,np.minimum(7.0e-5/Restar**0.7,3.0e-2/Restar**2.4))
            z0q = np.minimum(1.0e-4,np.minimum(7.5e-5/Restar**0.5,6.0e-4/Restar**1.2))
        elif which_z0tq in ['C4.X','C4.Xoptim']:    # COARE 4.X
            if which_z0tq == 'C4.X':
                A_C4X = 1.8e-3
                B_C4X = -1.5
            elif which_z0tq == 'C4.Xoptim':
                A_C4X = extravars['A_C4.X']
                B_C4X = extravars['B_C4.X']
            z0q = np.minimum(1.0e-4,np.minimum(7.5e-5/Restar**0.5,A_C4X*Restar**B_C4X))    # Same as C4.0 but last term is changed
            z0tz0q_G92_smooth = np.exp(13.6*kappa*(Sc**(2/3) - Pr**(2/3)))    # Ratio z0t/z0q [-], Garratt 1992, smooth
            z0t = z0q*z0tz0q_G92_smooth    # [m]
        elif which_z0tq == 'Hyp':    # Form of hyperbola, in development
            a_HYP = 2
            b_HYP = 1
            c_HYP = 8
            x0_HYP = -1
            y0_HYP = -2
            ex_HYP = 4
            z0q = 10**(y0_HYP - (a_HYP*((np.log10(Restar) - x0_HYP)**ex_HYP/b_HYP + c_HYP))**(1/ex_HYP))    # [m]
            z0q[Restar < 1e-1] = 1e-4
            z0t = np.copy(z0q)    # [m]
        elif which_z0tq == 'ReyAn':    # New Reynolds analogy scaling
            lnz0z0t_G92_smooth = kappa*(13.6*Pr**(2/3) - 12)    # Roughness ratio for heat, Garratt 1992, smooth
            lnz0z0q_G92_smooth = kappa*(13.6*Sc**(2/3) - 12)    # Roughness ratio for moisture, Garratt 1992, smooth
            z0tz0_smooth = 1/np.exp(lnz0z0t_G92_smooth)    # z0t/z0 per G92 in smooth wall limit [-]
            z0qz0_smooth = 1/np.exp(lnz0z0q_G92_smooth)    # z0q/z0 per G92 in smooth wall limit [-]
            z0_smooth = 0.11*nu_a/ustar    # Momentum roughness length for turbulent flow over a smooth wall [m]
            g1 = z0_smooth*(1 + (ustar/9e-2)**5)**0.5    # Smooth flow limit with transition [m]
            g2 =    4.0e-5/(1 + (ustar/4e-1)**5.2)    # Rough flow plateau and dropoff [m]
            z0_int_RA = np.where(ustar < 1e-1,g1,np.minimum(g1,g2))    # Reynolds analogy parameterization for interfacial z0 [m]
            z0t = z0_int_RA*z0tz0_smooth    # [m]
            z0q = z0_int_RA*z0qz0_smooth    # [m]
        if which_stress in ['C3.6_Wi','C3.6_SS']:    # Update ustar using new z0, L, and S_u
            ustar = S_u*kappa/(np.log(z_u/z0) - stabIntM(z_u/L))
        elif which_stress in ['ustar_bulk_given']:    # Update ustar using new gf with input value of ustar_bulk_in
            ustar = ustar_bulk_in*np.sqrt(gf)
        thstar_pr = -(th_0 - th_1)*kappa/(np.log(z_1/z0t) - stabIntH(z_1/L))
        qstar_pr  = -( q_0 -  q_1)*kappa/(np.log(z_1/z0q) - stabIntH(z_1/L))

        # 5. Calculate stress and heat fluxes without spray, and some other stability/feedback related fields =========
        # Fluxes
        G_S = rho_a*cp_a*kappa*ustar    # Dimensional group for sensible heat [W m-2 K-1]
        G_L = rho_a*Lv*kappa*ustar    # Dimensional group for latent heat [W m-2]
        H_S0pr = -G_S/kappa*thstar_pr    # Bulk SHF without spray [W m-2]
        H_L0pr = -G_L/kappa*qstar_pr    # Bulk LHF without spray [W m-2]
        tau = rho_a*ustar**2/gf    # Bulk stress [Pa]
        # Define spray layer thickness using input swh or parameterized swh.  Then, calculate some associated fields.
        U_10 = ustar/kappa/gf*(np.log(10/z0) - stabIntM(10/L))    # 10m windspeed [m s-1], gustiness removed
        swh_param = swh_WEA17_Hack(U_10)    # Wind-based model for swh [m]
        if param_delspr_Wi:
            delspr = np.minimum(swh_param,z_1)    # Spray layer thickness [m] is nominally one swh per M&V2014a.  Limited to z_1.
        elif param_delspr_Wi == False:
            delspr = np.minimum(swh,z_1)    # Spray layer thickness [m] is nominally one swh per M&V2014a.  Limited to z_1.
            delspr[np.isnan(swh)] = swh_param[np.isnan(swh)]    # Where input swh has nans, revert to parameterization (used for incomplete obs data).
        zref = delspr/2 if z_ref < 0 else np.full_like(U_u,z_ref)    # Set reference height to user-specified height or mid-spray-layer height
        p_delsprD2 = p_0 - rho_a*g*delspr/2    # Pressure at delspr/2 [Pa]
        p_zref     = p_0 - rho_a*g*zref    # Pressure at zref [Pa]
        # Stability functions and reference conditions
        psiH_1        = stabIntH(z_1/L)    # Stability integral for heat at z_1 [-]
        psiH_delspr   = stabIntH(delspr/L)    # Stability integral for heat at delspr [-]
        psiH_delsprD2 = stabIntH(delspr/2/L)    # Stability integral for heat at delspr/2 [-]
        psiH_zref     = stabIntH(zref/L)    # Stability integral for heat at zref [-]
        phisprH_delspr = stabIntSprayH(delspr/L)    # Stability integral for heat with spray at delspr [-]
        phisprH_zref   = stabIntSprayH(zref/L)    # Stability integral for heat with spray at zref [-]
        t_delsprD2pr = (th_0 - H_S0pr/G_S*(np.log(delspr/2/z0t) - psiH_delsprD2))*(p_delsprD2/1e5)**0.286    # t at mid-layer w/o fdbk [K]
        t_zref_pr    = (th_0 - H_S0pr/G_S*(np.log(zref/z0t)     - psiH_zref    ))*(p_zref/1e5)**0.286    # t at zref w/o fdbk [K]
        q_delsprD2pr =   q_0 - H_L0pr/G_L*(np.log(delspr/2/z0q) - psiH_delsprD2)    # q at mid-layer w/o fdbk [kg kg-1]
        q_zref_pr    =   q_0 - H_L0pr/G_L*(np.log(zref/z0q)     - psiH_zref)    # q at zref w/o fdbk [kg kg-1]
        s_delsprD2pr = satratio(t_delsprD2pr,p_delsprD2,q_delsprD2pr,0.99999)    # s at mid-layer w/o fdbk [-]
        s_zref_pr    = satratio(t_zref_pr   ,p_zref    ,q_zref_pr   ,0.99999)    # s at zref w/o fdbk [-]
        # Interfacial feedback coefficients for SHF and LHF
        if feedback:
            gamma_S = (np.log(delspr/z0t) - psiH_delspr - 1 + phisprH_delspr)/(np.log(z_1/z0t) - psiH_1)    # [-]
            gamma_L = (np.log(delspr/z0q) - psiH_delspr - 1 + phisprH_delspr)/(np.log(z_1/z0q) - psiH_1)    # [-]
        else:
            gamma_S = np.full_like(z_1,1.0)    # [-]
            gamma_L = np.full_like(z_1,1.0)    # [-]

        # 6. Calculate fields for spray heat fluxes that are not affected by subgrid feedback ====================
        # SSGF and droplet hydrodynamics
        if SSGFname in ['dissejec_SS_BCF23'     ,'dissejec_Wi_BCF23'     ,\
                        'dissejec_SS_BEA26_Cntl','dissejec_Wi_BEA26_Cntl',\
                        'dissejec_SS_BEA26_C3.6','dissejec_Wi_BEA26_C3.6','dissejec_Wi_BEA26_C3.6optim',\
                        'dissejec_SS_BEA26_C4.X','dissejec_Wi_BEA26_C4.X','dissejec_Wi_BEA26_C4.Xoptim']:    # Dissipation-ejection based models
            r0,delta_r0,M_spr,dmdr0 = ssgf_dissejec_BCF23(SSGFname,eps,swh,dcp,mss,ustar,z0,L,gf,\
                    r0=r0,delta_r0=delta_r0,extravars=extravars)    # [m],[m],[kg m-2 s-1],[kg m-2 s-1 m-1]
            if scaleSSGF:    # Scale SSGF to favor large or small droplets
                r0,delta_r0,M_spr_scale,dmdr0_scale = ssgf_dissejec_BCF23(SSGFname,eps,swh,dcp,mss,ustar,z0,L,gf,\
                        r0=r0,delta_r0=delta_r0,chi1=chi1,chi2=chi2,extravars=extravars)
                dmdr0 = dmdr0_scale*M_spr/M_spr_scale    # Normalize dmdr0_scale to have same integrated mass flux as dmdr0
        elif SSGFname in ['whitecap_Wi_F94_MOM80','whitecap_Wi_F94_BCF23_published',\
                          'whitecap_Wi_F94_BCF23_fixed']:    # Whitecap based models
            r0,delta_r0,M_spr,dmdr0 = ssgf_whitecap_F94(SSGFname,U_10,r0=r0,delta_r0=delta_r0)    # [m],[m],[kg m-2 s-1],[kg m-2 s-1 m-1]
        r0_rng = np.arange(np.size(r0))    # List to use for stepping through r0
        v_g = fall_velocity_PK97(r0)    # Droplet settling velocity [m s-1]
        tauf = np.array([delspr/v_g[i] for i in r0_rng])    # Characteristic droplet settling time [s]
        Fp = np.array([1.+0.25*(2.*v_g[i]*r0[i]/nu_a)**0.5 for i in r0_rng])    # Slip factor (Pr&Kl) [-]
        nospray = np.isnan(eps)    # No spray (e.g. over land) where there is no wave data
        zerospray = np.logical_and(~nospray,U_10 < sprayLB)    # Assume spray heat fluxes are zero below lower bound
        # Heat flux due to temperature change
        tauT = np.array([np.where(np.logical_or(nospray,zerospray),np.nan,\
                rho_sw*cp_sw*r0[i]**2/3./k_a/Fp[i,:,:]) for i in r0_rng])    # Characteristic cooling time [s]
        zT = np.array([np.minimum(0.5*delspr,0.5*v_g[i]*tauT[i,:,:]) for i in r0_rng])    # H_Tspr height [m]
        t_zT       = np.full_like(dmdr0,np.nan)    # Temperature at zT [K]
        wetdep_zT  = np.full_like(dmdr0,np.nan)    # Wetbulb depression at zT [K]
        # Heat flux due to size change
        zR_ig = np.where(np.logical_or(nospray,zerospray),np.nan,delspr/2)    # Initial guess for H_Rspr height [m]
        p_zR_ig = p_0 - rho_a*g*zR_ig    # Pressure at zR_ig [Pa]
        t_zR_ig = (th_0 - H_S0pr/G_S*np.log((z0t+zR_ig)/z0t))*(p_zR_ig/1e5)**0.286    # Initial guess for temp at zR [K]
        q_zR_ig =   q_0 - H_L0pr/G_L*np.log((z0q+zR_ig)/z0q)    # Initial guess for q at zR [kg kg-1]
        s_zR       = np.full_like(dmdr0,np.nan)    # Saturation ratio at zR [-]
        tauR       = np.full_like(dmdr0,np.nan)    # Characteristic evaporation time [s]
        zR         = np.full_like(dmdr0,np.nan)    # H_Rspr height [m]
        rf         = np.full_like(dmdr0,np.nan)    # Final droplet radius [m]
    
        # 7. Enter loop over X-Y gridpoints to calculate spray and total heat fluxes ====================
        dHTsprdr0 = np.full_like(dmdr0,np.nan)    # H_Tspr per increment in r0 [W m-2 m-1]
        dHRsprdr0 = np.full_like(dmdr0,np.nan)    # H_Rspr per increment in r0 [W m-2 m-1]
        H_Tspr    = np.where(nospray,np.nan,0)    # Spray heat flux due to temp change [W m-2]
        H_Sspr    = np.where(nospray,np.nan,0)    # Spray sensible heat flux [W m-2]
        H_Rspr    = np.where(nospray,np.nan,0)    # Spray heat flux due to size change [W m-2]
        H_Lspr    = np.where(nospray,np.nan,0)    # Spray latent heat flux [W m-2]
        H_Ssprpr  = np.where(nospray,np.nan,0)    # Spray sensible heat flux, no feedback [W m-2]
        H_Rsprpr  = np.where(nospray,np.nan,0)    # Spray heat flux due to size change, no feedback [W m-2]
        H_Lsprpr  = np.where(nospray,np.nan,0)    # Spray latent heat flux, no feedback [W m-2]
        H_Rspr_IG = np.full_like(eps,np.nan)    # If fdbkfsolve == 'iterIG', the IG for H_Rspr [W m-2]
        H_S0      = np.copy(H_S0pr)    # Interfacial sensible heat flux [W m-2]
        H_L0      = np.copy(H_L0pr)    # Interfacial latent heat flux [W m-2]
        H_S1      = np.where(zerospray,H_S0,np.nan)    # Total sensible heat flux [W m-2]
        H_L1      = np.where(zerospray,H_L0,np.nan)    # Total latent heat flux [W m-2]
        SP = ~np.logical_or(nospray,zerospray)    # Points where we will make calculations for spray
        SPindx = np.where(SP)    # Indices of SP
    
        print('Total spray points: %d' % np.sum(SP))
        for j in range(np.sum(SP)):

            if print_counter:
                print(j)

            # 8. Calculate spray and total heat fluxes for a single gridpoint, including feedback ========
            # List of parameters for heat flux calcs.  These are not changed by spray feedback.
            allHFs_params = (r0_rng,r0,delta_r0,v_g,\
                    np.array([zT[i,:,:][SP][j] for i in r0_rng]),\
                    np.array([tauf[i,:,:][SP][j] for i in r0_rng]),\
                    np.array([tauT[i,:,:][SP][j] for i in r0_rng]),\
                    t_zR_ig[SP][j],q_zR_ig[SP][j],\
                    np.array([Fp[i,:,:][SP][j] for i in r0_rng]),\
                    np.array([dmdr0[i,:,:][SP][j] for i in r0_rng]),\
                    p_0[SP][j],gammaWB[SP][j],y0,t_0[SP][j],rho_a[SP][j],\
                    Dv_a[SP][j],xs,delspr[SP][j],z0t[SP][j],z0q[SP][j],L[SP][j],\
                    th_0[SP][j],q_0[SP][j],G_S[SP][j],G_L[SP][j],H_S0pr[SP][j],H_L0pr[SP][j],\
                    gamma_S[SP][j],gamma_L[SP][j],M_spr[SP][j],\
                    Lv[SP][j],cp_a,rho_sw,nu,Phi_s,Mw,Ms,cp_sw,g,zRvaries)
            # Heat fluxes without feedback
            allHFs_rtrn = update_allHFs(0.0,0.0,0.0,allHFs_params)
            H_Ssprpr[SPindx[0][j],SPindx[1][j]] = allHFs_rtrn[1]
            H_Rsprpr[SPindx[0][j],SPindx[1][j]] = allHFs_rtrn[2]
            H_Lsprpr[SPindx[0][j],SPindx[1][j]] = allHFs_rtrn[3]
            # Heat fluxes with feedback
            if feedback:
                # Calculate initial guess using simple model
                etaT = 17.502*240.97/(t_delsprD2pr[SP][j]-273.15+240.97)**2    # [K-1]
                Cs_pr = (1+y0-s_delsprD2pr[SP][j])**2/(1-s_delsprD2pr[SP][j])    # [-]
                if delspr[SP][j] < 4:
                    C_HIG = 1.0    # Tuneable constant for equivalent height
                elif delspr[SP][j] > 10:
                    C_HIG = 0.7
                else:
                    C_HIG = -0.05*delspr[SP][j] + 1.2
                H_IG = min(C_HIG*delspr[SP][j],z_1[SP][j])    # Equivalent height for heating in simple model [m]
                Psi = etaT*np.log(z_1[SP][j]/H_IG)/(rho_a[SP][j]*cp_a*kappa*ustar[SP][j])*allHFs_rtrn[0]    # [-]
                Chi = etaT*np.log(z_1[SP][j]/H_IG)/(rho_a[SP][j]*cp_a*kappa*ustar[SP][j]) \
                         + np.log(z_1[SP][j]/H_IG)/(rho_a[SP][j]*Lv[SP][j]*kappa*ustar[SP][j]*q_delsprD2pr[SP][j])    # [m2 W-1]
                Lambda = (Psi - 1 + (1+y0)/s_delsprD2pr[SP][j])/Chi    # [W m-2]
                A = allHFs_rtrn[2]/Cs_pr + 1/s_delsprD2pr[SP][j]/Chi    # [W m-2]
                B = -Lambda - y0/s_delsprD2pr[SP][j]/Chi    # [W m-2]
                C = y0*Lambda    # [W m-2]
                s_hatPOS = (-B + np.sqrt(B**2 - 4*A*C))/2/A    # Larger root [-], seems physical
                #s_hatNEG = (-B - np.sqrt(B**2 - 4*A*C))/2/A    # Smaller root [-], seems unphysical
                H_Rspr_IGj = s_hatPOS**2/(s_hatPOS-y0)*allHFs_rtrn[2]/Cs_pr    # IG for H_Rspr [W m-2]
                if np.isnan(H_Rspr_IGj):
                    H_Rspr_IGj = 0
                # Solve for feedback using chosen method
                if fdbksolve == 'fsolve':    # Solve feedback using fsolve
                    # Solve for spray heat fluxes with feedback
                    IG = (allHFs_rtrn[1],H_Rspr_IGj,H_Rspr_IGj + (allHFs_rtrn[0] - allHFs_rtrn[1]))    # Use simple model for IG
                    H_spr_j,infodict,ier,mesg = fsolve(feedback_fsolve_residual,\
                            IG,args = allHFs_params,full_output = True)
                    # Get other associated parameters.  If feedback did not converge, force NaNs in results.
                    if ier != 1:
                        print('*** Feedback did not converge: j = %d ***' % j)
                        print('Fsolve failure flag: ier = %d' % ier)
                        print('Failure message: ' + mesg)
                        print('10m Windspeed: %f m/s' % U_10[SP][j])
                        r0nans = np.full_like(r0,np.nan)    # Dummy SSGF-sized array of nans
                        allHFs_rtrn = (np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,\
                                       r0nans,r0nans,r0nans,r0nans,r0nans,r0nans,r0nans,r0nans,\
                                       np.nan,np.nan)    # Dummy rtrn of nans
                    else:
                        allHFs_rtrn = update_allHFs(H_spr_j[0],H_spr_j[1],H_spr_j[2],allHFs_params)
                elif fdbksolve in ['iterNoIG','iterIG']:    # Perform a set number of iterations
                    if fdbksolve == 'iterNoIG':    # Use fluxes without feedback as IG
                        N_iter_fdbk = 5    # Number of iterations to perform
                        H_Sspr_n = allHFs_rtrn[1]
                        H_Rspr_n = allHFs_rtrn[2]
                        H_Lspr_n = allHFs_rtrn[3]
                    elif fdbksolve == 'iterIG':    # Use simple model to determine feedback IG
                        N_iter_fdbk = 1
                        H_Sspr_n = allHFs_rtrn[1]
                        H_Rspr_n = H_Rspr_IGj
                        H_Lspr_n = H_Rspr_IGj + (allHFs_rtrn[0] - allHFs_rtrn[1])
                    # Perform selected number of iterations
                    for n in range(N_iter_fdbk):
                        allHFs_rtrn = update_allHFs(H_Sspr_n,H_Rspr_n,H_Lspr_n,allHFs_params)
                        H_Sspr_n = allHFs_rtrn[1]
                        H_Rspr_n = allHFs_rtrn[2]
                        H_Lspr_n = allHFs_rtrn[3]
            # Unpack results
            H_Tspr[SPindx[0][j],SPindx[1][j]] = allHFs_rtrn[0]
            H_Sspr[SPindx[0][j],SPindx[1][j]] = allHFs_rtrn[1]
            H_Rspr[SPindx[0][j],SPindx[1][j]] = allHFs_rtrn[2]
            H_Lspr[SPindx[0][j],SPindx[1][j]] = allHFs_rtrn[3]
            H_S0[SPindx[0][j],SPindx[1][j]]   = allHFs_rtrn[4]
            H_L0[SPindx[0][j],SPindx[1][j]]   = allHFs_rtrn[5]
            H_S1[SPindx[0][j],SPindx[1][j]]   = allHFs_rtrn[6]
            H_L1[SPindx[0][j],SPindx[1][j]]   = allHFs_rtrn[7]
            if feedback and fdbksolve == 'iterIG':
                H_Rspr_IG[SPindx[0][j],SPindx[1][j]] = H_Rspr_IGj
            for i in r0_rng:
                t_zT[i,SPindx[0][j],SPindx[1][j]]      = allHFs_rtrn[8][i]
                wetdep_zT[i,SPindx[0][j],SPindx[1][j]] = allHFs_rtrn[9][i]
                s_zR[i,SPindx[0][j],SPindx[1][j]]      = allHFs_rtrn[10][i]
                tauR[i,SPindx[0][j],SPindx[1][j]]      = allHFs_rtrn[11][i]
                zR[i,SPindx[0][j],SPindx[1][j]]        = allHFs_rtrn[12][i]
                rf[i,SPindx[0][j],SPindx[1][j]]        = allHFs_rtrn[13][i]
                dHTsprdr0[i,SPindx[0][j],SPindx[1][j]] = allHFs_rtrn[14][i]
                dHRsprdr0[i,SPindx[0][j],SPindx[1][j]] = allHFs_rtrn[15][i]

        # 9. Calculate turbulent scales using total fluxes. Update wind-related items. =================
        thstar_tot = -H_S1*kappa/G_S    # Turbulent scale for theta with spray (total heat flux) [K]
        qstar_tot  = -H_L1*kappa/G_L    # Turbulent scale for q with spray (total heat flux)    [kg kg-1]
        thvstar    = thstar_tot*(1+0.61*q_1) + 0.61*t_1*qstar_tot    # thvstar includes both interfacial and spray contributions
        thvstar_pr =  thstar_pr*(1+0.61*q_1) + 0.61*t_1*qstar_pr    # thvstar without spray fluxes (but with spray effect on L)
        thvsstar   = thstar_tot*(1+0.51*q_1) + 0.51*t_1*qstar_tot    # This scale corresponds to the measured sonic buoyancy flux [K]
        # Don't update wind-related items if we are at the last iteration so that wind profiles stay internally consistent.
        if n < nits-1:
            if use_gf:
                Bf = -g/thv_1*ustar*thvstar    # Buoyancy flux [m2 s-3], used only to calculate gustiness
                gust = np.full_like(U_u,0.2)
                gust[Bf > 0] = np.maximum(0.2,1.2*(Bf[Bf > 0]*600)**(1/3))    # Using PBLH=600m and Beta=1.2 from C3.6 code
                S_u = np.sqrt(U_u**2 + gust**2)
                gf = S_u/U_u    # Gust factor [-]
                alpha_char = charnock(which_stress,ustar/kappa/gf*np.log(10/z0),swh,ustar,dcp)
            else:
                S_u = np.copy(U_u)    # Gusty wind at z_u equals mean windspeed [m s-1]
                gf = np.full_like(U_u,1.0)    # Gust factor equal to 1.0 means no gustiness [-]

    # 10. Calculate final diagnosed quantities after completion of all loops =========================
    H_BS1 = -G_S/kappa*thvsstar    # Bulk sonic temp flux [W m-2], for comparing to DC measurements
    # 10-m neutral conditions -- Assumes vertically constant heat fluxes that are equal to totals
    thvs_0 = th_0*(1+0.51*q_0)    # Virtual sonic potential temp at the surface [K]
    thvs_1 = th_1*(1+0.51*q_1)    # Virtual sonic potential temp at z_1 [K]
    S_10N = ustar/kappa*np.log(10/z0)    # 10m neutral windspeed [m s-1], with gustiness
    U_10N = S_10N/gf    # 10m neutral windspeed [m s-1], gustiness removed
    th_10N   =   th_1 +  H_S1/G_S*(np.log(z_1/10) - psiH_1)    # 10m neutral pot temp [K]
    q_10N    =    q_1 +  H_L1/G_L*(np.log(z_1/10) - psiH_1)    # 10m neutral spec hum [kg kg-1]
    thvs_10N = thvs_1 + H_BS1/G_S*(np.log(z_1/10) - psiH_1)    # 10m neutral virtual sonic potential temp [K]
    t_10N = th_10N*(p_10/1e5)**0.286    # 10m neutral temp [K]
    s_10N = satratio(t_10N,p_10,q_10N,0.99999)    # 10m neutral saturation ratio [-]
    # Specific available energies and heat transfer efficiencies
    betaWB_10N = 1/(1 + Lv*gammaWB*(1 + y0)/cp_a*qsat0(t_10N,p_10))    # 10m neutral WB coeff [-]
    wetdep_10N = (1 - s_10N/(1 + y0))*(1 - betaWB_10N)/gammaWB    # 10m neutral WB depression [K]
    reqBYr0_10N = (xs*(1 + nu*Phi_s*Mw/Ms/(1 - s_10N)))**(1/3)    # 10m neutral req/r0 [-]
    a_T = cp_sw*(t_0 - t_10N + wetdep_10N)    # Specific available energy for heat transfer due to temp change [J kg-1]
    a_R = Lv*(1 - reqBYr0_10N**3)    # Specific available energy for heat transfer due to size change [J kg-1]
    ET = np.array([dHTsprdr0[i,:,:]/a_T/dmdr0[i,:,:] for i in r0_rng])    # Efficiency for HT due to temp change [-]
    ER = np.array([dHRsprdr0[i,:,:]/a_R/dmdr0[i,:,:] for i in r0_rng])    # Efficiency for HT due to size change [-]
    ET_bar = H_Tspr/a_T/M_spr    # Mean efficiency for HT due to temp change [-]
    ER_bar = H_Rspr/a_R/M_spr    # Mean efficiency for HT due to size change [-]
    # Thermodynamic changes due to subgrid spray feedback
    if feedback == False:
        t_zref = np.copy(t_zref_pr)    # Temp at zref [K]
        q_zref = np.copy(q_zref_pr)    # q at zref [kg kg-1]
        s_zref = np.copy(s_zref_pr)    # s at zref [-]
    elif feedback == True:
        # Values if zref is within spray layer
        t_zref_spr = (th_0 - 1/G_S*(H_S0*(np.log(zref/z0t) - psiH_zref) \
                + zref/delspr*(1 - phisprH_zref)*(H_Sspr - H_Rspr)))*(p_zref/1e5)**0.286    # [K]
        q_zref_spr =   q_0 - 1/G_L*(H_L0*(np.log(zref/z0q) - psiH_zref) \
                + zref/delspr*(1 - phisprH_zref)*H_Lspr)    # [kg kg-1]
        # Values if zref is above spray layer
        t_zref_abv = (th_1 + H_S1/G_S*(np.log(z_1/zref) - psiH_1 + psiH_zref))*(p_zref/1e5)**0.286    # [K]
        q_zref_abv =   q_1 + H_L1/G_L*(np.log(z_1/zref) - psiH_1 + psiH_zref)    # [kg kg-1]
        # Select appropriate values
        t_zref = np.where(zref < delspr,t_zref_spr,t_zref_abv)
        q_zref = np.where(zref < delspr,q_zref_spr,q_zref_abv)
        s_zref = satratio(t_zref,p_zref,q_zref,0.99999)    # [-]
    delt_zref = t_zref - t_zref_pr    # Temp change at zref due to feedback [K], (+) = warming
    delq_zref = q_zref - q_zref_pr    # q change at zref due to feedback [kg kg-1], (+) = moistening
    dels_zref = s_zref - s_zref_pr    # s change at zref due to feedback [-], (+) = incr s
    delt_zref[zerospray] = 0
    delq_zref[zerospray] = 0
    dels_zref[zerospray] = 0
    # Spray heat flux subgrid feedback coefficients
    alpha_S = H_Sspr/H_Ssprpr    # [-]
    beta_S = H_Rspr/H_Rsprpr    # [-]
    beta_L = H_Lspr/H_Lsprpr    # [-]
    beta_S[np.abs(beta_S) == np.inf] = np.nan
    beta_L[np.abs(beta_L) == np.inf] = np.nan
    # 10-m neutral transfer coefficients (based on fluxes and 10N conditions)
    Cd10N  = tau/(rho_a*S_10N*U_10N)    # 10m neutral drag coefficient [-]
    Ch10N  = H_S1/(rho_a*cp_a*S_10N*(th_0 - th_10N))    # 10m neutral Stanton number [-]
    Cq10N  = H_L1/(rho_a*Lv*S_10N*(q_0 - q_10N))    # 10m neutral Dalton number [-]
    Ck10N  = (H_S1+H_L1)/(rho_a*S_10N*(cp_a*(th_0 - th_10N) + Lv*(q_0 - q_10N)))    # 10m neutral enth coeff [-]
    Cbs10N = H_BS1/(rho_a*cp_a*S_10N*(thvs_0 - thvs_10N))    # 10m neutral coeff for sonic temp flux [-]
    # "Equivalent" scalar roughness lengths accounting for spray
    z0t_EQ = z_1/np.exp(-kappa/thstar_tot*(th_0 - th_1) + psiH_1)    # Equivalent roughness for theta [m]
    z0q_EQ = z_1/np.exp(-kappa/qstar_tot*(q_0 - q_1) + psiH_1)    # Equivalent roughness for q [kg kg-1]

    # 11. Calculate vertical profiles of t, theta, q, and s =======================================================
    if getprofiles == False:
        profiles = None
    elif getprofiles == True:
        # Initialize lists
        z_t = []    # z-values counting from z0t [m]
        z_q = []    # z-values counting from z0q [m]
        th_prof = []    # Vertical potential temperature profiles [K]
        q_prof = []    # Vertical specific humidity profiles [kg kg-1]
        # 1. Roughness heights
        z_t.append(z0t)
        z_q.append(z0q)
        th_prof.append(th_0)
        q_prof.append(q_0)
        # 2. Spray layer
        nzspr = 50    # Number of points in spray layer
        zspr_inc = np.linspace(0,1,nzspr)[1:-1]    # Values at z0t/z0q and delspr added separately
        for k in zspr_inc:
            zK = 10**(-7+k*(np.log10(delspr)+7))    # These zK are relative to z0t or z0q
            z_t.append(z0t+zK)
            z_q.append(z0q+zK)
            if feedback == False:
                th_prof.append(th_0 - 1/G_S*(H_S0*(np.log((z0t+zK)/z0t) - stabIntH((z0t+zK)/L))))
                q_prof.append(  q_0 - 1/G_L*(H_L0*(np.log((z0q+zK)/z0q) - stabIntH((z0q+zK)/L))))
            elif feedback == True:
                th_prof.append(th_0 - 1/G_S*(H_S0*(np.log((z0t+zK)/z0t) - stabIntH((z0t+zK)/L)) + zK/delspr*(1 - stabIntSprayH((z0t+zK)/L))*(H_Sspr - H_Rspr)))
                q_prof.append(  q_0 - 1/G_L*(H_L0*(np.log((z0q+zK)/z0q) - stabIntH((z0q+zK)/L)) + zK/delspr*(1 - stabIntSprayH((z0q+zK)/L))*H_Lspr))
        # 3. Above spray layer
        nzabv = 20    # Number of points above spray layer
        zabv_inc = np.linspace(0,1,nzabv)[:-1]    # Values at z_1 are added separately
        for k in zabv_inc:
            zK = delspr + k*(z_1 - delspr)    # These zK are absolute heights
            z_t.append(zK)
            z_q.append(zK)
            if feedback == False:
                th_prof.append(th_1 + H_S0pr/G_S*(np.log(z_1/zK) - (psiH_1 - stabIntH(zK/L))))
                q_prof.append(  q_1 + H_L0pr/G_L*(np.log(z_1/zK) - (psiH_1 - stabIntH(zK/L))))
            elif feedback == True:
                th_prof.append(th_1 +   H_S1/G_S*(np.log(z_1/zK) - (psiH_1 - stabIntH(zK/L))))
                q_prof.append(  q_1 +   H_L1/G_L*(np.log(z_1/zK) - (psiH_1 - stabIntH(zK/L))))
        # 4. Lowest model level
        z_t.append(z_1)
        z_q.append(z_1)
        th_prof.append(th_1)
        q_prof.append(q_1)
        # Gather data for outputting
        z_t     = np.array(z_t)
        z_q     = np.array(z_q)
        th_prof = np.array(th_prof)    # Should be used with z_t
        q_prof  = np.array(q_prof)    # Should be used with z_q
        p_prof = np.array([p_0 - rho_a*g*z_t[k,:,:] for k in range(np.shape(z_t)[0])])    # Pressure corresponding to z_t [Pa]
        t_prof = th_prof*(p_t/1e5)**0.286    # Temp profile [K], should be used with z_t
        sMy0_prof = satratio(t_prof,p_prof,q_prof,0.99999) - y0    # s - y0 profile [-], we mix values at z_t and z_q to ensure saturation at the roughness height
        for k in range(np.shape(z_t)[0]):
            z_t[k,:,:][~SP] = np.nan
            z_q[k,:,:][~SP] = np.nan
            t_prof[k,:,:][~SP] = np.nan
            th_prof[k,:,:][~SP] = np.nan
            q_prof[k,:,:][~SP] = np.nan
            sMy0_prof[k,:,:][~SP] = np.nan
        profiles = [z_t,z_q,t_prof-273.15,th_prof-273.15,q_prof*1000,sMy0_prof]    # [m], [m], [C], [C], [g kg-1], [-]

    # 12. Return analysis results ===========================================================
    return [H_S0         ,H_L0      ,H_S1  ,H_L1      ,H_Tspr   ,H_Sspr    ,H_Rspr    ,H_Lspr           ,alpha_S    ,beta_S,\
            beta_L       ,gamma_S   ,t_zT  ,delt_zref ,delq_zref,r0        ,M_spr*1000,dmdr0/1e6        ,a_T        ,a_R,\
            ET_bar       ,ER_bar    ,ET    ,ER        ,tauR     ,profiles  ,zR        ,rf*1e6           ,s_zR - y0  ,zT,\
            wetdep_zT    ,U_10N     ,t_10N ,wetdep_10N,q_10N    ,s_10N - y0,H_S0pr    ,H_L0pr           ,q_0 - q_10N,Ch10N,\
            Cq10N        ,Ck10N     ,None  ,None      ,None     ,delta_r0  ,dels_zref ,U_10             ,gamma_L    ,H_Rspr_IG,\
            ustar        ,z0        ,z0t   ,z0q       ,gf       ,thstar_tot,qstar_tot ,tau              ,Cd10N      ,S_10N,\
            th_0 - th_10N,z0t_EQ    ,z0q_EQ,rho_a     ,thvsstar ,H_BS1     ,Cbs10N    ,thvs_0 - thvs_10N,L          ,q_0,\
            thvstar      ,thvstar_pr]

def update_allHFs(H_Sspr_0,H_Rspr_0,H_Lspr_0,allHFs_params):
    """
    Update spray and total heat fluxes for a single gridpoint using a previous
    set of values for spray heat fluxes.
    """
    r0_rng,r0,delta_r0,v_g,zT,tauf,tauT,t_zR_ig,q_zR_ig,Fp,dmdr0,\
            p_0,gammaWB,y0,t_0,rho_a,Dv_a,xs,delspr,z0t,z0q,L,\
            th_0,q_0,G_S,G_L,H_S0pr,H_L0pr,gamma_S,gamma_L,M_spr,\
            Lv,cp_a,rho_sw,nu,Phi_s,Mw,Ms,cp_sw,g,zRvaries = allHFs_params
    # 1. Calculate interfacial and total heat fluxes using previous values of spray heat fluxes
    H_S1_0 = H_S0pr + gamma_S*(H_Sspr_0 - H_Rspr_0)
    H_L1_0 = H_L0pr + gamma_L*H_Lspr_0
    H_S0_0 = H_S1_0 - (H_Sspr_0 - H_Rspr_0)
    H_L0_0 = H_L1_0 - H_Lspr_0
    # 2. Get thermodynamic parameters for H_Tspr
    t_zT,wetdep_zT,tWB_zT,tdropf = thermo_HTspr(zT,tauf,tauT,p_0,gammaWB,y0,t_0,rho_a,\
            delspr,z0t,z0q,L,th_0,q_0,G_S,G_L,H_S0_0,H_L0_0,H_Sspr_0,H_Rspr_0,H_Lspr_0,Lv,cp_a,g)
    # 3. Get thermodynamic parameters for H_Rspr
    s_zR,tauR,zR,rf = thermo_HRspr(r0_rng,r0,v_g,t_zR_ig,q_zR_ig,Fp,tauf,p_0,gammaWB,y0,\
            t_0,rho_a,Dv_a,xs,delspr,z0t,z0q,L,th_0,q_0,G_S,G_L,H_S0_0,H_L0_0,\
            H_Sspr_0,H_Rspr_0,H_Lspr_0,Lv,cp_a,rho_sw,nu,Phi_s,Mw,Ms,g,zRvaries)
    # 4. Calculate updated spray heat fluxes using new thermodynamic parameters
    dHTsprdr0 = cp_sw*(t_0 - tdropf)*dmdr0    # H_Tspr per increment in r0 [W m-2 m-1]
    dHSsprdr0 = cp_sw*np.sign(t_0 - tWB_zT)*np.minimum(np.abs(t_0 - tdropf),np.abs(t_0 - t_zT))*dmdr0    # H_Sspr per increment in r0 [W m-2 m-1]
    dHRsprdr0 = Lv*(1 - (rf/r0)**3)*dmdr0    # H_Rspr per increment in r0 [W m-2 m-1]
    H_Tspr = np.dot(dHTsprdr0,delta_r0)    # Spray HF due to temp change [W m-2]
    H_Sspr = np.dot(dHSsprdr0,delta_r0)    # Spray SHF [W m-2]
    H_Rspr = np.dot(dHRsprdr0,delta_r0)    # Spray HF due to size change [W m-2]
    H_Lspr = H_Rspr + H_Tspr - H_Sspr    # Spray LHF [W m-2]
    # 5. Calculate updated interfacial and total heat fluxes with new values of spray heat fluxes
    H_S1 = H_S0pr + gamma_S*(H_Sspr - H_Rspr)
    H_L1 = H_L0pr + gamma_L*H_Lspr
    H_S0 = H_S1 - (H_Sspr - H_Rspr)
    H_L0 = H_L1 - H_Lspr
    return (H_Tspr,H_Sspr,H_Rspr,H_Lspr,H_S0,H_L0,H_S1,H_L1,t_zT,wetdep_zT,s_zR,tauR,zR,rf,dHTsprdr0,dHRsprdr0)

def feedback_fsolve_residual(H_spr,*allHFs_params):
    """
    Residuals for spray heat fluxes, used when calculating spray feedback using fsolve.
    Performs calculations for a single gridpoint.
    """
    H_Sspr_0,H_Rspr_0,H_Lspr_0 = H_spr
    allHFs_rtrn = update_allHFs(H_Sspr_0,H_Rspr_0,H_Lspr_0,allHFs_params)
    Res_S = allHFs_rtrn[1] - H_Sspr_0    # Residual for H_Sspr [W m-2]
    Res_R = allHFs_rtrn[2] - H_Rspr_0    # Residual for H_Rspr [W m-2]
    Res_L = allHFs_rtrn[3] - H_Lspr_0    # Residual for H_Lspr [W m-2]
    return (Res_S,Res_R,Res_L)

