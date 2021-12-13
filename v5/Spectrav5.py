################################################
'''
Code is for Lab 3 Experimental Physics PHAS0051 in UCL
This code defines a main class which stores a particular spectrum data 
and some characteristic values, like positions of photepeaks, parameters
for calibration, etc., and whose methods can accomplish all tasks in the 
lab script, including calibration and spectrum analysis.

Author: Yaocheng Li (Randy)
Email: yaocheng.li.19@ucl.ac.uk

'''
################################################
#from typing_extensions import TypeVarTuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl #Some more style nonsense
from scipy.stats import chi2 # chi2
from scipy.optimize import curve_fit
from scipy.stats import linregress #linear regression

import inspect #find dof

mpl.rcParams['figure.figsize'] = [8, 4] #Inches... of course it is inches
mpl.rcParams["legend.frameon"] = True
mpl.rcParams['figure.dpi']=300 # dots per inch

class spectra:

    #construct
    def __init__(self,sample,refpeaks = None,bkg = None,label = None):

        self.sample = sample #the source we study 
        self.bkg = bkg #background radiatioin spectrum
        self.ch = np.arange(len(self.sample)) #get an array of channels
        self.label = label #how you name the source (spectrum)
        # value, err
        self.refpeaks = refpeaks #just for recording
        #default values of calibration data [value, error]
        self.m = [1,0]
        self.c = [0,0]
        self.q = [0,0]

        self.ifcali = False #calibration indicator

#=========================================calibration=======================================#
    cali_peaks = None #in ch's, will be shaped by function add_peaks 
    # structure of cali_peaks: [[peak1,err],[peak2,err],...]
    #para_label = ["A","mu","sigma","m","c"] #label the parameters used in our model
    para_label = ["A","mu","sigma","c"]#v3model

    ##Fundatmental Functions

    ##Auxiliary Functions
    #our model
    def model(self,x,A,mu,sig,m,c):
            # hypothesis function
            return A*np.exp(-(x-mu)**2/2/sig**2)+m*x+ c # Gaussian above a line segment(for background)


    #Chi^2 test
    def chi2test(self,y,y_exp,title = None,printout = False):
        
        dof = len(y) - len(inspect.getargspec(self.model)[0]) + 2#remove self and x

        chi = np.sum((y-y_exp)**2/y_exp)

        p_value = 1 - chi2.cdf(chi, dof)#chi2 probability, `chi2` in scipy is applied.

        
        print("===========================Chi^2 test==========================")

        if title != None:
            print(title)
            print("===============================================================")

        if printout == True:
            print("Chi^2 statistic:\t{0:.3f}\nDegree of freedom:\t{1}\n".format(chi,dof))
            print("The Chi^2 probabiity is ",p_value)
        print()#spacing

        return (p_value,chi,dof)

    # #get counting error (obeying poisson distribution)
    # def cerr(self,x):
    #     return np.sqrt(x)
 
    #Energy Scale error(from errors of calibration data):
    def E_err(self,E):
        #return the true error
            #return np.sqrt(((E-self.c[0])*self.m[1]/self.m[0])**2+self.c[1]**2)#<-for q=dq=0 
        return np.sqrt((self.ch_of(E)**2*self.q[1])**2+(self.ch_of(E)*self.m[1])**2+self.c[1]**2)#
   
    #Calibrate with an external info
    def get_E_scale(self,m,c,q = [0,0]):

        self.m = m
        self.c = c
        self.q = q

        self.E_scale = self.q[0]*self.ch**2+self.m[0]*self.ch + self.c[0]
        self.ifcali = True

    #passing out the calibration data
    def cali_info(self,printout = True):

        if printout == True:
            #show m and c
            print("=======================================")
            if self.label == None:
                print("Calibration constants")
            else: 
                print("Calibration constants in {} spectra".format(self.label))
            print("=======================================")
            print(
            "m = {0:f}+- {1:f}\nc = {2:f}+- {3:f}\np = {4:f}+- {5:f}".format(self.m[0],self.m[1],self.c[0],self.c[1],self.q[0],self.q[1])                                                         
                )
            print()#spacing

        return (self.m,self.c,self.q)
    
    #=====Peak manipulation===================================================
    #add a peak(in ch's)
    def cali_add_peak(self,peak):
        '''
        This function insert a peak value ([p,err]) into self.cali_peaks, where 
        resulting peak list will be ordered increasingly.
        '''
        if np.any(peak == self.cali_peaks):
            print("Duplicate")
        else:
            if np.all(self.cali_peaks == None):
                self.cali_peaks = np.array([peak])
            else:
                self.cali_peaks = np.insert(self.cali_peaks,np.sum(self.cali_peaks[:,0]<peak[0]),[peak],axis = 0)

    #add multiple peaks
    def cali_add_peaks(self,peaks):

        for p in peaks:
            self.cali_add_peak(p)

    #clear recorded peaks
    def cali_clear_peaks(self):
        self.cali_peaks = None

    #show all recorded peaks
    def cali_print_peaks(self):
        #fomal print photopeaks
        print("======================")
        if self.label == None:
            print("Photopeaks (ch)") 
        else:
            print("Photopeaks (ch) of {}".format(self.label))
        print("======================")

        for i,p in enumerate(self.cali_peaks):
            print(
            "peak{0} = {1:f}+- {2:f}\n".format(i+1,p[0],p[1])                                                         
                )
    #===================================================================================

    ##Central Methods - Calibration=====================================================
    #Set up
    def cali_exam(self,chmin = 0,chmax = None,log = False,fcolor = "#607DD8"):
        #plot a range of spectrum to determine a good range for fitting
        if chmax == None:
            chmax = len(self.sample)
    
        fig, axs = plt.subplots()

        if self.label != None:
            axs.bar(self.ch[chmin:chmax],self.sample[chmin:chmax],color = fcolor,zorder = 1,label = self.label,width = 1,alpha = 1)
        else:
            axs.bar(self.ch[chmin:chmax],self.sample[chmin:chmax],color = fcolor,zorder = 1,label = "Sample",width = 1,alpha = 1)

        axs.set_xlabel("chs")

        if log == True:
            axs.set_yscale("log")
            axs.set_ylabel("Counts (Logarithmic scale)")
        else:
            axs.set_ylabel("Counts")

        if self.label == None:
            axs.set_title("The uncalibrated spectrum of the sample: channel {} to {}".format(chmin,chmax))
        else:
            axs.set_title("The uncalibrated spectrum of {}: channel {} to {}".format(self.label,chmin,chmax))

        axs.legend(frameon = True)

        #plt.show()

    ##Fit a Gaussian(our model) in a range of data about a peak to get peak value(in ch's)
    def cali_fit_peak(self,chmin,chmax,barcolor = "#5DC6EE",ecolor = "#E45858",fcolor = "#E45858",log = True,plot = False,axs = None , plotoffset = 3,chi2test = False,printout = False, returnall = False):
        '''
        log: if use logarithmatic scale
        plot: if show a plot of fitting curve with histogram
        ch2test: if comfirm a chi square test
        printout: if show all best fit parameters with errors
        returnall: if return all parameters(only the peak returned by default)
        colors: stylish
        '''
        Ch = self.ch[chmin:chmax]
        Co = self.sample[chmin:chmax]
        #fit and plot the curve
        popt, pcov = curve_fit(self.model, Ch,Co,
                                sigma = np.sqrt(Co),
                                p0 = [np.max(Co),
                                    Ch[np.argmax(Co)],
                                    1,
                                    (Co[-1]-Co[0])/(Ch[-1]-Ch[0]),
                                    Ch[-1]*np.abs((Co[-1]-Co[0])/(Ch[-1]-Ch[0]))]
                                    #Co[0]]
                                )
        p = (popt,np.sqrt(np.diag(pcov)))


        if plot == True:#Show fitting with bars

            if axs == None:
                fig, axs = plt.subplots()

            Chp = self.ch[chmin-plotoffset:chmax+plotoffset]
            Cop = self.sample[chmin-plotoffset:chmax+plotoffset]
            if self.label == None:
                axs.bar(Chp,height=Cop,yerr = np.sqrt(Cop),color = barcolor,ecolor = ecolor,label = "sample",width=1)
            else:
                axs.bar(Chp,height=Cop,yerr = np.sqrt(Cop),color = barcolor,ecolor = ecolor,label = self.label,width=1)

            xline = np.linspace(np.min(Chp),np.max(Chp),30*len(Chp))
            axs.plot(xline,self.model(xline,*popt),color = fcolor,label = "Gaussian Fit")

            axs.set_xlabel("Ch.")

            if log == True:
                axs.set_yscale("log")
                axs.set_ylabel("Counts (Logarithmic scale)")
            else:
                axs.set_ylabel("Counts")

            if self.label == None:
                axs.set_title("Fit Model to the peak in uncalibrated {} spectrum between Channel {} to {}".format("sample",chmin,chmax))
            else:
                axs.set_title("Fit Model to the peak in uncalibrated {} spectrum between Channel {} to {}".format(self.label,chmin,chmax))

            axs.legend()
    
            #plt.show()


        if printout == True:#print out results of fitting
            label = self.para_label
            print("============================Parameters=============================")
            print("Fitting to the peak of the sample between channel {} - {}".format(chmin,chmax))
            print("===================================================================")

            for i in range(len(label)):
                print(label[i],"=",p[0][i],"+-",p[1][i])
                print()

        if chi2test == True:
            self.chi2test(Co,self.model(Ch,*popt),title = "Fitting to the peak of the sample between channel {} - {}".format(chmin,chmax),printout = True)

        if returnall == True:#if return everything
            return (popt,np.sqrt(np.diag(pcov)))

        else:
            return (popt[1],np.sqrt(np.diag(pcov))[1])

        
    ##linear regression=Get m,c(,and q)
    def get_cali(self,refpeaks = None,deg = 1,plot = False,printout = False,perc = [5,5,5]):
        '''
        refpeaks: update the reference values
        deg: 1 for linear, 2 for quadratic relation
        plot&printot: if show results

        return cali_info
        '''
        
        if refpeaks != None:
            self.refpeaks = refpeaks

        if np.all(self.refpeaks == None):
            print("***Input a reference pleas***")
        
        elif np.all(self.cali_peaks == None):
            print("***No peaks recorded***")

        elif len(self.cali_peaks) <= 1:
            print("***Too few peaks***")

        else:

            x = self.cali_peaks[:,0]
            x_err = self.cali_peaks[:,1]

            if len(x) == 2:

                m = (self.refpeaks[1] - self.refpeaks[0])/(x[1]-x[0])
                c = self.refpeaks[0] - m*x[0]

                m_err = m*np.sqrt(np.sum(np.array([x_err[0],x_err[1]])**2))/(x[1]-x[0])
                c_err = m*x_err[0]

                self.get_E_scale([m,m_err],[c,c_err])

                if plot == True : #Plot

                    fig, axs = plt.subplots()
                    axs.errorbar(x,self.refpeaks,xerr=x_err,fmt='r.',ecolor = "black",label = "Photopeaks")#error bar too small
                    x_line = np.linspace(x[0]-50,x[-1]+50,100)
                    axs.plot(x_line,c + m*x_line,label = "Fitting: $m = {0:.{b}f}\pm {1:.{b}f},c = {2:.{c}f}\pm {3:.{c}f}$"
                                                                    .format(m,m_err,c,c_err,b = perc[1],c = perc[2]))#error too small

                    axs.set_ylabel("Energy(keV)")
                    axs.set_xlabel("Ch.")

                    axs.set_title("Derive calibration parameter m,c")
                    axs.legend()

                    #plt.show()

                if printout == True:#printor
                    self.cali_info()

                return (m,m_err),(c,c_err),(0,0)

            
            elif len(x) >= 3 and deg == 1:

                res = linregress(x,self.refpeaks)

                m,c = res.slope, res.intercept

                m_err,c_err = res.stderr, res.intercept_stderr

                self.get_E_scale([m,m_err],[c,c_err],[0,0])

                if plot == True : #Plot

                    fig, axs = plt.subplots()
                    axs.errorbar(x,self.refpeaks,xerr=x_err,fmt='r.',ecolor = "black",label = "Photopeaks")#error bar too small
                    x_line = np.linspace(x[0]-50,x[-1]+50,100)
                    axs.plot(x_line,c + m*x_line,label = "Fitting: $m = {0:.{b}f}\pm {1:.{b}f},c = {2:.{c}f}\pm {3:.{c}f}$"
                                                                    .format(m,m_err,c,c_err,b = perc[1],c = perc[2]))

                    axs.set_ylabel("Energy(keV)")
                    axs.set_xlabel("Ch.")

                    axs.set_title("Derive calibration parameter m,c")
                    axs.legend()

                    #plt.show()

                if printout == True:#printor
                    self.cali_info()

                return (m,m_err),(c,c_err),(0,0)
            

            elif len(x) >= 3 and deg == 2:

                p,cov = np.polyfit(x,self.refpeaks,2,w = 1/x_err,cov='unscaled')

                q,m,c = p

                m_err,c_err,q_err = np.sqrt(np.diag(cov))

                self.get_E_scale([m,m_err],[c,c_err],[q,q_err])

                if plot == True : #Plot

                    fig, axs = plt.subplots()
                    axs.errorbar(x,self.refpeaks,xerr=x_err,fmt='r.',ecolor = "black",label = "Photopeaks")#error bar too small
                    x_line = np.linspace(x[0]-50,x[-1]+50,100)
                    axs.plot(x_line,c + m*x_line + q*x_line**2,label = "Fitting: $q = {0:.{a}f}\pm {1:.{a}f},m = {2:.{b}f}\pm {3:.{b}f},c = {4:.{c}f}\pm {5:.{c}f}$"
                                                                    .format(q,q_err,m,m_err,c,c_err,a = perc[0],b = perc[1],c = perc[2]))

                    axs.set_ylabel("Energy(keV)")
                    axs.set_xlabel("Ch.")

                    axs.set_title("Derive calibration parameter q,m,c")
                    axs.legend()

                    #plt.show()

                if printout == True:#printor
                    self.cali_info()

                return (m,m_err),(c,c_err),(q,q_err)

            else:
                print("***degree should be 1 or 2***")


##=======================Analysis========================##
    #Constants needed
    m_e = 9.1093837015e-31 # kg
    c_light = 299792458 # m/s
    Samplepeaks = None #[[peak,error],...] #will be shaped by add_peaks
    Sample_FWHMs = None #for resolution
    Sample_As = None #for efficiency

    ##Auxiliary Functions:
    
    #======Record peaks=============================================================================
    def add_peak(self,peak):
        if np.any(peak == self.Samplepeaks):
            print("Duplicate")
        else:
            if np.all(self.Samplepeaks == None):
                self.Samplepeaks = np.array([peak])
            else:
                self.Samplepeaks = np.insert(self.Samplepeaks,np.sum(self.Samplepeaks[:,0]<peak[0]),[peak],axis = 0)

    #add multiple peaks
    def add_peaks(self,peaks):

        for p in peaks:
            self.add_peak(p)
            

    #remove recorded peaks
    def clear_peaks(self):
        self.Samplepeaks = None


    def print_peaks(self):
        #fomal print all recorded photopeaks
        print("======================")
        if self.label == None:
            print("Photopeaks") 
        else:
            print("Photopeaks of {}".format(self.label))
        print("======================")

        for i,p in enumerate(self.Samplepeaks):
            print(
            "peak{0} = {1:f}+- {2:f}\n".format(i+1,p[0],p[1])                                                         
                )
    #===============================================================================================


    #======Record FWHM and A=============================================================================
    def add_FWHM(self,sigma,order = None):

        sigma = 2*np.sqrt(2*np.log(2))*sigma

        if np.all(self.Sample_FWHMs == None):
            self.Sample_FWHMs = np.array([sigma])
        else:
            if order == None:
                self.Sample_FWHMs = np.insert(self.Sample_FWHMs,np.shape(self.Sample_FWHMs)[0],[sigma],axis = 0)
            else:
                self.Sample_FWHMs = np.insert(self.Sample_FWHMs,order,[sigma],axis = 0)

    #add multiple peaks
    def add_FWHMs(self,FWHM,order = None):

        if order == None:
            for p in FWHM:
                self.add_FWHM(p)
        else:
            for i,p in enumerate(FWHM):
                self.add_FWHM(p,order = order[i])      

    #remove recorded peaks
    def clear_FWHMs(self):
        self.Sample_FWHMs = None

    def print_FWHMs(self):
        #fomal print all recorded photopeaks
        print("======================")
        if self.label == None:
            print("FWHMs") 
        else:
            print("FWHMs of {}".format(self.label))
        print("======================")

        for i,p in enumerate(self.Sample_FWHMs):
            print(
            "The FWHM of the peak{0} = {1:f}+- {2:f}\n".format(i+1,p[0],p[1])                                                         
                )

    #============= A ==============
    def add_A(self,A,order = None):
    
        if np.all(self.Sample_As == None):
            self.Sample_As = np.array([A])
        else:
            if order == None:
                self.Sample_As = np.insert(self.Sample_As,np.shape(self.Sample_As)[0],[A],axis = 0)
            else:
                self.Sample_As = np.insert(self.Sample_As,order,[A],axis = 0)

    #add multiple peaks
    def add_As(self,A,order = None):

        if order == None:
            for p in A:
                self.add_A(p)
        else:
            for i,p in enumerate(A):
                self.add_A(p,order = order[i])      

    #remove recorded peaks
    def clear_As(self):
        self.Sample_As = None

    def print_As(self):
        #fomal print all recorded photopeaks
        print("======================")
        if self.label == None:
            print("N_ps") 
        else:
            print("Nps of {}".format(self.label))
        print("======================")

        for i,p in enumerate(self.Sample_As):
            print(
            "The N_p of the peak{0} = {1:f}+- {2:f}\n".format(i+1,p[0],p[1])                                                         
                )
    #===============================================================================================

    #============= Integrated Function: ============================================================
    def record_popt(self,popt):

        peak = popt[:,1]
        FWHM = popt[:,2]
        A = popt[:,0]

        if np.any(peak == self.Samplepeaks):
            print("Duplicate")
        else:
            if np.all(self.Samplepeaks == None):
                self.Samplepeaks = np.array([peak])
                self.add_FWHM(FWHM)
                self.add_A(A)
            else:
                self.Samplepeaks = np.insert(self.Samplepeaks,np.sum(self.Samplepeaks[:,0]<peak[0]),[peak],axis = 0)
                self.add_FWHM(FWHM, order = np.sum(self.Samplepeaks[:,0]<peak[0]))
                self.add_A(A,order = np.sum(self.Samplepeaks[:,0]<peak[0]))


    def record_popts(self,popts):

        for p in popts:
            self.record_popt(p)

    def clear_records(self):
        self.Samplepeaks = None #[[peak,error],...] #will be shaped by add_peaks
        self.Sample_FWHMs = None #for resolution
        self.Sample_As = None #for efficiency
    #===============================================================================================


    #Retrive channel number from E
    def ch_of(self,E):
        m,c,q = self.m[0],self.c[0],self.q[0]

        if q == 0:
            return np.rint((E - c)/m)
        else: 
            return np.rint((np.sqrt(m**2-4*q*(c-E))-m)/(2*q))

    #dimension transformation
    def keVtoJoule(self,E,Inverse = False):
        if Inverse == False:
            return 1.6022e-16*E
        if Inverse == True:
            return E/1.6022e-16

    #define a Energy function due to Compton Scattering with respect to the incident angle theta
    def E_s(self,E,theta):
    # E = energy before
        return E/(1+E/self.keVtoJoule((self.m_e*self.c_light**2),Inverse=True)*(1-np.cos(theta)))

    #Compton Cut-off E
    def E_c(self,E):
        #get cut-off energy with their errors
        Ec = E - self.E_s(E,np.pi)
        dE = self.E_err(E)

        err =  (Ec**2/E**2)*dE #error on theta = pi cut-off
        
        return np.transpose([Ec,err])#err(phi = 0) = 0 


    ## Central Methods - Analysis
    #==================================================
    #Plot spectrum(ranging from Emin to Emax)
    #==================================================
    def spectra(self,Emin = 0,Emax = 3000,timescale = 1,log = False,fcolor = "#434B9A",bcolor = '#F2F365',bkg = False,axs = None,bkgalpha = 1,fgalpha = 1):
        '''
        timescale: scale of the background
        bkg: if include background spectrum
        axs: plot the spectra on an external canvas
        '''
        if self.ifcali == False:#if calibrated
            print("\n*****Please Calibrate first.*****\n")

        else:
            #Plot calibrated spectra in a chosen range
            if self.ch_of(Emin)>0:
                xmin = int(self.ch_of(Emin))
            else:
                xmin = 0
            xmax = int(self.ch_of(Emax))#retrieve channel scale

            if axs == None :
                fig,axs = plt.subplots()

            if bkg == True:
                if self.label != None:
                    axs.bar(self.E_scale[xmin:xmax],height=self.sample[xmin:xmax],color = fcolor,zorder = 1,label = "foreground,{}".format(self.label),alpha = fgalpha)
                else:
                    axs.bar(self.E_scale[xmin:xmax],height=self.sample[xmin:xmax],color = fcolor,zorder = 1,label = "foreground",alpha = fgalpha)

                axs.bar(self.E_scale[xmin:xmax],height=timescale*self.bkg[xmin:xmax],zorder = 5,alpha = bkgalpha,color = bcolor,label = "background, time scale = {0:f}".format(timescale))
            
            else:
                if self.label != None:
                    axs.bar(self.E_scale[xmin:xmax],height=self.sample[xmin:xmax],color = fcolor,zorder = 1,label = self.label,alpha = fgalpha)
                else:
                    axs.bar(self.E_scale[xmin:xmax],height=self.sample[xmin:xmax],color = fcolor,zorder = 1,label = "Sample",alpha = fgalpha)

            axs.set_xlabel("Energy/keV")

            if log == True:
                axs.set_yscale("log")
                axs.set_ylabel("Counts (Logarithmic scale)")
            else:
                axs.set_ylabel("Counts")

            if self.label == None:
                axs.set_title("The spectrum of the sample: from E = {} to {} keV".format(Emin,Emax))
            else:
                axs.set_title("The spectrum of {}: from E = {} to {} keV".format(self.label,Emin,Emax))

            axs.legend()
        
            

            #plt.show()

    #=====================================
    ##Fit a Gaussian
    #=====================================
    def fit_peak(self,Emin,Emax,barcolor = "#9682D2",ecolor = "#DD472C",fcolor = "#DD472C",log = True,plot = False,axs = None,plotoffset = 3, chi2test = False,printout = False, peakonly = False,returnall = False,width=0.2):
        '''
        Fit model to a peak on calibrated spectrum
        '''
        if self.ifcali == False:#if calibrated
            print("\n*****Please Calibrate first.*****\n")

        else:
            #Plot calibrated spectra in a chosen range
            xmin = int(self.ch_of(Emin))
            xmax = int(self.ch_of(Emax))#retrieve channel scale

            E = self.E_scale[xmin:xmax]
            Sample = self.sample[xmin:xmax]

            #fit and plot the curve
            popt, pcov = curve_fit(self.model, E,Sample,
                                    sigma = np.sqrt(Sample),
                                    p0 = [np.max(Sample),
                                        E[np.argmax(Sample)],
                                        1,
                                        (Sample[-1]-Sample[0])/(E[-1]-E[0]),
                                        E[-1]*np.abs((Sample[-1]-Sample[0])/(E[-1]-E[0]))]
                                        #Sample[0]]
                                    )
            p = (popt,np.sqrt(np.diag(pcov)))

            if plot == True:#Show fitting with bars

                if axs == None:
                    fig, axs = plt.subplots()

                Ep = self.E_scale[xmin-plotoffset:xmax+plotoffset]
                Samplep = self.sample[xmin-plotoffset:xmax+plotoffset]

                if self.label != None:
                    axs.bar(Ep,height=Samplep,yerr = np.sqrt(Samplep),color = barcolor,ecolor = ecolor,label = self.label,width = width)
                else:
                    axs.bar(Ep,height=Samplep,yerr = np.sqrt(Samplep),color = barcolor,ecolor = ecolor,label = "Sample",width = width)
                xline = np.linspace(np.min(Ep),np.max(Ep),30*len(Ep))
                axs.plot(xline,self.model(xline,*popt),color = fcolor,label = "Gaussian Fit")

                axs.set_xlabel("Energy/keV")
                
                if log == True:
                    axs.set_yscale("log")
                    axs.set_ylabel("Counts (Logarithmic scale)")
                else:
                    axs.set_ylabel("Counts")

                if self.label == None:
                    axs.set_title("Fit model to the peak in sample spectrum between E = {} and {} keV".format(Emin,Emax))
                else:
                    axs.set_title("Fit model to the peak in {} spectrum between E = {} and {} keV".format(self.label,Emin,Emax))

                axs.legend()
        
                #plt.show()


            if printout == True:#print out results of fitting
                label = self.para_label
                print("============================Parameters=============================")
                if self.label != None:
                    print("Fitting to the peak of {} between Energy {} - {} keV".format(self.label,Emin,Emax))
                else:
                    print("Fitting to the peak of the sample between Energy {} - {} keV".format(Emin,Emax))
                print("===================================================================")

                for i in range(len(label)):
                    print(label[i],"=",p[0][i],"+-",p[1][i])
                    print()
                print("Energy Scale Error at mu = :{}".format(self.E_err(p[0][1])))
                print()#spacing

            if chi2test == True:
                if self.label != None:
                    self.chi2test(Sample,self.model(E,*popt),title = "Fitting to the peak of {} between Energy {} - {} keV".format(self.label,Emin,Emax),printout = True)
                else:
                    self.chi2test(Sample,self.model(E,*popt),title = "Fitting to the peak of the sample between Energy {} - {} keV".format(Emin,Emax),printout = True)
            
            if returnall == True:#if return everything
                return np.array([popt,np.sqrt(np.diag(pcov))])

            elif peakonly == True:
                return np.array([popt[1],np.sqrt(np.diag(pcov))[1]])

            else:
                return np.array([popt[0:3],np.sqrt(np.diag(pcov))[0:3]])

    #==========================================
    #label the peaks
    #==========================================
    def label_peaks(self,axs,n1 = 0, n2 = None, E_err = True, label = None, tcolor = 'black',fontsize = 12,halign ='center',valign = 'bottom',perc = 2):
        
        if label == None:
            if E_err == True:
                if n2 == None:
                    for i,p in enumerate(self.Samplepeaks[n1:,0]):
                        top = np.max(self.sample[int(self.ch_of(p))-3:int(self.ch_of(p))+3])
                        axs.text(p,top,"$peak_{0}$\n${1:.{perc}f}\pm{2:.{perc}f}$keV".format(n1+i+1,self.Samplepeaks[n1+i,0],self.E_err(self.Samplepeaks[n1+i,0]),perc=perc),color = tcolor,fontsize=fontsize,horizontalalignment=halign,verticalalignment=valign)
                else:
                    for i,p in enumerate(self.Samplepeaks[n1:n2,0]):
                        top = np.max(self.sample[int(self.ch_of(p))-3:int(self.ch_of(p))+3])
                        axs.text(p,top,"$peak_{0}$\n${1:.{perc}f}\pm{2:.{perc}f}$keV".format(n1+i+1,self.Samplepeaks[n1+i,0],self.E_err(self.Samplepeaks[n1+i,0]),perc=perc),color = tcolor,fontsize=fontsize,horizontalalignment=halign,verticalalignment=valign)
                
            else:
                if n2 == None:
                    for i,p in enumerate(self.Samplepeaks[n1:,0]):
                        top = np.max(self.sample[int(self.ch_of(p))-3:int(self.ch_of(p))+3])
                        axs.text(p,top,"$peak_{0}$\n${1:.{perc}f}\pm{2:.{perc}f}$keV".format(n1+i+1,self.Samplepeaks[n1+i,0],self.Samplepeaks[n1+i,1],perc=perc),color = tcolor,fontsize=fontsize,horizontalalignment=halign,verticalalignment=valign)
                else:
                    for i,p in enumerate(self.Samplepeaks[n1:n2,0]):
                        top = np.max(self.sample[int(self.ch_of(p))-3:int(self.ch_of(p))+3])
                        axs.text(p,top,"$peak_{0}$\n${1:.{perc}f}\pm{2:.{perc}f}$keV".format(n1+i+1,self.Samplepeaks[n1+i,0],self.Samplepeaks[n1+i,1],perc=perc),color = tcolor,fontsize=fontsize,horizontalalignment=halign,verticalalignment=valign)
        
        else:
            # for i,p in enumerate(self.Samplepeaks[n1:n2,0]):
            #             top = np.max(self.sample[int(self.ch_of(p))-3:int(self.ch_of(p))+3])
            #             axs.text(p,top,label[i+n1],color = tcolor,fontsize=fontsize,horizontalalignment=halign,verticalalignment=valign)
            
            if E_err == True:
                if n2 == None:
                    for i,p in enumerate(self.Samplepeaks[n1:,0]):
                        top = np.max(self.sample[int(self.ch_of(p))-3:int(self.ch_of(p))+3])
                        axs.text(p,top,"{0}\n${1:.{perc}f}\pm{2:.{perc}f}$keV".format(label[i+n1],self.Samplepeaks[n1+i,0],self.E_err(self.Samplepeaks[n1+i,0]),perc=perc),color = tcolor,fontsize=fontsize,horizontalalignment=halign,verticalalignment=valign)
                else:
                    for i,p in enumerate(self.Samplepeaks[n1:n2,0]):
                        top = np.max(self.sample[int(self.ch_of(p))-3:int(self.ch_of(p))+3])
                        axs.text(p,top,"{0}\n${1:.{perc}f}\pm{2:.{perc}f}$keV".format(label[i+n1],self.Samplepeaks[n1+i,0],self.E_err(self.Samplepeaks[n1+i,0]),perc=perc),color = tcolor,fontsize=fontsize,horizontalalignment=halign,verticalalignment=valign)
                
            else:
                if n2 == None:
                    for i,p in enumerate(self.Samplepeaks[n1:,0]):
                        top = np.max(self.sample[int(self.ch_of(p))-3:int(self.ch_of(p))+3])
                        axs.text(p,top,"{0}\n${1:.{perc}f}\pm{2:.{perc}f}$keV".format(label[i+n1],self.Samplepeaks[n1+i,0],self.Samplepeaks[n1+i,1],perc=perc),color = tcolor,fontsize=fontsize,horizontalalignment=halign,verticalalignment=valign)
                else:
                    for i,p in enumerate(self.Samplepeaks[n1:n2,0]):
                        top = np.max(self.sample[int(self.ch_of(p))-3:int(self.ch_of(p))+3])
                        axs.text(p,top,"{0}\n${1:.{perc}f}\pm{2:.{perc}f}$keV".format(label[i+n1],self.Samplepeaks[n1+i,0],self.Samplepeaks[n1+i,1],perc=perc),color = tcolor,fontsize=fontsize,horizontalalignment=halign,verticalalignment=valign)
        
        

    #========================================
    #Compton Edge indicator
    #========================================
    def ComptonEdge(self,axs,n1 = 0,n2 = None,lstyle = '--',lcolor = '#F5DC5F',tcolor = 'black',prec = 2,fontsize = 10,halign = 'center',valign = 'bottom'):
        
        if n2 == None:
            E_c = self.E_c(self.Samplepeaks[n1:,0])
        else:
            E_c = self.E_c(self.Samplepeaks[n1:n2,0])

        for i,Ec in enumerate(E_c):
            E,err = Ec[0],Ec[1]
            top = np.max(self.sample[int(self.ch_of(E))-3:int(self.ch_of(E))+3])

            #Label
            axs.text(E,top,"Compton Edge from\n $peak_{0}$ energy\n$E_c =( {1:.{prec}f} \pm {2:.{prec}f} )keV$".format(n1+i+1,E,err,prec = prec),
                        color = tcolor,
                        fontsize=fontsize,horizontalalignment=halign,verticalalignment=valign)
            #verticals
            axs.axvline(E,LineStyle = lstyle,color = lcolor)


    #========================================
    #Resolution
    #========================================
    
    def Fit_Resolution(self,axs=None,fmt = 'rx',ecolor = "black",plot = True,printout = True):

        w,cov = np.polyfit(self.Samplepeaks[:,0],self.Sample_FWHMs[:,0]**2,1,w = 1/(2*self.Sample_FWHMs[:,0]*self.Sample_FWHMs[:,1]),cov='unscaled')

        dw = np.sqrt(np.diag(cov))

        p = np.transpose([w,dw])

        w_e, dw_e = np.sqrt(p[1,0]),p[1,1]/np.sqrt(p[1,0])/2
        w_d, dw_d = np.sqrt(p[0,0]),p[0,1]/np.sqrt(p[0,0])/2

        self.w_e = np.array([w_e,dw_e])
        self.w_d = np.array([w_d,dw_d])

        self.w_e2 = np.array([p[1,0],p[1,1]])
        self.w_d2 = np.array([p[0,0],p[0,1]])

        if plot == True:
            axs.errorbar(self.Samplepeaks[:,0],self.Sample_FWHMs[:,0]**2,xerr=self.Samplepeaks[:,1],
                        yerr=self.Sample_FWHMs[:,1],fmt=fmt,ecolor = ecolor,label = self.label)#error bar too small

            x = np.linspace(np.min(self.Samplepeaks[:,0])-80,np.max(self.Samplepeaks[:,0])+80,200)
            axs.plot(x, w[0]*x + w[1])

            axs.set_ylabel("$FWHM^2/keV^2$")
            axs.set_xlabel("E/keV")

            axs.set_title("$FWHM^2$ vs Energy")
            axs.legend()

        if printout == True:
            print("=========================================================")
            print("       w_e and w_d determine the energy resolution      ")
            print("=========================================================")
            print("w_e = {} +- {}".format(w_e, dw_e))
            print()
            print("w_d = {} +- {}".format(w_d,dw_d))

        return np.array([[w_d, dw_d],[w_e, dw_e]])

    
    def Resolution(self,E):

        RE2 = self.w_d2[0]*E + self.w_e2[0]
        RE = np.sqrt(RE2)
        R = RE/E
        
        dRE2 = np.sqrt(E**2*self.w_d2[1]**2 + self.w_e2[1]**2)
        dRE = dRE2/2/RE
        dR = dRE/E

        return (R,dR)

