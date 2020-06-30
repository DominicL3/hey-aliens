#!/usr/bin/env python
# A separate function to extract and plot Heimdall candidates
# This script is a modified version of the Heimdall plotting scipt 'trans_freq_time.py'

### THIS SCRIPT IS A MODIFICATION OF PlotCand.py by Vishal Gajjar ###

import os,sys,math
import numpy as np
import glob
from itertools import chain
from os.path import basename
from itertools import tee, izip, izip_longest
import matplotlib

import matplotlib.pyplot as plt

import psrchive as psr
from sigpyproc.Readers import FilReader
import subprocess as sb
import shlex
import time as tt
import pandas as pd

def find_nearidx(array,val):
        idx = (np.abs(array-val)).argmin()
        return idx

def grouper(array,n,fillvalue=None):
	args = [iter(array)]*n
	return izip_longest(*args, fillvalue=fillvalue)

def exeparallel(cmd_array):
     '''
     This will execute an array of commands simultaneously in groups
     number provided
     '''
     ncmd=20 #Number of commands to execute simultaneously
     if ncmd>len(cmd_array): ncmd=len(cmd_array)

     child_processes = [] # list holding all processes spawned by subprocess
     for grpcmd in grouper(cmd_array,ncmd):
		grpcmd1 = list(filter(None,grpcmd)) #Remove None elements from the groups
     		cmd = ' & '.join(grpcmd1)
		print cmd

		proc = sb.Popen(cmd, shell=True)
                child_processes.append(proc)

		while proc.poll() == None:
                        continue
        	#os.system(cmd)

     for proc in child_processes:
                out, err = proc.communicate()
                print("[exeparallel] ERRORS: {}".format(err))
                tt.sleep(1)

def dedispblock(ar,lodm,hidm):
    fpsr = psr.Archive_load(ar)
    toplot = []
    dmstep = 1
    dmrange = range(lodm,hidm,dmstep)
    for dm in dmrange:
        fpsr.remove_baseline()
        fpsr.set_dispersion_measure(dm)
        fpsr.dedisperse()
        ds = fpsr.get_data().squeeze()
        w = fpsr.get_weights().flatten()
        w = w/np.max(w) # Normalized it
        idx = np.where(w==0)[0]
        ds = np.multiply(ds, w[np.newaxis,:,np.newaxis]) # Apply it
        ds[:,idx,:] = np.nan
        data1 = ds[0,:,:]
        time = np.nanmean(data1[:,:],axis=0)
        toplot.append(time)

    tbin = float(fpsr.integration_length()/fpsr.get_nbin())
    taxis = np.arange(0,fpsr.integration_length(),tbin)
    taxis=taxis*1000 #Get to msec
    toplot = np.array(toplot)
    toplot = [list(i) for i in zip(*toplot)]
    toplot = np.transpose(toplot)
    return toplot,taxis

def negDMplot(ar,FTdirection,nchan):
    fpsr = psr.Archive_load(ar)
    fpsr.remove_baseline()
    ds = fpsr.get_data().squeeze()
    w = fpsr.get_weights().flatten()
    w = w/np.max(w) # Normalized it
    idx = np.where(w==0)[0]
    ds = np.multiply(ds, w[np.newaxis,:,np.newaxis]) # Apply it
    ds[:,idx,:] = np.nan
    data = ds[0,:,:]
    if FTdirection == 'nT':
    	ndata = data[...,::-1]
	print "Will be flipped in Time"
    elif FTdirection == 'nF':
	ndata = data[::-1,...]
	print "Will be flipped in freq"
    elif FTdirection == 'nTnF':
	ndata = data[::-1,::-1]
	print "Will be flipped in time and freq"
    else:
	ndata = data
	print "No flip"
    return ndata

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

def plotParaCalc(snr,filter,dm,fl,fh,tint,nchan):
        #Extract block factor plot in seconds
        extimefact = 1.0

        # Total extract time Calc
        # Extract according to the DM delay
        cmd = 'dmsmear -f %f -b %f -n 2048 -d ' % (fl+(fh-fl)/2,fh-fl) + str(dm) + " -q 2>&1 "
        p = os.popen(cmd)
        cand_band_smear = p.readline().strip()
        p.close()

        # Tbin calc
        # For Filter widths startting from 2^0 to 2^12=4096
        bin_width = tint * (2 ** filter)

	extime = 2*float(cand_band_smear)
        if extime < 1.0: extime = 1.0

	#So that we have at least 4 bins on pulse
	if filter <= 4 and snr > 20:
	        tbin = 4.0*int(extime / bin_width)
	else:
		tbin = 2.0*int(extime / bin_width)

        if tbin < 16:
            tbin = 16

	if tint > (extime/tbin):
	   tbin = int(extime/tint)

        #Fbin Calc
        fbin = int(round(math.pow(float(snr)/4.0,2)))
	if (fbin > nchan): fbin=nchan

	#if nchan is not power of 2, get fbin modulo of nchan
	i=0
	while nchan%(fbin+i): i+=1
	fbin+=i

	if fbin > 512:
            #fbin = 512
	    i=0
            while nchan%(512-i): i+=1
            fbin=512-i

	if fbin<16:
	    i=0
            while nchan%(16+i): i+=1
            fbin=i+16

	bins_per_plot=1024.0

        # Fraction of extraction to plot each time calc (we expect pulse to be in first half)
        if tbin>bins_per_plot:
            frac = np.linspace(0,0.5,np.ceil(tbin/bins_per_plot))
        else:
            frac = np.array([0,0.5])

        return tbin,fbin,extime,frac,cand_band_smear

def extractPlotCand(fil_file,frb_cands,noplot,fl,fh,tint,Ttot,kill_time_range,kill_chans,source_name,nchan,
                        num_pred_channels, num_pred_time, mask_file,smooth,zerodm,csv_file,save_png,frbcand_path):
	parallel=1
	if(frb_cands.size >= 1 and noplot is not True):
			if(frb_cands.size>1):
                                frb_cands = np.sort(frb_cands)
                                frb_cands[:] = frb_cands[::-1]
                        if(frb_cands.size==1): frb_cands = [frb_cands]
                        # remove pre-existing files already in prediction directory
                        cmd = "rm {0}/*.png {0}/*.ps {0}/*.pdf {0}/*.pickle".format(os.path.dirname(frbcand_path))
			print cmd
                        os.system(cmd)
			cmd_array=[]
			for indx,frb in enumerate(frb_cands):
                                time = frb['time']
                                dm = frb['dm']
                                filter = frb['filter']
                                width = tint * (2 ** filter)*(10**3) # Width in msec
                                snr = frb['snr']
				if np.size(frb.to_list()) > 6: prob = frb['FRBprob']
				else: prob = ""

				#print "here"
                                tbin,fbin,extime,frac,cand_band_smear = plotParaCalc(snr,filter,dm,fl,fh,tint,nchan)
				fbin = num_pred_channels
                                bin_width = (2 ** filter)
				#So that we have at least 4 bins on pulse
			        if filter <= 4 and filter > 0 and snr > 20:
			                downfact = int(bin_width/4.0)
				elif filter > 2:
					downfact = int(bin_width/2.0)
			        else:
			                downfact = 1
				if downfact == 0: downfact = 1

				#print fbin,filter,bin_width,downfact
				#stime = time-(extimeplot*0.1) # Go back data
                                #stime = time - float(cand_band_smear)
				#TotDisplay = (downfact*bin_width)*tint*128 # To display 256 times the pulse width in the plot
				#print TotDisplay

				# TotDisplay = (width/10**3)*256 #Roughly 256 times the pulse width window for display
                                # NOTE: experimenting with TotDisplay to get 256 bins out
                                TotDisplay = num_pred_time * tint * downfact

                                stime = time-(TotDisplay/2.0)

				if smooth: smooth_bins  = int(smooth*(bin_width)) # As downsampling is done after the smoothing, we do not need to multiplie downsample here
				else: smooth_bins = 0

                                if(stime<0): stime = 0
                                if(stime+extime>=Ttot): extime=Ttot-stime
                                if(any(l<=time<=u for (l,u) in kill_time_range) or extime < 0.0):
					print "Candidate inside bad-time range"
				else:
                                        # place candidate in directory with FRBcand file, labeling with index, timestamp, and DM
					candname = os.path.dirname(frbcand_path) + '/%04d' % (indx) + "_" + '%.3f' % (time) + "sec_DM" + '%.2f' % (dm)
					cmd = "python /home/dleduc/hey-aliens/simulateFRBclassification/waterfaller_vg_dom.py --show-ts " + \
                                               " -t " + str(TotDisplay) + \
				 	       " --colour-map=viridis " + \
					       " -T "  + str(stime) +  \
					       " -d "  + str(dm) + \
					       " --sweep-dm " + str(dm) + \
					       " -s "  + str(fbin) +  \
					       " -o "  + str(candname) + \
					       " --scaleindep " + \
					       " --downsamp " + str(downfact) + \
					       " --width-bins " + str(smooth_bins) + \
					       " --snr " + str(snr) + \
					       " --width " + str(width) + " " + \
 					       fil_file
					if mask_file: cmd = cmd + " --maskfile  " + str(mask_file)
					if zerodm: cmd = cmd + " --zerodm "
					if csv_file: cmd = cmd + " --logs " + str(csv_file)
					if prob: cmd = cmd + " --prob " + str(prob)
                                        if save_png: cmd = cmd + " --save_png "
					if parallel:
						cmd_array.append(cmd)
					else:
						print cmd
						os.system(cmd)

		  	if parallel:
				exeparallel(cmd_array)
				open('cand_plot_commands','wb').write('\n'.join(i for i in cmd_array))

			tt.sleep(2)
			print "Plotting Done"
	else:
		print "No candidate found"
                return