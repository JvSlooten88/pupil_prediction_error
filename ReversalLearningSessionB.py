from __future__ import division, print_function

#!/usr/bin/env python
# encoding: utf-8
"""
EyeLinkSession.py

Created by Tomas Knapen on 2011-04-27.
Copyright (c) 2011 __MyCompanyName__. All rights reserved.
"""

import os, sys, datetime, pickle
import math

import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib 
matplotlib.use('TkAgg') 
import matplotlib.pyplot as pl
import pandas as pd
import numpy.linalg as LA
import bottleneck as bn
import glob
import seaborn as sn
import scipy.signal as signal
import sympy
import mne 
import matplotlib.image as mpimg

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy import optimize, polyval, polyfit
from scipy.linalg import sqrtm, inv
from tables import NoSuchNodeError
import matplotlib.lines as mlines
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from sklearn import linear_model
from sklearn.preprocessing import Imputer

from joblib import Parallel, delayed
import itertools
from itertools import chain

import logging, logging.handlers, logging.config

sys.path.append( os.environ['ANALYSIS_HOME'] )
from Tools.log import *
from Tools.Operators import ArrayOperator, EDFOperator, HDFEyeOperator, EyeSignalOperator
from Tools.Operators.EyeSignalOperator import detect_saccade_from_data
from Tools.Operators.CommandLineOperator import ExecCommandLine
from Tools.other_scripts.plotting_tools import *
from Tools.other_scripts import functions_jw as myfuncs
from Tools.other_scripts import functions_jw_GLM as GLMfuncs
from Tools.other_scripts import savitzky_golay as savitzky_golay

from IPython import embed as shell

from ReversalLearningSession import ReversalLearningSession
#deze relatieve import lukt niet! 
#from ..model.TD import TD, TD_runner, DelayCalculator
import peak_detect 
from fir import FIRDeconvolution


class ReversalLearningSessionB(ReversalLearningSession):
	"""ReversalLearningSessionB is used for the analysis of behaviour (key presses) in the second Reversal Learning pupil experiment (run.py). It inherits functionality from ReversalLearningSession."""

	def __init__(self, subject, experiment_name, project_directory, version, aliases, pupil_hp, loggingLevel = logging.DEBUG):
		"""all important class and module data is imported from PupilPredictionErrorSession """
		super(ReversalLearningSessionB, self).__init__(subject = subject, experiment_name = experiment_name, project_directory = project_directory, version=version, aliases = aliases, pupil_hp = pupil_hp, loggingLevel = logging.DEBUG)
		
		
	def events_and_signals_in_time_behav(self, data_type = 'pupil_bp', requested_eye = 'L', plot_reversal_blocks=False): 
		"""events_and_signals_in_time_behav takes all aliases' data from the hdf5 file / events_and_signals_in_time. It adds behavioural analyses on top of the analyses that 
		are already done in events_and_signals_in_time. 
		"""
		trials_per_run = [0] 
		domain_time = []
		all_keypresstimes = []
		
		all_padded_keypress_times=[]
		all_padded_blink_times=[]
		all_padded_saccade_times=[]
		all_padded_colour_times=[]
		all_padded_sound_times=[]

		all_keypressamples = []
		trial_start = [] 
		trials_before_keypress = [] 
		all_trial_index_for_space_presses = []
		unclear_domain_start_c=[]
		clear_domain_start_estim_c=[]
		unclear=[]
		clear=[]
		clear_estim=[]
		
		self.events_and_signals_in_time(data_type = data_type, requested_eye= requested_eye)

		counter = 0
		session_time = 0
		padding_time = 60.0 #seconds
		
		for alias in self.aliases:
			trial_times = self.ho.read_session_data(alias, 'trials')
			trial_phase_times = self.ho.read_session_data(alias, 'trial_phases')
			session_start_EL_time = np.array(trial_phase_times[trial_phase_times['trial_phase_index'] == 1]['trial_phase_EL_timestamp'])[0] # np.array(trial_times['trial_start_EL_timestamp'])[0]#
			session_stop_EL_time = np.array(trial_times['trial_end_EL_timestamp'])[-1]			
			total_time = np.array(((session_stop_EL_time - session_start_EL_time)/1000)/60) #total time in minutes
			trial_start_times = np.array(((trial_phase_times[trial_phase_times['trial_phase_index'] == 1]['trial_phase_EL_timestamp']) - session_start_EL_time + session_time) / self.sample_rate)

			trial_parameters = self.ho.read_session_data(alias, 'parameters')
			button_events = self.ho.read_session_data(alias, 'events')
			trials_per_run.append(len(trial_parameters['trial_nr']))
						
			#Calculate block reversals 
			domain_time.append(np.array(trial_parameters['domain']))
			block_reversals = [np.array(np.sum(np.abs(np.diff(dt)))) for dt in domain_time]
			reversals = [np.array(np.diff(dt)) for dt in domain_time]

			last_trial_per_run = np.array(trials_per_run[1:])				
			self.reversal_positions = [np.array(np.where(np.any([rev !=0], axis=0 )))[-1][:] for rev in reversals]
			
			reversal_indices = [np.append(0,[(np.append(self.reversal_positions[i], last_trial_per_run[i]))]).astype(int) for i in range(len(last_trial_per_run))]
			reversal_indices_split = [np.round(np.diff(reversal_indices[i])/2) for i in range(len(last_trial_per_run))]

			#Calculate participant's reversal key presses 
			keypressamples = np.array(button_events[button_events['key'] == 32]['EL_timestamp'])
			keypresstimes = np.array((np.array(button_events[button_events['key'] == 32]['EL_timestamp']) - session_start_EL_time + session_time)/ self.sample_rate)
			keypresstimes = keypresstimes[keypresstimes > trial_start_times[0]] #spacebar press na trial start indexeren 
			keypressamples = keypressamples[keypresstimes > trial_start_times[0]]
			all_keypressamples.append(keypressamples)  #get EL timestamp to look op which trial this was 
			all_keypresstimes.append(keypresstimes) #spacebar = 32	
			assert len(keypresstimes) > 0, 'No space bar domain reports in run %s'%alias
			trial_index_for_space_presses = np.array([np.arange(trial_start_times.shape[0])[trial_start_times<kpt][-1] for kpt in keypresstimes])
			all_trial_index_for_space_presses.append(trial_index_for_space_presses)
						
			#Zero-padding for keypresses
			padded_keypresstimes = padding_time + keypresstimes + ((2*i)*padding_time)			
			#append all padded event times 
			all_padded_keypress_times.append(padded_keypresstimes)				

			session_time += session_stop_EL_time - session_start_EL_time

		self.padded_keypress_times = np.concatenate(all_padded_keypress_times)

		#remove accidental keypresses 
		corrected_trial_index_for_space_presses = list(np.copy(all_trial_index_for_space_presses))
		if self.subject.initials == 'ta':
			corrected_trial_index_for_space_presses[2]=np.delete(all_trial_index_for_space_presses[2],0)#remove first reversal press
		elif self.subject.initials == 'kv':
			corrected_trial_index_for_space_presses[0]=np.delete(all_trial_index_for_space_presses[0],-1)#remove last reversal press 
		elif self.subject.initials == 'mt':
			corrected_trial_index_for_space_presses[5]=np.delete(all_trial_index_for_space_presses[5],-1)#remove double reversal press
		elif self.subject.initials == 'lha':
			corrected_trial_index_for_space_presses[5]=np.delete(all_trial_index_for_space_presses[5],1) #remove second reversal press
			corrected_trial_index_for_space_presses[6]=np.delete(all_trial_index_for_space_presses[6],0) #remove first reversal press 
		elif self.subject.initials == 'sa':
			corrected_trial_index_for_space_presses[4]=np.delete(all_trial_index_for_space_presses[4],1) #remove second reversal press
		elif self.subject.initials == 'bcm':
			corrected_trial_index_for_space_presses[0]=np.delete(all_trial_index_for_space_presses[0],[1,2]) #remove last two presses
			corrected_trial_index_for_space_presses[7]=np.delete(all_trial_index_for_space_presses[7],0) #remove first reversal
			corrected_trial_index_for_space_presses[5]=np.delete(all_trial_index_for_space_presses[5],0) #remove first reversal 
		elif self.subject.initials == 'des': 
			corrected_trial_index_for_space_presses[0]=np.delete(all_trial_index_for_space_presses[0],0) #remove first double button press
			corrected_trial_index_for_space_presses[2]=np.delete(all_trial_index_for_space_presses[2],[0,3]) #remove first accidental and last accidental buttonpress
		elif self.subject.initials == 'iv': 
			corrected_trial_index_for_space_presses[1]=np.delete(all_trial_index_for_space_presses[1],-1) #remove accidental last press 

		
		#correct keypresstimes using corrected_trial_index_for_space_presses
		######THIS SHOULD BE DONE USING A DIFFERENT APPROACH #######
		#check if there are as much button presses as reversals				
		equal_runs = np.array([(len(self.reversal_positions[i]) == len(corrected_trial_index_for_space_presses[i])) for i in range(len(self.reversal_positions))])
		self.equal_indices = np.arange(len(equal_runs))[equal_runs]
		#check if reversal occurred before button_press 
		button_press = np.array([self.reversal_positions[i] - corrected_trial_index_for_space_presses[i] for i in range(len(self.reversal_positions)) if len(corrected_trial_index_for_space_presses[i]) == len(self.reversal_positions[i])]) 
		press_after_rev =np.array([bp < 0 for bp in button_press])
		press_correct = np.array([len(press) == sum(press) for press in press_after_rev])
		press_correct_percentage =  np.sum(press_correct)/len(press_correct)
		
		#correct run indices: as much reversals as button presses AND button presses occurred after the reversal
		self.correct_indices = self.equal_indices[press_correct]
		if self.subject.initials == 'mtt': 
			self.correct_indices = np.delete(self.correct_indices, 1) #delete the block where mtt pressed at the end of the run 3x (algorithm doesn't filter this out)	
		
		###CORRECT RUNS, BUTTON PRESSES AND DOMAIN TIMES### 
		self.correct_reversal_blocks = np.array([self.reversal_positions[ci] for ci in self.correct_indices])
		self.correct_button_presses = np.array([corrected_trial_index_for_space_presses[ci] for ci in self.correct_indices])
		self.correct_aliases = np.array([self.aliases[ci] for ci in self.correct_indices])						  	
		correct_domain_time = np.array([domain_time[ci] for ci in self.correct_indices])
		self.correct_domain_time = np.hstack(correct_domain_time)

		#accumulate trials of correct runs 
		self.last_trial_per_run_corrected = np.array(last_trial_per_run[self.correct_indices])
		trials_per_run_corrected = np.array(np.r_[0, last_trial_per_run[self.correct_indices]])		
		self.run_trial_limits_corrected = np.array([np.cumsum(trials_per_run_corrected)[:-1],np.cumsum(trials_per_run_corrected)[1:]]).T
		
		#estimate average amount of trials needed to discover reward probability. 		
		av_unclear_length = np.array(np.abs(np.round(np.mean(np.concatenate(self.correct_reversal_blocks - self.correct_button_presses))))).astype(int)
		
		self.logger.info('participant %s discovered the correct amount of reversals in %i of %i blocks and correctly pressed after the reversal in %i of these blocks' %(self.subject.initials, sum(equal_runs), len(corrected_trial_index_for_space_presses), len(self.correct_indices)))
		
		unclear_domain_start = [np.append(0, self.correct_reversal_blocks[i]).astype(int) for i in range(len(self.last_trial_per_run_corrected))]
		clear_domain_start = [np.append(av_unclear_length, self.correct_button_presses[i]) for i in range(len(self.last_trial_per_run_corrected))]
		
		#concatenate all correct trial indices for clear and unclear periods
		for i in range(len(self.correct_reversal_blocks)):
				unclear.append(unclear_domain_start[i] + counter)
				clear.append(clear_domain_start[i] + counter)
				counter = self.run_trial_limits_corrected[i][1]
		
		#true = unclear, false = clear 
		self.clarity_indices= np.array([(np.arange(self.run_trial_limits_corrected[-1][-1])<x[1]) * (np.arange(self.run_trial_limits_corrected[-1][-1])>=x[0]) for x in zip(np.hstack(unclear), np.hstack(clear))]).sum(axis =0, dtype = bool) #select trials on basis of first and second domain indices
		#Button press events 
		self.keypress_times = np.concatenate(all_keypresstimes)
		self.reversal_keypresses = np.copy(corrected_trial_index_for_space_presses)

		##save reversal keypresses, all key presses 
		with pd.get_store(self.ho.inputObject) as h5_file:
			h5_file.put("/%s/%s"%('TD', 'reversal_keypresses'), pd.Series(np.array(corrected_trial_index_for_space_presses)))
			h5_file.put("/%s/%s"%('keypresses', 'all_keypresses'), pd.Series(np.array(self.keypress_times)))
			h5_file.put("/%s/%s"%('keypresses', 'press_correct'), pd.Series(press_correct_percentage))


		np.save(os.path.join(os.path.split(self.ho.inputObject)[0], 'reversal_keypresses.npy'), corrected_trial_index_for_space_presses)
		
		#calculate reward and prediction information for correct runs 		  		
		button_press_trial = [np.zeros(run) for run in self.last_trial_per_run_corrected]
		for i in range(len(button_press_trial)): 
			button_press_trial[i][self.correct_button_presses[i]] = 1 
		dts_per_run = [self.correct_domain_time[i[0]:i[1]] for i in self.run_trial_limits_corrected] 
		colours = self.real_reward_probability * self.hue_indices
		green = colours[0]; purple = colours[1]	
		lp_colour, hp_colour = self.reward_prob_indices[0].astype(int), self.reward_prob_indices[1].astype(int)	
		low_rw_sound, high_rw_sound = self.sound_indices[0].astype(int), self.sound_indices[1].astype(int)
		green_correct_runs = np.array([green[i[0]:i[1]] for i in self.run_trial_limits])[self.correct_indices]
		purple_correct_runs = np.array([purple[i[0]:i[1]] for i in self.run_trial_limits])[self.correct_indices]
		PRPE= np.ma.masked_equal(lp_colour * high_rw_sound,0)+.1 
		NRPE= np.ma.masked_equal(hp_colour * low_rw_sound,0)+.1
		PRPE_runs = [PRPE[i[0]:i[1]] for i in self.run_trial_limits]; NRPE_runs = [NRPE[i[0]:i[1]] for i in self.run_trial_limits]
		PRPE_correct_runs = np.array(PRPE_runs)[self.correct_indices]
		NRPE_correct_runs = np.array(NRPE_runs)[self.correct_indices]
		
		#plot correct runs
		if plot_reversal_blocks == True: 					 
			f = pl.figure(figsize = (10,12)) 
			for i in range(len(self.run_trial_limits_corrected)): 
				s = f.add_subplot(len(self.run_trial_limits_corrected),1,i+1)
				pl.plot(dts_per_run[i], 'r')
				pl.plot(green_correct_runs[i], 'g')
				pl.plot(purple_correct_runs[i], 'm')
				pl.plot(button_press_trial[i], 'y--', alpha = 0.8)
				pl.plot(PRPE_correct_runs[i], 'g*')
				pl.plot(NRPE_correct_runs[i], 'k*')
				s.set_title(self.correct_aliases[i])
				s.set_ylim([-0.1,1.2])
				simpleaxis(s)
				spine_shift(s)	
			pl.tight_layout()
			pl.savefig(os.path.join(self.base_directory, 'figs' ,'correct_blocks_and_keypresses.pdf'))	
		else: 
			pass 

		 
	def deconvolve_colour_sound_button(self, analysis_sample_rate=20, interval = [-0.5,4.5],  data_type = 'pupil_bp_zscore', requested_eye = 'L', microsaccades_added=False ):
		"""deconvolution of the pupil signal of data_type, to see how colours, sounds and button press events influence pupil dilation/contraction."""

		self.logger.info('starting basic pupil deconvolution with data of type %s and sample_rate of %i Hz in the interval %s, microsaccades_added = %s' % (data_type, analysis_sample_rate, str(interval), str(microsaccades_added)))

		if not hasattr(self, 'pupil_data'): # we assume that we'll grab the same events and data whatever the present deconvolve_reward_probabilities method does
			self.events_and_signals_in_time(data_type = data_type, requested_eye= requested_eye, microsaccades_added=microsaccades_added)
			self.events_and_signals_in_time_behav(data_type= data_type, requested_eye= requested_eye, microsaccades_added=microsaccades_added)		

		input_signal = sp.signal.decimate(self.pupil_data, int(self.sample_rate / analysis_sample_rate))

		#add microsaccade events to colour, sound and button press events  
		if microsaccades_added: 
			events = [self.blink_times + interval[0], self.colour_times + interval[0], self.sound_times + interval[0], self.keypress_times + interval[0], self.microsaccade_times + interval[0]] 
		else:
			events = [self.blink_times + interval[0], self.colour_times + interval[0], self.sound_times + interval[0], self.keypress_times + interval[0]]
		
		#add eye jitter events
		dx_signal = np.array(sp.signal.decimate(self.dx_data, int(self.sample_rate / analysis_sample_rate)), dtype = np.float32)	
		nr_sample_times = np.arange(interval[0], interval[1], 1.0/analysis_sample_rate).shape[0]
		added_jitter_regressors = np.zeros((nr_sample_times, dx_signal.shape[0]))
		
		for i in range(nr_sample_times):
			added_jitter_regressors[i,(i+1):] = dx_signal[:-(i+1)]
		
		#run deconvolution without nuisance regressor 
		doNN = ArrayOperator.DeconvolutionOperator( inputObject = input_signal,
							eventObject = events, TR = 1.0/analysis_sample_rate, deconvolutionSampleDuration = 1.0/analysis_sample_rate, 
							deconvolutionInterval = interval[1] - interval[0], run = True )
		doNN.residuals()
		
		self.logger.info('explained variance (r^sq) without jitter %1.4f, microsaccades_added = %s, analysis_sample_rate = %i'%((1.0 -(np.sum(np.array(doNN.residuals)**2) / np.sum(input_signal**2))), str(microsaccades_added), analysis_sample_rate))
		rsquared_full_input_signal = 1.0 -(np.sum(np.array(doNN.residuals)**2) / np.sum(input_signal**2))
		
		time_points = np.linspace(interval[0], interval[1], np.squeeze(doNN.deconvolvedTimeCoursesPerEventType).shape[1])
		

		f = pl.figure() 
		ax = f.add_subplot(111)
		for x in range(len(events)):
			pl.plot(time_points, np.squeeze(doNN.deconvolvedTimeCoursesPerEventType)[x]) 
		ax.set_title('standard deconvolution  (no eye jitter added)')
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		ax.set_xlim(xmin=interval[0], xmax=interval[1])
		if microsaccades_added: 
			pl.legend(['blinks', 'colour', 'sound', 'keypress', 'microsaccade'], loc='best') 
		else: 
			pl.legend(['blinks', 'colour', 'sound', 'keypress'], loc='best')
		simpleaxis(ax)
		spine_shift(ax)
		pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials + '_standard_event_keypress_microsaccades_%s_%i_Hz.pdf'%(str(microsaccades_added), analysis_sample_rate)))

		
		folder_name = 'standard_deconvolve_keypress_%s'%str(microsaccades_added) 
		#store deconvolution 
		with pd.get_store(self.ho.inputObject) as h5_file:
			h5_file.put("/%s/%s"%(folder_name, 'residuals'), pd.Series(np.squeeze(np.array(doNN.residuals))))
			h5_file.put("/%s/%s"%(folder_name, 'time_points'), pd.Series(time_points))
			h5_file.put("/%s/%s"%(folder_name, 'dec_time_course'), pd.DataFrame(np.squeeze(doNN.deconvolvedTimeCoursesPerEventType).T))
			h5_file.put("/%s/%s"%(folder_name, 'rsquared'), pd.Series(rsquared_full_input_signal))

	def deconvolve_blinks_colour_button(self, analysis_sample_rate=20, interval =[-0.5,4.5],  data_type = 'pupil_bp_zscore', requested_eye = 'L', microsaccades_added=False): 
		"""deconvolve blinks and keypresses deconvolves  blinks and keypresses colour event""" 
		self.logger.info('starting basic pupil deconvolution with data of type %s and sample_rate of %i Hz in the interval %s, microsaccades_added = %s' % (data_type, analysis_sample_rate, str(interval), str(microsaccades_added)))

		if not hasattr(self, 'pupil_data'): # we assume that we'll grab the same events and data whatever the present deconvolve_reward_probabilities method does
			self.events_and_signals_in_time(data_type = data_type, requested_eye= requested_eye, microsaccades_added=microsaccades_added)
			self.events_and_signals_in_time_behav(data_type= data_type, requested_eye= requested_eye, microsaccades_added=microsaccades_added)		

		input_signal = sp.signal.decimate(self.pupil_data, int(self.sample_rate / analysis_sample_rate))

		
		events = [self.blink_times + interval[0], self.colour_times + interval[0], self.keypress_times + interval[0]]

		#run deconvolution without nuisance regressor 
		do = ArrayOperator.DeconvolutionOperator( inputObject = input_signal,
							eventObject = events, TR = 1.0/analysis_sample_rate, deconvolutionSampleDuration = 1.0/analysis_sample_rate, 
							deconvolutionInterval = interval[1] - interval[0], run = True )
		do.residuals()
		
		self.logger.info('explained variance (r^sq) without jitter %1.4f, analysis_sample_rate = %i'%((1.0 -(np.sum(np.array(do.residuals)**2) / np.sum(input_signal**2))), analysis_sample_rate))
		rsquared_full_input_signal = 1.0 -(np.sum(np.array(do.residuals)**2) / np.sum(input_signal**2))
		
		time_points = np.linspace(interval[0], interval[1], np.squeeze(do.deconvolvedTimeCoursesPerEventType).shape[1])

		f = pl.figure() 
		ax = f.add_subplot(111)
		for x in range(len(events)):
			pl.plot(time_points, np.squeeze(do.deconvolvedTimeCoursesPerEventType)[x]) 
		ax.set_title('Standard deconvolution (no sound response deconvolved)')
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		ax.set_xlim(xmin=interval[0], xmax=interval[1])
		pl.legend(['blinks', 'colour', 'keypress'], loc='best')
		simpleaxis(ax)
		spine_shift(ax)
		pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials + '_standard_event_keypress_no_sound_%i_Hz.pdf'%analysis_sample_rate))

		folder_name = 'standard_deconvolve_keypress_no_sound'
		#store deconvolution 
		with pd.get_store(self.ho.inputObject) as h5_file:
			h5_file.put("/%s/%s"%(folder_name, 'residuals'), pd.Series(np.squeeze(np.array(do.residuals))))
			h5_file.put("/%s/%s"%(folder_name, 'time_points'), pd.Series(time_points))
			h5_file.put("/%s/%s"%(folder_name, 'dec_time_course'), pd.DataFrame(np.squeeze(do.deconvolvedTimeCoursesPerEventType).T))
			h5_file.put("/%s/%s"%(folder_name, 'rsquared'), pd.Series(rsquared_full_input_signal))

	

	def deconvolve_clear_domain(self, analysis_sample_rate=20, interval = [-0.5,4.5],  data_type = 'pupil_bp_zscore', requested_eye = 'L', use_domain = 'clear', standard_deconvolution = 'no_sound' ):
		"""deconvolve_clear_domain takes the keypresses from events_and_signals_in_time_behav and uses them to evaluate the runs where reversals were correctly reported. Incorrect runs (incorrect report of reversals) 
		are left out of this analysis """
						
		self.logger.info('starting clear_domain pupil deconvolution with data of type %s and sample_rate of %i Hz in the interval %s' % (data_type, analysis_sample_rate, str(interval)))
		
		if not hasattr(self, 'pupil_data'): # we assume that we'll grab the same events and data whatever the present deconvolve_reward_probabilities method does
			self.events_and_signals_in_time_behav(data_type= data_type, requested_eye= requested_eye)		

		#only evaluate blocks with correct behaviour (indicated correct amount of reversals after actual reversal points)
		self.aliases = self.correct_aliases
		#run events and signals in time on the the correct aliases
		self.events_and_signals_in_time(data_type= data_type, requested_eye= requested_eye)

		if use_domain == 'clear': 
			domain_indices_used_now = -self.clarity_indices  #evaluate the periods of time where the participant knows the reward probabilities 
		if use_domain == 'unclear': 
			domain_indices_used_now = self.clarity_indices   #evaluate the periods of time where the participant does not know the reward probabilities 

		# if standard_deconvolution == 'sound': #residual signal sound response deconvolved 
		# 	data_folder='standard_deconvolve_keypress_%s'%str(microsaccades_added)
		# elif standard_deconvolution == 'no_sound': #residual signal sound response not deconvolved
		# 	data_folder='standard_deconvolve_keypress_no_sound'
		# elif standard_deconvolution == 'raw': #raw pupil signal, no residuals 
		# with pd.get_store(self.ho.inputObject) as h5_file:
		# 	try:
		# 		residuals_standard_deconvolve_keypress = h5_file.get("/%s/%s"%(data_folder, 'residuals'))					
		# 	except (IOError, NoSuchNodeError):
		# 		self.logger.error("no residuals present")
		
		input_signal = sp.signal.decimate(self.pupil_data, int(self.sample_rate / analysis_sample_rate))

		#select regressors 
		blink_times = [self.blink_times + interval[0]]						
		keypress_times = [self.keypress_times + interval[0]]
		saccade_times = [self.saccade_times + interval[0]] 
		colour_times = [self.colour_times + interval[0]]
		sound_times = [self.sound_times + interval[0]]
		
		cue_indices = [			
			self.reward_prob_indices[0] * domain_indices_used_now,	 #LP cue  		
			self.reward_prob_indices[1] * domain_indices_used_now,	 #HP cue  
		]
		cue_times = [self.colour_times[ci] + interval[0] for ci in cue_indices]

		reward_event_indices = [
			self.reward_prob_indices[0] * self.sound_indices[0] * domain_indices_used_now, #LP NR 
			self.reward_prob_indices[0] * self.sound_indices[1] * domain_indices_used_now, #LP HR  
			self.reward_prob_indices[1] * self.sound_indices[0] * domain_indices_used_now, #HP LR  
			self.reward_prob_indices[1] * self.sound_indices[1] * domain_indices_used_now, #HP HR  			
		]	
		reward_event_times = [self.sound_times[re_i] + interval[0] for re_i in reward_event_indices]

		events=[]							#regressor events: 
		events.extend(blink_times) 			#[0]
		events.extend(keypress_times)		#[1]
		events.extend(saccade_times)		#[2]
		events.extend(colour_times)			#[3]
		events.extend(cue_times)			#LP[4], HP[5]
		events.extend(sound_times)			#[6]		
		events.extend(reward_event_times) 	#LP_NR[7], LP_HR[8], HP_LR[9], HP_HR[10] 

		# events = [self.sound_times[ev_i] + interval[0] for ev_i in event_indices]
		
		self.logger.info('starting clear domain deconvolution with data of type %s and sample_rate of %i Hz in the interval %s. Currently the %s domain is analysed using residuals of the full pupil signal' % (data_type, analysis_sample_rate, str(interval), use_domain))

		do = ArrayOperator.DeconvolutionOperator( inputObject = input_signal,
							eventObject = events, TR = 1.0/analysis_sample_rate, deconvolutionSampleDuration = 1.0/analysis_sample_rate, deconvolutionInterval = interval[1] - interval[0], run = True )
		time_points = np.linspace(interval[0], interval[1], np.squeeze(do.deconvolvedTimeCoursesPerEventType).shape[1])
		do.residuals()

		self.logger.info('explained variance (r^sq) without jitter %1.4f, analysis_sample_rate = %i'%((1.0 -(np.sum(np.array(do.residuals)**2) / np.sum(input_signal**2))), analysis_sample_rate))
		rsquared_on_residuals = 1.0 -(np.sum(np.array(do.residuals)**2) / np.sum(input_signal**2))

		#all deconvolved regressor timecourses 
		timecourses = np.squeeze(do.deconvolvedTimeCoursesPerEventType) 
		nuisance_timecourses = timecourses[:3,:]
		standard_response_timecourses = np.r_[[timecourses[3], timecourses[6]]]
		cue_high_low_timecourses = np.r_[[timecourses[4], timecourses[5]]]
		reward_timecourses = np.r_[[timecourses[7], timecourses[8], timecourses[9], timecourses[10]]]
		rpe_timecourses = np.r_[[timecourses[8] - timecourses[10], timecourses[9] - timecourses[7]]] #PRPE: LPHR - HPHR, NRPE: HPNR - LPNR 
		pe_timecourses = np.r_[[timecourses[10] - timecourses[7], timecourses[8] - timecourses[9]]]  #Predicted reward - predicted loss, Unpredicted reward - unpredicted los 
				
		folder_name = 'deconvolve_%s_domain_'%use_domain 
		#store deconvolution 
		with pd.get_store(self.ho.inputObject) as h5_file:
			h5_file.put("/%s/%s"%(folder_name, 'residuals'), pd.Series(np.squeeze(np.array(do.residuals))))
			h5_file.put("/%s/%s"%(folder_name, 'time_points'), pd.Series(time_points))
			h5_file.put("/%s/%s"%(folder_name, 'dec_time_course'), pd.DataFrame(np.squeeze(do.deconvolvedTimeCoursesPerEventType).T))
			h5_file.put("/%s/%s"%(folder_name, 'rsquared'), pd.Series(rsquared_on_residuals))
	

	def deconvolve_clear_domain_covariates(self, analysis_sample_rate=20, interval = [-0.5,4.5], fix_dur=0.5,  data_type = 'pupil_bp_zscore', requested_eye = 'L', use_domain = 'clear'):
		"""deconvolve_clear_domain_covariates performs a deconvolution analysis using FIRDeconvolution on the pupil signal from correctly performed runs. The deconvolution operator 
		takes in events, variable event durations and covariates to explain the pupil signal more extensively """	

		self.logger.info('starting clear_domain_covariates pupil FIRDeconvolution with data of type %s and sample_rate of %i Hz in the interval %s' % (data_type, analysis_sample_rate, str(interval)))
		
		if not hasattr(self, 'pupil_data'): 
			self.events_and_signals_in_time_behav(data_type= data_type, requested_eye= requested_eye)	

		self.aliases = self.correct_aliases	 #only select correctly performed runs
		self.events_and_signals_in_time(data_type= data_type, requested_eye= requested_eye)	#run events and signals in time on correctly performed runs 
		

		if use_domain == 'unclear': 
			domain_indices_used_now = self.clarity_indices   #evaluate the periods of time where the participant does not know the reward probabilities 
		if use_domain == 'clear': 
			domain_indices_used_now = -self.clarity_indices  #evaluate the periods of time where the participant knows the reward probabilities 

		input_signal = sp.signal.decimate(self.pupil_data, int(self.sample_rate / analysis_sample_rate))

		#select regressors 
		blink_times = [self.blink_times + interval[0]]						
		keypress_times = [self.keypress_times + interval[0]]
		saccade_times = [self.saccade_times + interval[0]] 
		colour_times = [self.colour_times + interval[0]]
		sound_times = [self.sound_times + interval[0]]
		
		cue_indices = [			
			self.reward_prob_indices[0] * domain_indices_used_now,	 #LP cue  		
			self.reward_prob_indices[1] * domain_indices_used_now,	 #HP cue  
		]
		cue_times = [self.colour_times[ci] + interval[0] for ci in cue_indices]

		reward_event_indices = [
			self.reward_prob_indices[0] * self.sound_indices[0] * domain_indices_used_now, #LP NR 
			self.reward_prob_indices[0] * self.sound_indices[1] * domain_indices_used_now, #LP HR  
			self.reward_prob_indices[1] * self.sound_indices[0] * domain_indices_used_now, #HP LR  
			self.reward_prob_indices[1] * self.sound_indices[1] * domain_indices_used_now, #HP HR  			
		]	
		reward_event_times = [self.sound_times[re_i] + interval[0] for re_i in reward_event_indices]
		
		events=[]							#regressor events: 
		events.extend(blink_times) 			#[0]
		events.extend(keypress_times)		#[1]
		events.extend(saccade_times)		#[2]
		events.extend(cue_times)			#LP[3], HP[4]
		events.extend(reward_event_times) 	#LP_NR[5], LP_HR[6], HP_LR[7], HP_HR[8] 
		events.extend(colour_times)			#[9]
		events.extend(sound_times)			#[10]

		#pupil baseline on fixation interval  
		ds_pupil_baseline_data = sp.signal.decimate(self.pupil_baseline_data, int(self.sample_rate / analysis_sample_rate)) #downsampled pupil_baseline_zscore signal	
		fix_start_idx = np.around(self.fix_times*analysis_sample_rate).astype(int) 
		fix_period = int(fix_dur * analysis_sample_rate)
		pupil_baseline_fix = np.array([ds_pupil_baseline_data[fix:fix+fix_period].mean(axis=0) for fix in fix_start_idx])
		
		#covariates 
		covariates = {
			'cue_low.gain': np.ones(len(events[3])),
			'cue_low.pupil_baseline': pupil_baseline_fix,
			'cue_high.gain': np.ones(len(events[4])), 
			'cue_high.pupil_baseline': pupil_baseline_fix, 
			'LP_NR.gain': np.ones(len(events[5])),
			'LP_NR.pupil_baseline': pupil_baseline_fix,
			'LP_HR.gain': np.ones(len(events[6])), 
			'LP_HR.pupil_baseline': pupil_baseline_fix, 
			'HP_NR.gain': np.ones(len(events[7])),
			'HP_NR.pupil_baseline': pupil_baseline_fix, 
			'HP_HR.gain': np.ones(len(events[8])),
			'HP_HR.pupil_baseline': pupil_baseline_fix, 
			'colour.gain': np.ones(len(events[9])), 
			'colour.pupil_baseline': pupil_baseline_fix,
			'sound.gain': np.ones(len(events[10])),
			'sound.pupil_baseline': -pupil_baseline_fix,
		} 
				
		fd = FIRDeconvolution.FIRDeconvolution(
					signal = input_signal, 
					events = events,
					event_names = ['blink', 'keypress', 'saccade','cue_low','cue_high','LP_NR','LP_HR','HP_NR','HP_HR', 'colour', 'sound'], #,'colour', 'cue_low', 'cue_high', 'sound',  'blink', 'keypress', 'saccade', 'colour', 'cue_low', 'cue_high', 'sound', 'LP_NR', 'LP_HR', 'HP_NR', 'HP_HR'
					sample_frequency = analysis_sample_rate, 
					deconvolution_interval = [-0.5, 4.5], 
					deconvolution_frequency = analysis_sample_rate,
					covariates = covariates
					 ) 

		fd.create_design_matrix()
		fd.ridge_regress(cv=20, alphas=None) #do ridge regression (no intercept is fitted) 
		
		fd.calculate_rsq() 

		f = pl.figure(figsize=(8,4)) 
		s = f.add_subplot(211)
		s.set_title('data and predictions (colour and sound deconvolved, covariates added)')
		pl.plot(np.linspace(0,plot_time, int(plot_time * fd.deconvolution_frequency/fd.sample_frequency)), 
			fd.resampled_signal[:,:int(plot_time * fd.deconvolution_frequency/fd.sample_frequency)].T, 'r')
		pl.plot(np.linspace(0,plot_time, int(plot_time * fd.deconvolution_frequency/fd.sample_frequency)), 
			fd.predict_from_design_matrix(fd.design_matrix[:,:int(plot_time * fd.deconvolution_frequency/fd.sample_frequency)]).T, 'k')
		pl.legend(['signal','explained'])
		sn.despine(offset=10)
		pl.tight_layout()
		pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials + '_%s_deconvolution_ridge_covariates.pdf'%use_domain))

		 
		folder_name = 'deconvolve_%s_domain_FIR_ridge'%use_domain 
		#store deconvolution 
		with pd.get_store(self.ho.inputObject) as h5_file:
			h5_file.put("/%s/%s"%(folder_name, 'residuals'), pd.Series(np.squeeze(np.array(fd.residuals))))
			h5_file.put("/%s/%s"%(folder_name, 'time_points'), pd.Series(fd.deconvolution_interval_timepoints))
			h5_file.put("/%s/%s"%(folder_name, 'dec_time_course'), pd.DataFrame(np.squeeze(fd.betas_per_event_type).T))
			h5_file.put("/%s/%s"%(folder_name, 'rsquared'), pd.Series(fd.rsq))

	def pupil_around_keypress(self, analysis_sample_rate=20, period_of_interest= 60, data_type='pupil_baseline', requested_eye='L'):
		"""def pupil around keypress() extracts the pupil signal around a key press (60 seconds before, 60 seconds after) and deconvolves low frequency 
		pupil signals of data_type 'pupil_baseline_zscore'. input_signal can be 1): the original, downsampled pupil signal or 2): padded_input_signal: the 
		detrended and zero-padded pupil signal. When using padded_input_signal, one should use padded events as well. """
			
		if not hasattr(self, 'pupil_data'): 
			self.events_and_signals_in_time(data_type = data_type, requested_eye= requested_eye)
			self.events_and_signals_in_time_behav(data_type= data_type, requested_eye= requested_eye)
		
		input_signal = sp.signal.decimate(self.pupil_baseline_data_z, int(self.sample_rate / analysis_sample_rate))
		# padded_input_signal = sp.signal.decimate(self.padded_pupil_data, int(self.sample_rate / analysis_sample_rate))

		# the code below searches specifically for detrended residuals that were created at 20 Hz!!!!!
		#folder_name = 'detrended_pupil_baseline_z'
		folder_name = 'detrended_pupil_baseline'		
		with pd.get_store(self.ho.inputObject) as h5_file:
			padded_input_signal = h5_file.get("/%s/%s_%i"%(folder_name, 'detrended_residuals', analysis_sample_rate))	

		#get start indices of spacebar keypresses  
		period_of_interest = int(period_of_interest * analysis_sample_rate)
		keypress_start_idx = (self.keypress_times * analysis_sample_rate).astype(int)    
		pupil_around_keypress = np.array([input_signal[key-period_of_interest:key+period_of_interest] for key in keypress_start_idx])
		
		#get ascending trial indices of true reversal positions
		chrono_reversal_position=[]
		counter = 0
		for i in range(len(self.reversal_positions)):
			chrono_reversal_position.append(self.reversal_positions[i] + counter)		
			counter = self.run_trial_limits[i][1]
		chrono_reversal_position = np.concatenate(chrono_reversal_position)
		
		#get start times of padded_sound_times (to be used for reversal_times)
		reversal_start_idx = np.array([self.sound_times[rev] * analysis_sample_rate for rev in chrono_reversal_position]).astype(int)
		self.padded_reversal_times = np.array([self.padded_sound_times[rev] for rev in chrono_reversal_position]).astype(int)	

		##### FIRDeconvolution ##### 
		keypress_times = [self.keypress_times] 
		reversal_times = [reversal_start_idx/analysis_sample_rate]

		padded_keypress_times = [self.padded_keypress_times]
		padded_reversal_times = [self.padded_reversal_times]
		
		events=[]
		events.extend(padded_keypress_times)
		events.extend(padded_reversal_times)

		covariates = {
			'padded_keypress_times.gain': np.ones(len(events[0])), 
			'padded_reversal_times.gain': np.ones(len(events[1])),
		} 
		
		fd = FIRDeconvolution(
			signal = padded_input_signal, 
			events = events,
			event_names = ['padded_keypress_times', 'padded_reversal_times'], 
			sample_frequency = analysis_sample_rate, 
			deconvolution_interval = [-60, 60], 
			deconvolution_frequency = analysis_sample_rate,
			covariates = covariates,
			) 

		fd.create_design_matrix()	
		
		# plot_time = 16000
		# f = pl.figure()
		# sn.set(font_scale=1)
		# sn.set_style("ticks")
		# s = f.add_subplot(111)		
		# s.set_title('design matrix (%i Hz)'%analysis_sample_rate, fontsize=12)
		# pl.imshow(fd.design_matrix[:,4000:plot_time], aspect = 0.075 * plot_time/fd.deconvolution_interval_size,  interpolation = 'nearest', rasterized = True)#cmap = 'RdBu',
		# pl.xlabel('time (samples)', fontsize=11)
		# pl.ylabel('regressors')
		# sn.despine(offset=10)
		# pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials + 'designmatrix_example.pdf'))
 			
		fd.regress() 
		fd.calculate_rsq()
		# fd.bootstrap_on_residuals()	
		# key_bootstrap = fd.bootstrap_betas_per_event_type[0,:,:]
		# peak_baseline_bootstrap = [key_bootstrap[:,i].argmax() for i in range(key_bootstrap.shape[1])] 

		betas= np.zeros((len(fd.events), fd.deconvolution_interval_size ))
		for i,b in enumerate(fd.covariates.keys()): 
			beta = np.squeeze(fd.betas_for_cov(covariate=b))
			betas[i,0:fd.deconvolution_interval_size] = beta

		folder_name = 'detrended_padded_pupil_around_keypress' 		
		with pd.get_store(self.ho.inputObject) as h5_file:
			h5_file.put("/%s/%s"%(folder_name, 'deconvolved_pupil_timecourses'), pd.DataFrame(np.squeeze(betas).T, columns=fd.covariates.keys()))
			h5_file.put("/%s/%s"%(folder_name, 'covariate_keys'), pd.DataFrame(fd.covariates.keys()))
			h5_file.put("/%s/%s"%(folder_name, 'rsquared'), pd.Series(fd.rsq))
			#h5_file.put("/%s/%s"%(folder_name, 'peak_bootstrap'), pd.Series(peak_baseline_bootstrap))

	def calculate_slope_and_acceleration_pupil_around_keypress(self):
		"""Takes deconvolved pupil response around keypress and calculates the slope and maximum acceleration """

		#get deconvolved keypress timecourses  
		folder_name = 'detrended_padded_pupil_around_keypress' 		
		with pd.get_store(self.ho.inputObject) as h5_file:
			deconvolved_pupil_timecourses = h5_file.get("/%s/%s"%(folder_name, 'deconvolved_pupil_timecourses')).as_matrix()
			covariate_keys = h5_file.get("/%s/%s"%(folder_name, 'covariate_keys'))
	
		#smooth the deconvolved baseline signal a little 
		deconvolved_keypress = myfuncs.smooth(deconvolved_pupil_timecourses[:,0], window_len=100)
		[max_peaks, min_peaks] = peak_detect.peakdetect(deconvolved_keypress, lookahead=250, delta=0)
		max_peak_val = np.mean(max_peaks, 0)[1]
		max_peak_idx = max_peaks[0][0]

		#calculate 2nd order derivative from deconvolved pupil response until the peak (so, upwards slope)
		upwards_deconvolved_keypress = deconvolved_keypress[0:max_peak_idx]
		first_derivative = np.diff(upwards_deconvolved_keypress)
		second_derivative = np.diff(first_derivative)

		max_slope, av_slope = max(first_derivative), np.mean(first_derivative)
		max_accel, av_accel = max(second_derivative), np.mean(second_derivative)

		folder_name = 'slope_acceleration_pupil_around_keypress' 		
		with pd.get_store(self.ho.inputObject) as h5_file:
			h5_file.put("/%s/%s"%(folder_name, 'upwards_deconvolved_keypress'), pd.Series(upwards_deconvolved_keypress))
			h5_file.put("/%s/%s"%(folder_name, 'max_slope'), pd.Series(max_slope))
			h5_file.put("/%s/%s"%(folder_name, 'av_slope'), pd.Series(av_slope))
			h5_file.put("/%s/%s"%(folder_name, 'max_accel'), pd.Series(max_accel))
			h5_file.put("/%s/%s"%(folder_name, 'av_accel'), pd.Series(av_accel))
			h5_file.put("/%s/%s"%(folder_name, 'samples_to_peak'), pd.Series(max_peak_idx))

	

	def pupil_around_keypress_filterbank_signals(self, 
												 analysis_sample_rate=5, 
												 data_type = 'pupil_bp_zscore', 
												 requested_eye = 'L',
												 zscore=False): 
		"""Performs ridge regression on the slow pupil signal around keypress using different frequency_bands as input signal. """

		self.filter_bank_pupil_signals(data_type=data_type, requested_eye=requested_eye, do_plot=False, zscore=zscore)
		self.events_and_signals_in_time_behav(data_type= data_type, requested_eye= requested_eye)
		filt_keys_no_dots = [self.filter_bank.keys()[i].rpartition('.')[-1] for i in range(len(self.filter_bank.keys()))] 

		subsample_ratio = int(self.sample_rate/analysis_sample_rate)	
		
		#get ascending trial indices of true reversal positions
		chrono_reversal_position=[]
		counter = 0
		for i in range(len(self.reversal_positions)):
			chrono_reversal_position.append(self.reversal_positions[i] + counter)		
			counter = self.run_trial_limits[i][1]
		chrono_reversal_position = np.concatenate(chrono_reversal_position)
		
		#get start times of padded_sound_times (to be used for reversal_times)
		keypress_start_idx = (self.keypress_times * analysis_sample_rate).astype(int)    
		reversal_start_idx = np.array([self.sound_times[rev] * analysis_sample_rate for rev in chrono_reversal_position]).astype(int)
		self.padded_reversal_times = np.array([self.padded_sound_times[rev] for rev in chrono_reversal_position]).astype(int)	

		##### FIRDeconvolution ##### 
		keypress_times = [self.keypress_times] 
		reversal_times = [reversal_start_idx/analysis_sample_rate]
		padded_keypress_times = [self.padded_keypress_times]
		padded_reversal_times = [self.padded_reversal_times]
		
		events=[]
		events.extend(keypress_times)
		events.extend(reversal_times)

		covariates = {
			'keypress_times.gain': np.ones(len(events[0])), 
			'reversal_times.gain': np.ones(len(events[1])),
		} 
		
		#make FIRDeconvolution object for each self.filter_bank signal 
		for key, signal in self.filter_bank.items(): 

			fd = FIRDeconvolution(
				signal = self.filter_bank[key][::subsample_ratio], 
				events = events,
				event_names = ['keypress_times', 'reversal_times'], 
				sample_frequency = analysis_sample_rate, 
				deconvolution_interval = [-60,60], 
				deconvolution_frequency = analysis_sample_rate,
				covariates = covariates,
				) 
			fd.create_design_matrix()
			self.logger.info('Starting pupil_around_keypress deconvolution with signal of %s Hz for participant %s' %(key, self.subject.initials))
			fd.regress()
			fd.calculate_rsq()
			self.logger.info('Rsquared pupil_around_keypress deconvolution with signal of %s Hz for participant %s is: %.3f' %(key, self.subject.initials, fd.rsq))	

			#pre-allocate betas matrix  
			betas = pd.DataFrame(np.zeros((fd.deconvolution_interval_size, len(fd.covariates))), columns=fd.covariates.keys())
			for i,b in enumerate(fd.covariates.keys()): 
				betas[b] = np.squeeze(fd.betas_for_cov(covariate=b))

			#remove dot from key to be able to save it as a variable in hdf5 
			key = key.rpartition('.')[-1]
			#save all important variables in hdf5
			#folder_name = 'deconvolve_pupil_around_keypress_%i_tonic_baseline'%len(self.filter_bank)
			folder_name = 'deconvolve_pupil_around_keypress_%i_tonic_baseline_zscore_%s'%(len(self.filter_bank), str(zscore))
			with pd.get_store(self.ho.inputObject) as h5_file:
				h5_file.put("/%s/%s"%(folder_name, 'time_points'), pd.Series(fd.deconvolution_interval_timepoints))
				h5_file.put("/%s/%s"%(folder_name, 'deconvolved_pupil_timecourses_%s_hz'%key), betas)
				h5_file.put("/%s/%s"%(folder_name, 'rsquared_%s_hz'%key), pd.Series(fd.rsq))
				h5_file.put("/%s/%s"%(folder_name, 'keys'), pd.Series(filt_keys_no_dots))


	
	

	def exponential_func(self, data, gain, timescale, offset):
		""" exponential function used to fit average pupil diameter drift within a run in detrend_pupil_signals """

		return gain * np.exp(-timescale * data) + offset


	def detrend_pupil_signals(self, data_type='pupil_baseline', requested_eye='L', sample_rate=1000, analysis_sample_rate=20, starting_values = (0,0,0), padding_duration = 60): 
		"""detrend_pupil_signals takes pupil signals of data_type and calculates the average pupil signal per sample within a run 
		and estimates using exponential_func() the average drift. The average drift in pupil signal within a run is subtracted from 
		each run and z-scored, resulting in detrended_residuals_all_runs_zscore. These detrended signals are zero-padded and saved as one continuous timeseries: detrended_residuals_all_runs_raveled """

		subsample_ratio = int(sample_rate/analysis_sample_rate)
		pupil_data=[]
		for alias in self.aliases:

			trial_times = self.ho.read_session_data(alias, 'trials')
			trial_phase_times = self.ho.read_session_data(alias, 'trial_phases')

			session_start_EL_time = np.array(trial_phase_times[trial_phase_times['trial_phase_index'] == 1]['trial_phase_EL_timestamp'])[0] # np.array(trial_times['trial_start_EL_timestamp'])[0]#
			session_stop_EL_time = np.array(trial_times['trial_end_EL_timestamp'])[-1]			

			pupil_baseline = np.squeeze(self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], alias = alias, signal = data_type, requested_eye = requested_eye)).as_matrix()
			pupil_baseline_z = ((pupil_baseline - np.mean(pupil_baseline)) / pupil_baseline.std())

			#downsample signal 
			ds_pupil_baseline_z = pupil_baseline_z[::subsample_ratio]
			pupil_data.append(ds_pupil_baseline_z)

		detrended_sample_rate = sample_rate / subsample_ratio
		
		#get maximum length of a pupil run 
		max_run_length = max(np.array([len(run) for run in pupil_data]))
		#make NaN matrix of nr_runs x max pupil run  
		pupil_data_matrix = np.zeros((len(pupil_data), max_run_length))*np.nan
		#fill in the pupil data for each run
		for i in range(len(pupil_data)): 
			pupil_data_matrix[i,0:len(pupil_data[i])] = pupil_data[i]
		#calculate the average 
		av_pupil_within_run = np.mean(pupil_data_matrix, axis=0)
		av_pupil_no_nans = av_pupil_within_run[~np.isnan(av_pupil_within_run)] 

		#perform fit on minimal xdata
		xdata_for_fit = np.linspace(0, len(av_pupil_no_nans)/detrended_sample_rate, len(av_pupil_no_nans))	
		popt, pcov = sp.optimize.curve_fit(self.exponential_func, xdata_for_fit, av_pupil_no_nans, p0=starting_values, maxfev = 10000)
		# perform subtraction on inclusive xdata
		xdata_for_subtraction = np.linspace(0, len(pupil_data_matrix[0])/detrended_sample_rate, len(pupil_data_matrix[0]))	
		# standard trend to subtract
		y_fit_this_subject = self.exponential_func(xdata_for_subtraction, *popt)
		# detrend all runs
		detrended_residuals_all_runs = pupil_data_matrix - y_fit_this_subject
		#z-score detrended residuals to be able to compare subjects in a similar space
		detrended_residuals_all_runs_zscore = ((detrended_residuals_all_runs.T - np.nanmean(detrended_residuals_all_runs, axis=1)) / np.nanstd(detrended_residuals_all_runs, axis=1)).T		
		#subselect non-nan numbers and immediate ravel of NON PADDED detrended pupil signal 
		detrended_residuals_zscore_all_runs_raveled_not_padded = detrended_residuals_all_runs_zscore[~np.isnan(detrended_residuals_all_runs_zscore)]

		#zero-pad the pupil signals 
		padded_size = (pupil_data_matrix.shape[0],pupil_data_matrix.shape[1]+(detrended_sample_rate * padding_duration * 2))
		padded_pupil = np.zeros( padded_size )
		padded_pupil[:,detrended_sample_rate * padding_duration:-detrended_sample_rate * padding_duration] = detrended_residuals_all_runs_zscore

		# subselect non-nan numbers and immediate ravel of PADDED pupil signal 
		detrended_residuals_all_runs_raveled_padded = padded_pupil[~np.isnan(padded_pupil)]
		
		folder_name = 'detrended_%s'%data_type	
		with pd.get_store(self.ho.inputObject) as h5_file:
			h5_file.put("/%s/%s"%(folder_name, 'detrended_residuals_%i'%detrended_sample_rate), pd.Series(detrended_residuals_all_runs_raveled_padded))
			h5_file.put("/%s/%s"%(folder_name, 'detrended_residuals_%i_not_padded'%detrended_sample_rate), pd.Series(detrended_residuals_zscore_all_runs_raveled_not_padded))			
			h5_file.put("/%s/%s"%(folder_name, 'y_fit'), pd.Series(y_fit_this_subject))

		#plot detrended and zero-padded signals
		fig = pl.figure()
		s = fig.add_subplot(121)
		pl.plot(detrended_residuals_all_runs_raveled_padded, 'g', pupil_data_matrix[~np.isnan(pupil_data_matrix)], 'r', alpha=0.5)
		s.set_title('Detrended and padded pupil signal')
		pl.ylabel('Z')
		pl.legend(['detrended residual signal', 'original pupil_bp_zscore signal'])
		sn.despine(offset=10)
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')	
		pl.tight_layout()		
		#plot average pupil diameter within a run before and after detrending operation
		s = fig.add_subplot(122)
		sn.tsplot(pupil_data_matrix, err_style="ci_band", color='indianred')
		sn.tsplot(detrended_residuals_all_runs, err_style="ci_band", color='darkgreen')
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')	
		sn.despine(offset=10)
		#pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials + 'detrended_pupil_bp_zscore_signal.pdf'))

	def filter_detrended_pupil_signals(self, data_type='pupil_baseline', hp = 1.0/160.0, lp = 1.0/70.0, sample_rate=1000, subsample_ratio=50, padding_duration = 60): 
		"""filter_detrended_pupil_signals filters the detrended and zero-padded pupil signal from detrend_pupil_signals() using the 
		EyeSignalOperator filter function.  """
		filtered_detrended_pupil_baseline=[]
		self.events_and_signals_in_time()
		detrended_sample_rate = sample_rate / subsample_ratio
		
		folder_name = 'detrended_%s'%data_type	
		with pd.get_store(self.ho.inputObject) as h5_file:
			detrended_pupil_baseline_z = h5_file.get("/%s/%s"%(folder_name, 'detrended_residuals_%i'%detrended_sample_rate))

		#calculate samples per run 
		padding_samples = int(padding_duration * detrended_sample_rate)
		ds_samples_per_run = np.cumsum(self.samples_per_run / subsample_ratio).astype(int)		
		sample_start_run = [padding_samples + ds_samples_per_run[i] + (2*i)*padding_samples for i in range(len(ds_samples_per_run))]
		sample_to_cut_signal = np.array([s_s_r - padding_samples for s_s_r in sample_start_run]).astype(int)[1:-1]
		#split the raveled detrended_pupil_signal into number of runs 
		detrended_pupil_baseline_z_run = np.split(detrended_pupil_baseline_z, sample_to_cut_signal)

		#filter the detrended and padded pupil signal per run 
		f = pl.figure() 
		for i in range(len(detrended_pupil_baseline_z_run)): 
			filtered_detrended_pupil_op = EyeSignalOperator.EyeSignalOperator(inputObject={'timepoints':np.linspace(0,detrended_pupil_baseline_z_run[i].shape[0]/detrended_sample_rate, detrended_pupil_baseline_z_run[i].shape[0]), 'gaze_X':np.zeros((2,detrended_pupil_baseline_z_run[i].shape[0])),'gaze_Y':np.zeros((2,detrended_pupil_baseline_z_run[i].shape[0])), 'pupil': detrended_pupil_baseline_z_run[i]})
			filtered_detrended_pupil_op.interpolated_pupil = detrended_pupil_baseline_z_run[i]			
			filtered_detrended_pupil_op.filter_pupil(hp = hp, lp = lp) 
			filtered_detrended_pupil_op.zscore_pupil()
			#remove zero-padding again and append filtered pupil signal 
			filtered_detrended_pupil_baseline.append(filtered_detrended_pupil_op.baseline_filt_pupil[padding_samples:-padding_samples])	
			s = f.add_subplot(4,3,i+1)			
			pl.plot(detrended_pupil_baseline_z_run[i][padding_samples:-padding_samples], 'b', filtered_detrended_pupil_baseline[i], 'g', alpha=0.5)
			pl.legend(['detrend', 'filter_detrend'])
			s.set_title('run %s'%str(i+1))			
			pl.tight_layout()
			sn.despine(offset=5)
		pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials + 'filtered_detrended_pupil_bp_zscore_signal.pdf'))	
		#concatenate all filtered_detrended runs
		self.filtered_detrended_pupil_baseline = np.hstack(filtered_detrended_pupil_baseline)			
		
		folder_name = 'filtered_detrended_%s'%data_type	
		with pd.get_store(self.ho.inputObject) as h5_file:
			h5_file.put("/%s/%s"%(folder_name, 'filtered_detrended_residuals_%i'%detrended_sample_rate), pd.Series(self.filtered_detrended_pupil_baseline))


	def correlate_tonic_phasic_pupil_around_keypress(self, data_type='pupil_baseline_zscore', analysis_sample_rate=20, subsample_ratio=50, t_start=1.0): 
		self.events_and_signals_in_time_behav(data_type='pupil_bp_zscore')

		#get all relevant tonic and phasic pupil signals 
		with pd.get_store(self.ho.inputObject) as h5_file:
			detrended_tonic_pupil_data = h5_file.get("/%s/%s"%('detrended_%s'%data_type, 'detrended_residuals_%i'%analysis_sample_rate))
			filtered_detrended_tonic_pupil_data = h5_file.get("/%s/%s"%('filtered_detrended_%s'%data_type, 'filtered_detrended_residuals_%i'%analysis_sample_rate))
		tonic_pupil_data = self.pupil_baseline_data_z[::subsample_ratio]
		phasic_pupil_data = self.pupil_data[::subsample_ratio]
		padded_phasic_pupil_data = self.padded_pupil_data[::subsample_ratio]

		#select keypress_idx [0.5 - 1.0 s.] after keypress to inspect correlation tonic and phasic pupil 
		start_sample = int(t_start * analysis_sample_rate)
		keypress_start_idx = (self.keypress_times * analysis_sample_rate).astype(int) 
		padded_keypress_start_idx = (self.padded_keypress_times * analysis_sample_rate).astype(int) 
		
		#select pupil response interval
		phasic_pupil_change, phasic_pupil_1s_after_keypress = [], []
		for input_signal in [phasic_pupil_data]: 
			av_pupil_before_keypress = np.mean(np.array([input_signal[key-start_sample:key] for key in keypress_start_idx]), axis=1)
			av_pupil_after_keypress = np.mean(np.array([input_signal[key:key+start_sample] for key in keypress_start_idx]), axis=1)
			av_pupil_change = av_pupil_after_keypress - av_pupil_before_keypress
			pupil_1s_after_keypress = np.array([input_signal[key+start_sample] for key in keypress_start_idx]) #:key+(2*start_sample)
			phasic_pupil_1s_after_keypress.append(pupil_1s_after_keypress)
			phasic_pupil_change.append(av_pupil_change)
		#select padded phasic pupil response inteval 
		padded_phasic_pupil_change, padded_phasic_pupil_1s_after_keypress =[],[]
		for padded_input_signal in [padded_phasic_pupil_data]:  
			av_pupil_before_keypress = np.mean(np.array([padded_input_signal[key-start_sample:key] for key in padded_keypress_start_idx]),axis=1)
			av_pupil_after_keypress = np.mean(np.array([padded_input_signal[key:key+start_sample] for key in padded_keypress_start_idx]), axis=1)
			av_pupil_change = av_pupil_after_keypress - av_pupil_before_keypress 
			pupil_1s_after_keypress = np.array([padded_input_signal[key+start_sample] for key in padded_keypress_start_idx]) #:key+(2*start_sample)
			padded_phasic_pupil_1s_after_keypress.append(pupil_1s_after_keypress)
			padded_phasic_pupil_change.append(av_pupil_change)
		#select average tonic pupil around keypress
		tonic_pupil_around_keypress =[]
		for input_signal in [filtered_detrended_tonic_pupil_data, tonic_pupil_data]: 
			av_tonic_pupil = np.mean(np.array([input_signal[key-start_sample:key+start_sample] for key in keypress_start_idx]), axis=1)
			tonic_pupil_after_key = np.mean(np.array([input_signal[key: key+start_sample] for key in keypress_start_idx]), axis=1)
			tonic_pupil_around_keypress.append(av_tonic_pupil)
		#select average padded tonic pupil around keypress 
		padded_tonic_pupil_around_keypress =[]
		for padded_input_signal in [detrended_tonic_pupil_data]: 
			av_padded_tonic_pupil = np.mean(np.array([padded_input_signal[key-start_sample:key+start_sample] for key in padded_keypress_start_idx]), axis=1)
			padded_tonic_pupil_around_keypress.append(av_padded_tonic_pupil)
		
		folder_name = 'tonic_phasic_pupil_keypress' 		
		with pd.get_store(self.ho.inputObject) as h5_file:
			h5_file.put("/%s/%s"%(folder_name, 'tonic_pupil_filter_detrend'), np.squeeze(pd.DataFrame(tonic_pupil_around_keypress[0])))
			h5_file.put("/%s/%s"%(folder_name, 'tonic_pupil'), np.squeeze(pd.DataFrame(tonic_pupil_around_keypress[1])))
			h5_file.put("/%s/%s"%(folder_name, 'phasic_pupil_change'), np.squeeze(pd.DataFrame(phasic_pupil_change[0])))
			h5_file.put("/%s/%s"%(folder_name, 'tonic_pupil_detrended_padded'), np.squeeze(pd.DataFrame(padded_tonic_pupil_around_keypress[0])))
			h5_file.put("/%s/%s"%(folder_name, 'phasic_pupil_change_padded'), np.squeeze(pd.DataFrame(padded_phasic_pupil_change[0])))
			h5_file.put("/%s/%s"%(folder_name, 'phasic_pupil_1s_after_keypress'), np.squeeze(pd.DataFrame(phasic_pupil_1s_after_keypress[0])))
			h5_file.put("/%s/%s"%(folder_name, 'padded_phasic_pupil_1s_after_keypress'), np.squeeze(pd.DataFrame(padded_phasic_pupil_1s_after_keypress[0])))


	def calculate_distance(self, inputObject=None, compareObject=None): 
		"""calculate_distance takes inputObject and compares each seperate element of inputObject with all elements of compareObject. after
		all comparisons are done, it selects and returns for each comparison the value of compareObject that is closest to zero (thus, closest to 
			inputObject). """

		#calculate all possible distances between inputObject and compareObject 
		diffs = []
		for i in range(len(inputObject)): 
			diffs.append([])
			this_element = inputObject[i]
			for j in range(len(compareObject)): #compare each reversal to all keypress times  
				rev_diffs = compareObject[j] - this_element 
				diffs[i].append(rev_diffs)
		
		#select compareObject nearest to inputObject 		
		nearest_distance = []
		for d in diffs: 
			idx = (np.abs(np.array(d))).argmin() #find index closest to zero	
			nearest_distance.append(d[idx])#append value of this index
		
		return np.array(nearest_distance) #seconds


	def calculate_distance_and_amplitude_reversals_to_peak_and_keypresses(self, analysis_sample_rate=20, data_type='pupil_baseline_zscore', requested_eye='L'): 
		"""Distance between 1) true reversal points & key presses and 2) true reversal points & the peak in pupil_baseline_zscore signals is correlated to inspect
		if timing of behavior and signals is correlated with each other. """

		self.events_and_signals_in_time_behav(data_type= data_type, requested_eye= requested_eye)

		#get chronological trial indices of all reversal positions
		chrono_reversal_position =[]
		counter = 0
		for i in range(len(self.reversal_positions)):
			chrono_reversal_position.append(self.reversal_positions[i] + counter)
			counter = self.run_trial_limits[i][1]
		chrono_reversal_position = np.concatenate(chrono_reversal_position)	
		
		#padded reversal and keypress timings
		padded_reversal_times = np.array([self.padded_sound_times[rev] for rev in chrono_reversal_position])
		padded_keypress_times = self.padded_keypress_times	
		
		#calculate distance between reversal and closest keypress
		reversal_keypress_distance_times = self.calculate_distance(inputObject=padded_reversal_times, compareObject=padded_keypress_times)
				
		#get index of positive distance reversal <> keypress
		positive_distance_indices = np.where(reversal_keypress_distance_times > 0)
		positive_distance_times = reversal_keypress_distance_times[positive_distance_indices]
		
		#select reversal_positions and keypress_positions with positive_distance
		pos_reversal_positions = chrono_reversal_position[positive_distance_indices]
		percentage_positive = len(pos_reversal_positions) / len(chrono_reversal_position) * 100
		self.logger.info('Percentage correctly pressed reversals: %i for participant %s ' %(percentage_positive, self.subject.initials))

		#get deconvolved keypress timecourses to estimate the distance of baseline peak 
		folder_name = 'detrended_padded_pupil_around_keypress' 		
		with pd.get_store(self.ho.inputObject) as h5_file:
			deconvolved_pupil_timecourses = h5_file.get("/%s/%s"%(folder_name, 'deconvolved_pupil_timecourses')).as_matrix()
			covariate_keys = h5_file.get("/%s/%s"%(folder_name, 'covariate_keys'))

		
		#smooth the deconvolved baseline signal a little 
		deconvolved_keypress = myfuncs.smooth(deconvolved_pupil_timecourses[:,0], window_len=100)
		#get the estimate of the max_peak using peak_detect.py 
		[max_peaks, min_peaks] = peak_detect.peakdetect(deconvolved_keypress, lookahead=250, delta=0)
		max_peak_val = np.mean(max_peaks, 0)[1]
		max_peak_idx = max_peaks[0][0]		
		bottom_to_peak_amp = max_peak_val - np.min(deconvolved_keypress)
		self.logger.info('Max peak of %.2f detected at index %i for participant %s ' %(max_peak_val, max_peak_idx, self.subject.initials))

		#calculate the distance in seconds from reversal point to keypress 
		distance_reversal_keypress = np.median(positive_distance_times)
		#calculate the distance in seconds from reversal point to baseline peak
		distance_peak_keypress = ((len(deconvolved_keypress)/2) - max_peak_idx) / analysis_sample_rate
		distance_reversal_peak =  np.median(positive_distance_times) - distance_peak_keypress

		#save relevant variables to correlate later 
		folder_name = 'distance_reversal_keypress_behaviour' 		
		with pd.get_store(self.ho.inputObject) as h5_file:
			h5_file.put("/%s/%s"%(folder_name, 'distance_reversal_keypress'), pd.Series(distance_reversal_keypress))
			h5_file.put("/%s/%s"%(folder_name, 'distance_peak_keypress'), pd.Series(distance_peak_keypress))
			h5_file.put("/%s/%s"%(folder_name, 'distance_reversal_peak'), pd.Series(distance_reversal_peak))
			h5_file.put("/%s/%s"%(folder_name, 'percentage_positive'), pd.Series(percentage_positive))
			h5_file.put("/%s/%s"%(folder_name, 'positive_distance_times'), pd.Series(positive_distance_times))
			h5_file.put("/%s/%s"%(folder_name, 'peak'), pd.DataFrame([bottom_to_peak_amp, max_peak_val, max_peak_idx]))

	
	def calculate_powerspectra_pupil_and_experiment(self, 
											analysis_sample_rate=20.0,
											data_type='pupil_int', 
											requested_eye='L', 
											run_cutoff=400):
		"""calculate_powerspectrum_of_tonic_pupil calculates the per run power spectrum of the tonic pupil signal (signal < 0.1Hz).""" 	
		self.events_and_signals_in_time()
		
		subsample_ratio = int(self.sample_rate/analysis_sample_rate)
		av_trial_duration = np.mean(np.r_[self.end_time_trial[0], np.diff(self.end_time_trial)])
		power_spectrum_run_data=[]
		
		for i, alias in enumerate(self.aliases): 		

			trial_times = self.ho.read_session_data(alias, 'trials')
			trial_phase_times = self.ho.read_session_data(alias, 'trial_phases')
			session_start_EL_time = np.array(trial_phase_times[trial_phase_times['trial_phase_index'] == 1]['trial_phase_EL_timestamp'])[0] 
			session_stop_EL_time = np.array(trial_times['trial_end_EL_timestamp'])[-1]	
			self.sample_rate = self.ho.sample_rate_during_period([session_start_EL_time, session_stop_EL_time], alias)
			subsample_ratio = int(self.sample_rate / analysis_sample_rate)

			#calculate powerspectrum of interpolated pupil data  
			int_pupil_run = np.squeeze(self.ho.signal_during_period(time_period = [session_start_EL_time, session_start_EL_time+(run_cutoff*self.sample_rate)], alias = alias, signal = data_type, requested_eye = requested_eye)) 
			int_pupil_run_zscore = (int_pupil_run - np.mean(int_pupil_run)) /int_pupil_run.std()						
			ds_int_pupil_run = int_pupil_run_zscore[::subsample_ratio]

			signal_length = len(ds_int_pupil_run)
			samples = np.arange(signal_length)
			time = signal_length/analysis_sample_rate
			freqs = samples/time #0-20Hz
			one_sided_freqs = freqs[range(int(signal_length/2))]
			#Fast Forier Transfrom of pupil data
			FFT_pupil = sp.fft(ds_int_pupil_run)/signal_length  #fft and normalise
			one_sided_FFT_pupil = FFT_pupil[range(int(signal_length/2))] #select single-side FFT
			power_spectrum_run = abs(one_sided_FFT_pupil)
			power_spectrum_run_data.append(power_spectrum_run)

		pupil_power_spectrum = np.mean(np.array(power_spectrum_run_data), axis=0)

		#calculate powerspectrum of experimental design 
		min_number_of_trials = self.domain_time[:500]
		exp_length = len(min_number_of_trials)
		exp_samples = np.arange(exp_length)
		exp_time = exp_length*av_trial_duration
		exp_freqs = exp_samples/exp_time
		one_sided_exp_freqs = exp_freqs[range(int(exp_length/2))]
		FFT_exp = sp.fft(min_number_of_trials)/exp_length  #fft and normalise
		one_sided_FFT_exp = FFT_exp[range(int(exp_length/2))] #select single-side FFT
		experiment_power_spectrum= abs(one_sided_FFT_exp)

		folder_name = 'power_spectra_pupil_and_experiment' 		
		with pd.get_store(self.ho.inputObject) as h5_file:
			h5_file.put("/%s/%s"%(folder_name, 'pupil_power_spectrum'), pd.Series(pupil_power_spectrum))
			h5_file.put("/%s/%s"%(folder_name, 'experiment_power_spectrum'), pd.Series(experiment_power_spectrum))
			h5_file.put("/%s/%s"%(folder_name, 'pupil_freqs'), pd.Series(one_sided_freqs))
			h5_file.put("/%s/%s"%(folder_name, 'exp_freqs'), pd.Series(one_sided_exp_freqs))


		
	
	def calculate_time_frequency_spectrum_per_run(self, data_type='pupil_bp_zscore', requested_eye='L', sample_rate=1000, analysis_sample_rate=20, subsample_ratio=50): 
		"""Calculate the FFT powerspectrum of each run to inspect changes in pupil powerspectrum over time """
		
		ds_pupil_data = []

		session_time = 0 
		f = pl.figure(figsize=(8,8)) 
		for i, alias in enumerate(self.aliases): 
			#get pupil data per run 
			trial_times = self.ho.read_session_data(alias, 'trials')
			trial_phase_times = self.ho.read_session_data(alias, 'trial_phases')
			session_start_EL_time = np.array(trial_phase_times[trial_phase_times['trial_phase_index'] == 1]['trial_phase_EL_timestamp'])[0] 
			session_stop_EL_time = np.array(trial_times['trial_end_EL_timestamp'])[-1]	
			pupil = np.squeeze(self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], alias = alias, signal = data_type, requested_eye = requested_eye))
			ds_pupil = pupil[::subsample_ratio]
			ds_pupil_data.append(ds_pupil) #20Hz

			#plot time-frequency spectrum per run  
			Fs = analysis_sample_rate * 2 
			NFFT = int(Fs*10)	#time-window of 10 seconds		
			noverlap = int(Fs*5) #window overlap = 5 seconds 			
			f.subplots_adjust(right=0.8) #reorder placement plot and colorbar
			s = f.add_subplot(4,3,i+1)
			Pxx, freqs, bins, im = pl.specgram(ds_pupil, NFFT=NFFT, Fs=Fs, window=matplotlib.mlab.window_none, detrend=matplotlib.mlab.detrend_none,  noverlap=noverlap, pad_to=None, sides='onesided', scale_by_freq=True, cmap='afmhot') 
			s.set_title('run %s'%str(i+1))
			cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
			f.colorbar(im, cax=cbar_ax)	
			f.text(0.5, 0.98, 'Time-frequency plot of per run pupil signal', ha='center',fontsize=10)
			f.text(0.01, 0.5,'Frequency', va='center', rotation='vertical', fontsize=9)		
			f.text(0.5, 0.01, 'bin #', fontsize=9)	
		pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials + 'time_frequency_spectrum.pdf'))	


	def deconvolve_ridge_filtered_detrended_pupil_baseline(self, 
															analysis_sample_rate=20, 
															interval = [-0.5,5.5], fix_dur=0.5,  
															data_type = 'pupil_bp_zscore', 
															requested_eye = 'L', 
															subsample_ratio=50): 
		"""Ridge regression deconvolution of the padded pupil signal. At the begin and end of each run, 60 seconds of padding is added. Consequenty, all events and other pupil signals used in this analysis are zero-padded as well. """
		#get events and relevant pupil signals 
		self.events_and_signals_in_time_behav(data_type=data_type)
			#get filtered and detrended pupil baseline signal 		
		with pd.get_store(self.ho.inputObject) as h5_file:
			filtered_detrended_pupil_baseline = h5_file.get("/%s/%s"%('filtered_detrended_pupil_baseline_zscore', 'filtered_detrended_residuals_%i'%analysis_sample_rate))
			detrended_pupil_baseline_zscore = h5_file.get("/%s/%s"%('detrended_pupil_baseline_zscore', 'detrended_residuals_%i_not_padded'%analysis_sample_rate))
		
		#downsampling to 20Hz of different tonic pupil signals
		df_filtered_detrended_pupil_baseline = np.diff(filtered_detrended_pupil_baseline)
		ds_pupil_baseline_zscore = self.pupil_baseline_data_z[::subsample_ratio]
		ds_pupil_bp_zscore = self.pupil_data[::subsample_ratio]
		
		#downsampling to 20Hz of phasic pupil signal 
		input_signal =  self.pupil_data[::subsample_ratio] 		
		#relevant regressors 
		blink_times =    [self.blink_times]						
		keypress_times = [self.keypress_times]
		saccade_times =  [self.saccade_times] 
		colour_times =   [self.colour_times] 
		sound_times =    [self.sound_times]
		cue_green =      [self.colour_times[self.hue_indices[0]]]
		cue_purple =     [self.colour_times[self.hue_indices[1]]]
		sound_loss = 	 [self.sound_times[self.sound_indices[0]]]
		sound_win = 	 [self.sound_times[self.sound_indices[1]]]
 
		#per trial scalar of filtered_detrended_pupil_baseline and df_filtered_detrended_pupil_baseline (based on whole trial duration)
		fix_start_idx = np.around(self.fix_times*analysis_sample_rate).astype(int) 
		trial_duration = np.r_[self.end_time_trial[0], np.diff(self.end_time_trial)]
		trial_period = np.array([int(t_d * analysis_sample_rate) for t_d in trial_duration])
		#average tonic pupil signal per trial 
		detrended_tonic_pupil_zscore = np.array([detrended_pupil_baseline_zscore[fix:fix+trial_period[i]].mean(axis=0) for i,fix in enumerate(fix_start_idx)])
		phasic_pupil_baseline_zscore = np.array([ds_pupil_bp_zscore[fix:fix+trial_period[i]].mean(axis=0) for i,fix in enumerate(fix_start_idx)])
		tonic_pupil_zscore = np.array([ds_pupil_baseline_zscore[fix:fix+trial_period[i]].mean(axis=0) for i,fix in enumerate(fix_start_idx)])
		av_filtered_detrended_tonic_pupil = np.array([filtered_detrended_pupil_baseline[fix:fix+trial_period[i]].mean(axis=0) for i,fix in enumerate(fix_start_idx)])
		df_av_filtered_detrended_tonic_pupil = np.array([df_filtered_detrended_pupil_baseline[fix:fix+trial_period[i]].mean(axis=0) for i,fix in enumerate(fix_start_idx)])

		events=[]							#append regressors to events list: 
		events.extend(blink_times) 			#[0]
		events.extend(keypress_times)		#[1]
		events.extend(saccade_times)		#[2]
		events.extend(cue_green)			#[3]
		events.extend(cue_purple)			#[4]
		events.extend(sound_loss)			#[5]
		events.extend(sound_win)			#[6]
		events.extend(colour_times)		    #[7]
		events.extend(sound_times)		    #[8]

		#covariates 
		covariates = {
			'blink.gain': np.ones(len(events[0])), 
			'blink.phasic_pupil_baseline_zscore':phasic_pupil_baseline_zscore,
			'keypress.gain': np.ones(len(events[1])),
			'keypress.phasic_pupil_baseline_zscore':phasic_pupil_baseline_zscore,
			'saccade.gain': np.ones(len(events[2])),
			'saccade.phasic_pupil_baseline_zscore':phasic_pupil_baseline_zscore,
			'cue_green.gain': np.ones(len(events[3])),
			'cue_green.phasic_pupil_baseline_zscore':phasic_pupil_baseline_zscore,
			'cue_purple.gain': np.ones(len(events[4])), 
			'cue_purple.phasic_pupil_baseline_zscore':phasic_pupil_baseline_zscore, 
			'sound_loss.gain': np.ones(len(events[5])), 
			'sound_loss.phasic_pupil_baseline_zscore':phasic_pupil_baseline_zscore,
			'sound_win.gain': np.ones(len(events[6])),
			'sound_win.phasic_pupil_baseline_zscore':phasic_pupil_baseline_zscore,
			'colour_times.phasic_pupil_baseline_zscore': phasic_pupil_baseline_zscore, 
			'sound_times.phasic_pupil_baseline_zscore': phasic_pupil_baseline_zscore, 						
		} 
		
		fd = FIRDeconvolution(
					signal = input_signal, 
					events = events,
					event_names = ['blink', 'keypress', 'saccade','cue_green','cue_purple','sound_loss','sound_win', 'colour_times', 'sound_times'], 
					sample_frequency = analysis_sample_rate, 
					deconvolution_interval = [-0.5, 5.5], 
					deconvolution_frequency = analysis_sample_rate,
					covariates = covariates,
					) 
		
		fd.create_design_matrix()
		#inspect design matrix 
		plot_time = 5000
		f = pl.figure()
		s = f.add_subplot(111)		
		s.set_title('design matrix (%i Hz)'%analysis_sample_rate)
		pl.imshow(fd.design_matrix[:,:plot_time], aspect = 0.075 * plot_time/fd.deconvolution_interval_size, interpolation = 'nearest', rasterized = True)
		sn.despine(offset=10)
		# pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials + '_designmatrix_%iHz_%s.pdf'%(analysis_sample_rate, best_sim_params)))
	
		#ridge regression
		self.logger.info('Starting ridge regression on using phasic_pupil_baseline_zscore as covariate for all regressors for participant %s' %self.subject.initials)
		fd.ridge_regress(cv=10, alphas=np.linspace(1,200,30)) #(no intercept is fitted) cv = 20, np.logspace(7, 0, 20), alphas=np.linspace(1,200,30)
		fd.calculate_rsq()
		self.logger.info('Rsqured for participant %s: %f' %(self.subject.initials, fd.rsq))

		betas=[]
		for b in fd.covariates.keys(): #save the betas in the order of the covariates keys order
			beta = fd.betas_for_cov(covariate=b)
			betas.append(beta)
		
		#folder_name = 'deconvolve_padded_ridge_regression_%s'%analysis_sample_rate
		# folder_name = 'deconvolve_padded_ridge_regression_%s_av_pupil_baseline'%analysis_sample_rate
		#folder_name = 'deconvolve_ridge_regression_%s_tonic_pupil_zscore'%analysis_sample_rate
		#folder_name = 'deconvolve_ridge_regression_%s_detrended_tonic_pupil_zscore'%analysis_sample_rate
		folder_name = 'deconvolve_ridge_regression_%s_phasic_pupil_baseline_zscore'%analysis_sample_rate #ran until id des
		#store deconvolution 
		with pd.get_store(self.ho.inputObject) as h5_file:
			h5_file.put("/%s/%s"%(folder_name, 'time_points'), pd.Series(fd.deconvolution_interval_timepoints))
			h5_file.put("/%s/%s"%(folder_name, 'dec_time_course'), pd.DataFrame(np.squeeze(betas).T))
			h5_file.put("/%s/%s"%(folder_name, 'rsquared'), pd.Series(fd.rsq))
			h5_file.put("/%s/%s"%(folder_name, 'covariate_keys'), pd.DataFrame(fd.covariates.keys()))
			h5_file.put("/%s/%s"%(folder_name, 'alpha_value'), pd.Series(fd.rcv.alpha_))

		self.logger.info('Saved deconvolution data for %s in hdf5 with folder_name: %s ' %(self.subject.initials, folder_name))

		#plot regression
		plot_time = 1000
		f = pl.figure(figsize = (10,7)) 
		pl.title('Ridge regression of pupil response')
		s = f.add_subplot(331)
		hues = sn.color_palette()[:3]
		for i,betas in enumerate(['blink.gain', 'keypress.gain', 'saccade.gain']): 
			beta = fd.betas_for_cov(covariate=betas)
			pl.plot(fd.deconvolution_interval_timepoints, beta, color=hues[i])
		for i,betas in enumerate(['blink.phasic_pupil_baseline_zscore', 'keypress.phasic_pupil_baseline_zscore', 'saccade.phasic_pupil_baseline_zscore']): 
			beta = fd.betas_for_cov(covariate=betas)
			pl.plot(fd.deconvolution_interval_timepoints, beta, ls='--', color=hues[i])
		pl.legend(['blink', 'keypress', 'saccade','blink*phasic_pupil_baseline_zscore', 'keypress*phasic_pupil_baseline_zscore', 'saccade*phasic_pupil_baseline_zscore'])
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')	
		sn.despine(offset=10)		
		s = f.add_subplot(332)
		hues = sn.color_palette()[1:4:2] #select green and purple 
		for i, betas in enumerate(['cue_green.gain', 'cue_purple.gain']): 
			beta = fd.betas_for_cov(covariate=betas)
			pl.plot(fd.deconvolution_interval_timepoints, beta, color=hues[i])
		for i, betas in enumerate(['cue_green.phasic_pupil_baseline_zscore', 'cue_purple.phasic_pupil_baseline_zscore']): 
			beta = fd.betas_for_cov(covariate=betas)
			pl.plot(fd.deconvolution_interval_timepoints, beta, ls='--', color=hues[i])
		pl.legend(['cue green', 'cue purple', 'cue_green*phasic_pupil_baseline_zscore', 'cue_purple*phasic_pupil_baseline_zscore'])
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')	
		sn.despine(offset=10)		
		s = f.add_subplot(333)	
		hues = sn.color_palette()[1:3] #select green and red 			
		for i,betas in enumerate(['sound_win.gain','sound_loss.gain']):  
			beta = fd.betas_for_cov(covariate=betas)
			pl.plot(fd.deconvolution_interval_timepoints, beta, color=hues[i])
		for i,betas in enumerate(['sound_win.phasic_pupil_baseline_zscore','sound_loss.phasic_pupil_baseline_zscore']): 
			beta = fd.betas_for_cov(covariate=betas)
			pl.plot(fd.deconvolution_interval_timepoints, beta, ls='--', color=hues[i])
		pl.legend(['sound win','sound loss', 'sound_win*phasic_pupil_baseline_zscore', 'sound_loss*phasic_pupil_baseline_zscore'])
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')	
		sn.despine(offset=10)
		s = f.add_subplot(334)
		for betas in ['sound_times.phasic_pupil_baseline_zscore']: 
			beta = fd.betas_for_cov(covariate=betas)
			pl.plot(fd.deconvolution_interval_timepoints, beta)
		pl.legend(['sound*phasic_pupil_baseline_zscore'])		
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')	
		sn.despine(offset=10)			
		s = f.add_subplot(335)
		for betas in ['colour_times.phasic_pupil_baseline_zscore']: 
			beta = fd.betas_for_cov(covariate=betas)
			pl.plot(fd.deconvolution_interval_timepoints, beta)
		pl.legend(['colour*phasic_pupil_baseline_zscore'])		
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')	
		sn.despine(offset=10)	
		s = pl.subplot2grid((3,3), (2,0), colspan=3)
		s.set_title('data and predictions, R squared: %1.3f,  ridge alpha: %1.3f '%(fd.rsq, fd.rcv.alpha_))
		pl.plot(np.linspace(0,plot_time, int(plot_time * fd.deconvolution_frequency/fd.sample_frequency)), 
			fd.resampled_signal[:,:int(plot_time * fd.deconvolution_frequency/fd.sample_frequency)].T, 'r')
		pl.plot(np.linspace(0,plot_time, int(plot_time * fd.deconvolution_frequency/fd.sample_frequency)), 
			fd.predict_from_design_matrix(fd.design_matrix[:,:int(plot_time * fd.deconvolution_frequency/fd.sample_frequency)]).T, 'k')
		pl.legend(['signal','explained'])
		sn.despine(offset=10)		
		pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials + '_deconv_%iHz_ridge_phasic_pupil_baseline_zscore.pdf'%analysis_sample_rate)) 
		self.logger.info('Plotted %s for %s ' %(folder_name, self.subject.initials))



	def deconvolve_ridge_phasic_pupil_and_tonic_baselines(self, 
														  analysis_sample_rate=20, 
														  interval=[-0.5, 6.0], 
														  fix_dur=0.5, 
														  data_type='pupil_bp_zscore', 
														  requested_eye = 'L', 
														  subsample_ratio=50):
		"""Performs ridge regression on the phasic pupil signal using phasic event regressors, as well as slow pupil signals from self.filter_bank. 
		self.filter_bank consists of low frequency pupil signals with frequencies ranging from [0.1Hz - 0.002Hz]. The analysis calculates typical 
		pupil responses (gains) to the events and correlations between each filter_bank signal with the phasic pupil signal (on a per-trial basis).
		All event-regressor and correlation values are stored in the dict 'covariate', where event-regressor values are referred
		to as name_of_event_regressor.gain (e.g. 'blink.gain') and per-trial-baseline values are referred as 
		name_of_event_regressor.frequency (e.g. 'blink.00200'). After the analysis, betas and other relevant information is stored in HDF5 """

		#import tonic filter bank signals 
		self.filter_bank_pupil_signals(data_type=data_type, requested_eye=requested_eye, do_plot=True)
		#import phasic pupil events and signal 
		self.events_and_signals_in_time_behav(data_type= data_type, requested_eye=requested_eye)
		#downsample phasic pupil signal to be analysed in ridge regression  
		input_signal =  self.pupil_data[::subsample_ratio] 
		
		#relevant phasic event regressors 
		blink_times =    [self.blink_times]						
		keypress_times = [self.keypress_times]
		saccade_times =  [self.saccade_times] 
		colour_times =   [self.colour_times] 
		sound_times =    [self.sound_times]
		cue_green =      [self.colour_times[self.hue_indices[0]]]  
		cue_purple =     [self.colour_times[self.hue_indices[1]]]  
		cue_low =  		 [self.colour_times[self.reward_prob_indices[0]]]
		cue_high = 	 	 [self.colour_times[self.reward_prob_indices[1]]]
		sound_loss = 	 [self.sound_times[self.sound_indices[0]]]
		sound_win = 	 [self.sound_times[self.sound_indices[1]]]

		#get per trial fixation start and duration to calculate the pupil baseline across the trial for each self.filter_bank frequency 
		fix_start_idx = np.around(self.fix_times*analysis_sample_rate).astype(int) 
		trial_duration = np.r_[self.end_time_trial[0], np.diff(self.end_time_trial)]
		trial_period = np.array([int(t_d * analysis_sample_rate) for t_d in trial_duration])
		
		#extract per trial baseline of each filter_bank signal 
		filter_bank_per_trial_bl = {}
		for key, signal in self.filter_bank.items(): 
			#downsample
			ds_filter_bank_signal = self.filter_bank[key][::subsample_ratio]
			#calculate per-trial baseline 
			this_filter_baseline = np.array([ds_filter_bank_signal[fix:fix+trial_period[i]].mean(axis=0) for i, fix in enumerate(fix_start_idx)])
			filter_bank_per_trial_bl[key] = this_filter_baseline #klopt dit??? 

		events=[]							#append regressors to events list: 
		events.extend(blink_times) 			#[0]
		events.extend(keypress_times)		#[1]
		events.extend(saccade_times)		#[2]
		events.extend(cue_green)			#[3]
		events.extend(cue_purple)			#[4]
		events.extend(cue_low)				#[5]
		events.extend(cue_high)				#[6]
		events.extend(sound_loss)			#[7]
		events.extend(sound_win)			#[8]
		events.extend(colour_times)		    #[9]
		events.extend(sound_times)		    #[10]

		#covariate gains
		covariates = {
			'blink.gain': np.ones(len(events[0])),
			'keypress.gain': np.ones(len(events[1])),
			'saccade.gain': np.ones(len(events[2])),
			'cue_green.gain': np.ones(len(events[3])),
			'cue_purple.gain': np.ones(len(events[4])), 
			'cue_low.gain': np.ones(len(events[5])),
			'cue_high.gain': np.ones(len(events[6])),
			'sound_loss.gain': np.ones(len(events[7])), 
			'sound_win.gain': np.ones(len(events[8])),
		}
		
		#remove dots from filter_bank_keys so FIRDeconvolution does not get confused by multiple dots in covariate names
		filt_keys_no_dots = [self.filter_bank.keys()[i].rpartition('.')[-1] for i in range(len(self.filter_bank.keys()))] 

		#update covariates dict with filter_bank baseline signals
		for name in ['keypress.','cue_low.', 'cue_high.', 'sound_loss.', 'sound_win.', 'colour_times.', 'sound_times.']: 
			#make covariate keys for FIRDeconvolution
			covariate_name = [name+filt for filt in filt_keys_no_dots] 
			#zip covariate_names to filter_bank values and put in dict
			covariate_pair = dict(zip(covariate_name, filter_bank_per_trial_bl.values()))
			#update the covariates dict with covariate pair 
			covariates.update(covariate_pair)

		#make FIRDeconvolution object
		fd = FIRDeconvolution(
			signal = input_signal, 
			events = events,
			event_names = ['blink', 'keypress', 'saccade','cue_green','cue_purple', 'cue_low', 'cue_high', 'sound_loss','sound_win', 'colour_times', 'sound_times'], 
			sample_frequency = analysis_sample_rate, 
			deconvolution_interval = [-0.5, 5.5], 
			deconvolution_frequency = analysis_sample_rate,
			covariates = covariates,
			) 

		fd.create_design_matrix()	
		#inspect design matrix 
		# plot_time = 5000
		# f = pl.figure()
		# s = f.add_subplot(111)		
		# s.set_title('design matrix (%i Hz)'%analysis_sample_rate)
		# pl.imshow(fd.design_matrix[:,:plot_time], aspect = 0.075 * plot_time/fd.deconvolution_interval_size, interpolation = 'nearest', rasterized = True)
		# sn.despine(offset=10)

		#ridge regression
		self.logger.info('Starting ridge regression deconvolve_ridge_phasic_pupil_and_tonic_baselines for participant %s' %self.subject.initials)
		fd.ridge_regress(cv=10, alphas=np.linspace(0,2000,30)) #how big should alpha get? linspace or logspace? 

		#pre-allocate betas matrix 
		betas = pd.DataFrame(np.zeros((fd.deconvolution_interval_size, len(fd.covariates))), columns=fd.covariates.keys())
		for i,b in enumerate(fd.covariates.keys()): 
			betas[b] = np.squeeze(fd.betas_for_cov(covariate=b))

		#save all important variables in hdf5
		folder_name = 'ridge_phasic_%s_tonic_baselines' %str(len(self.filter_bank))
		with pd.get_store(self.ho.inputObject) as h5_file:
			h5_file.put("/%s/%s"%(folder_name, 'time_points'), pd.Series(fd.deconvolution_interval_timepoints))
			h5_file.put("/%s/%s"%(folder_name, 'deconvolved_pupil_timecourses'), betas)
			#h5_file.put("/%s/%s"%(folder_name, 'rsquared'), pd.Series(fd.rsq))
			h5_file.put("/%s/%s"%(folder_name, 'alpha_value'), pd.Series(fd.rcv.alpha_))

		self.logger.info('Saved deconvolution data for %s in hdf5 with folder_name: %s ' %(self.subject.initials, folder_name))


	def deconvolve_ridge_covariates(self, analysis_sample_rate=20, interval = [-0.5,5.5], fix_dur=0.5,  data_type = 'pupil_bp_zscore', requested_eye = 'L', use_domain = 'full', best_sim_params='average_TD_params'): 
		"""Single pass deconvolution using FIRDeconvolution."""

		#import TD value, rpe regressor, events_and_signals_in_time regressors 
		self.TD_states(data_type = data_type, best_sim_params='average_TD_params')
		self.events_and_signals_in_time(data_type = data_type, requested_eye= requested_eye)
		self.events_and_signals_in_time_behav(data_type= data_type, requested_eye= requested_eye)

		input_signal = sp.signal.decimate(self.pupil_data, int(self.sample_rate / analysis_sample_rate))[20:]
		
		#select regressors 
		blink_times =    [self.blink_times]						
		keypress_times = [self.keypress_times]
		saccade_times =  [self.saccade_times] 
		colour_times =   [self.colour_times] 
		sound_times =    [self.sound_times]
		cue_green =      [self.colour_times[self.hue_indices[0]]]
		cue_purple =     [self.colour_times[self.hue_indices[1]]]
		sound_loss = 	 [self.sound_times[self.sound_indices[0]]]
		sound_win = 	 [self.sound_times[self.sound_indices[1]]]

		#downsampled pupil baseline data 
		pupil_baseline_data_z = sp.signal.decimate(self.pupil_baseline_data_z, int(self.sample_rate / analysis_sample_rate))[20:] #downsampled pupil_baseline_zscore signal	
		pupil_baseline_data_raw = sp.signal.decimate(self.pupil_baseline_data, int(self.sample_rate / analysis_sample_rate))[20:] #downsampled pupil_baseline non zscored signal
		#derivative of downsampled pupil baseline data signal 
		df_pupil_baseline_raw = np.diff(pupil_baseline_data_raw)
		df_pupil_baseline_z = df_pupil_baseline_raw - np.mean(df_pupil_baseline_raw, axis=0)/df_pupil_baseline_raw.std()
		
		#per trial pupil baseline scalar and df pupil baseline scalar(based on fixation period) 
		fix_start_idx = np.around(self.fix_times*analysis_sample_rate).astype(int) 
		fix_period = int(fix_dur * analysis_sample_rate)
		pupil_baseline_fix = np.array([pupil_baseline_data_z[fix:fix+fix_period].mean(axis=0) for fix in fix_start_idx])
		pupil_df_fix = np.array([df_pupil_baseline_z[fix:fix+fix_period].mean(axis=0) for fix in fix_start_idx])
		
		#per trial TD value regressor 
		TD_statevalues_diff = np.copy(self.best_statevalues[:,-1])
		zscored_TD_statevalues_diff = (TD_statevalues_diff - np.mean(TD_statevalues_diff, axis=0))/TD_statevalues_diff.std() #more extreme than original TD state values

		#per trial RPE regressor 
		signed_RPE = np.copy(self.raw_signed_prediction_error)
		zscored_signed_RPE =  np.copy(self.z_scored_signed_prediction_error)
		zscored_unsigned_RPE = np.copy(self.z_scored_unsigned_prediction_error)

		events=[]							#append regressors to events list: 
		events.extend(blink_times) 			#[0]
		events.extend(keypress_times)		#[1]
		events.extend(saccade_times)		#[2]
		events.extend(cue_green)			#[3]
		events.extend(cue_purple)			#[4]
		events.extend(sound_loss)			#[5]
		events.extend(sound_win)			#[6]
		events.extend(colour_times)		    #[7]
		events.extend(sound_times)		    #[8]

		#covariates 
		covariates = {
			'blink.gain': np.ones(len(events[0])), 
			#'blink.pupil_baseline': pupil_baseline_fix,
			'blink.df_baseline':pupil_df_fix,
			'keypress.gain': np.ones(len(events[1])),
			#'keypress.pupil_baseline': pupil_baseline_fix,
			'keypress.df_baseline':pupil_df_fix,
			'saccade.gain': np.ones(len(events[2])),
			#'saccade.pupil_baseline': pupil_baseline_fix,
			'saccade.df_baseline':pupil_df_fix,
			'cue_green.gain': np.ones(len(events[3])),
			#'cue_green.pupil_baseline': pupil_baseline_fix,
			'cue_green.df_baseline':pupil_df_fix,
			'cue_purple.gain': np.ones(len(events[4])), 
			#'cue_purple.pupil_baseline': pupil_baseline_fix,
			'cue_purple.df_baseline':pupil_df_fix, 
			'sound_loss.gain': np.ones(len(events[5])), 
			#'sound_loss.pupil_baseline': pupil_baseline_fix,
			'sound_loss.df_baseline':pupil_df_fix,
			'sound_win.gain': np.ones(len(events[6])),
			#'sound_win.pupil_baseline': pupil_baseline_fix,
			'sound_win.df_baseline':pupil_df_fix,
			'colour_times.TD_value': zscored_TD_statevalues_diff, 
			'sound_times.signed_RPE': zscored_signed_RPE, 						
		} 
				
		fd = FIRDeconvolution.FIRDeconvolution(
					signal = input_signal, 
					events = events,
					event_names = ['blink', 'keypress', 'saccade','cue_green','cue_purple','sound_loss','sound_win', 'colour_times', 'sound_times'], # 'colour_times', 'sound_times'
					sample_frequency = analysis_sample_rate, 
					deconvolution_interval = [-0.5, 5.5], 
					deconvolution_frequency = analysis_sample_rate,
					covariates = covariates,
					) 

		
		fd.create_design_matrix()	
		#inspect design matrix 
		plot_time = 1000
		f = pl.figure()
		s = f.add_subplot(111)		
		s.set_title('design matrix (%i Hz)'%analysis_sample_rate)
		pl.imshow(fd.design_matrix[:,:plot_time], aspect = 0.075 * plot_time/fd.deconvolution_interval_size, cmap = 'RdBu', interpolation = 'nearest', rasterized = True)
		sn.despine(offset=10)
		pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials + '_designmatrix_%iHz_%s.pdf'%(analysis_sample_rate, best_sim_params)))

		#ridge regression
		self.logger.info('Starting ridge regression for participant %s with %s ' %(self.subject.initials, best_sim_params))
		fd.ridge_regress(cv=10, alphas=np.linspace(1,200,30)) #(no intercept is fitted) cv = 20, np.logspace(7, 0, 20)
		fd.calculate_rsq()
		self.logger.info('Rsqured for participant %s: %f' %(self.subject.initials, fd.rsq))

		betas=[]
		for b in fd.covariates.keys(): #save the betas in the order of the covariates keys order
			beta = fd.betas_for_cov(covariate=b)
			betas.append(beta)
		
		folder_name = 'deconvolve_full_%iHz_ridge_%s_derivative'%(analysis_sample_rate, best_sim_params) 
		#store deconvolution 
		with pd.get_store(self.ho.inputObject) as h5_file:
			h5_file.put("/%s/%s"%(folder_name, 'time_points'), pd.Series(fd.deconvolution_interval_timepoints))
			h5_file.put("/%s/%s"%(folder_name, 'dec_time_course'), pd.DataFrame(np.squeeze(betas).T))
			h5_file.put("/%s/%s"%(folder_name, 'rsquared'), pd.Series(fd.rsq))
			h5_file.put("/%s/%s"%(folder_name, 'covariate_keys'), pd.DataFrame(fd.covariates.keys()))
			h5_file.put("/%s/%s"%(folder_name, 'alpha_value'), pd.Series(fd.rcv.alpha_))

		#plot regression
		plot_time = 1000
		f = pl.figure(figsize = (8,7)) 
		pl.title('Ridge regression of pupil response')
		s = f.add_subplot(421)
		for betas in ['blink.gain', 'keypress.gain', 'saccade.gain']: 
			beta = fd.betas_for_cov(covariate=betas)
			pl.plot(fd.deconvolution_interval_timepoints, beta)
		pl.legend(['blink', 'keypress', 'saccade'])
		s.set_title('IRF')
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')	
		sn.despine(offset=10)
		s = f.add_subplot(422)
		# for betas in ['blink.pupil_baseline', 'keypress.pupil_baseline', 'saccade.pupil_baseline']: 
		# 	beta = fd.betas_for_cov(covariate=betas)
		# 	pl.plot(fd.deconvolution_interval_timepoints, beta, ls='--')
		# pl.legend(['blink*baseline', 'key*baseline', 'saccade*baseline'])	
		# s.set_title('IRF * pupil baseline')	
		# pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		# pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')	
		# sn.despine(offset=10)	
		s = f.add_subplot(423)
		for betas in ['blink.df_baseline', 'keypress.df_baseline', 'saccade.df_baseline']: 
			beta = fd.betas_for_cov(covariate=betas)
			pl.plot(fd.deconvolution_interval_timepoints, beta, ls='--')
		pl.legend(['blink*df_base', 'key*df_base', 'saccade*df_base'])	
		s.set_title('IRF * derivative pupil baseline')	
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')	
		sn.despine(offset=10)	
		s= f.add_subplot(424)
		for betas in ['cue_green.gain', 'cue_purple.gain', 'sound_loss.gain', 'sound_win.gain']: 
			beta = fd.betas_for_cov(covariate=betas)
			pl.plot(fd.deconvolution_interval_timepoints, beta)
		pl.legend(['cue green', 'cue purple', 'sound loss', 'sound win'])
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')	
		sn.despine(offset=10)
		# s = f.add_subplot(435)
		# for betas in ['cue_green.pupil_baseline', 'cue_purple.pupil_baseline', 'sound_loss.pupil_baseline', 'sound_win.pupil_baseline']: 
		# 	beta = fd.betas_for_cov(covariate=betas)
		# 	pl.plot(fd.deconvolution_interval_timepoints, beta, ls='--')
		# pl.legend(['cue_green*baseline', 'cue_purple*baseline', 'sound_loss*baseline', 'sound_win*baseline'])		
		# pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		# pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')	
		# sn.despine(offset=10)	
		s = f.add_subplot(425)
		for betas in ['cue_green.df_baseline', 'cue_purple.df_baseline', 'sound_loss.df_baseline', 'sound_win.df_baseline']: 
			beta = fd.betas_for_cov(covariate=betas)
			pl.plot(fd.deconvolution_interval_timepoints, beta, ls='--')
		pl.legend(['cue_green*df_base', 'cue_purple*df_base', 'sound_loss*df_base', 'sound_win*df_base'])		
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')	
		sn.despine(offset=10)	
		s = f.add_subplot(426)
		for betas in ['colour_times.TD_value', 'sound_times.signed_RPE']: 
			beta = fd.betas_for_cov(covariate=betas)
			pl.plot(fd.deconvolution_interval_timepoints, beta)
		pl.legend(['TD_value(zscored)', 'Signed RPE (zscored)'])		
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')	
		sn.despine(offset=10)	
		s = pl.subplot2grid((4,2), (3,0), colspan=2)
		s.set_title('data and predictions, R squared: %1.3f,  ridge alpha: %1.3f '%(fd.rsq, fd.rcv.alpha_))
		pl.plot(np.linspace(0,plot_time, int(plot_time * fd.deconvolution_frequency/fd.sample_frequency)), 
			fd.resampled_signal[:,:int(plot_time * fd.deconvolution_frequency/fd.sample_frequency)].T, 'r')
		pl.plot(np.linspace(0,plot_time, int(plot_time * fd.deconvolution_frequency/fd.sample_frequency)), 
			fd.predict_from_design_matrix(fd.design_matrix[:,:int(plot_time * fd.deconvolution_frequency/fd.sample_frequency)]).T, 'k')
		pl.legend(['signal','explained'])
		sn.despine(offset=10)		
		pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials + '_full_deconv_%iHz_ridge_%s_deriv.pdf'%(analysis_sample_rate, best_sim_params)))


		

	def single_trial_GLM_dual_kernel_behav(self, 
								input_data_type = 'residuals',
								irf_kernel_params = {'a_event': 2.66242099, 'loc_event': 0.34845018,'scale_event': 0.4001978,
													'a_reward':2.13286891, 'loc_reward': 1.03410114,'scale_reward': 1.11341402},
								kernel_time_interval = [0,10.0], 
								analysis_sample_rate = 25,
								requested_eye = 'L',
								subsample_ratio = 5, 
								microsaccades_added=False,
													):
		""" single_trial_GLM_dual_gamma_kernel_results performs single trial GLM on input_data_type = 'residuals' of first deconvolution (main effects of cue, sound and blinks are regressed out). 
		The function uses hard coded IRF parameter values from kernel_fit_gamma (de Gee, 2014): {'a_event/reward': shape of event kernel, 'loc_event/reward': location of the event kernel, 
		'scale_event/reward': scale of the event kernel} and adjusts this to fit the experiment space. 
		Per trial, the two IRF shapes are fitted to the two event types cue and sound (in total 4 regressors per trial), resulting in 800 regressors per  event type, 3200 regressors in total per participant. 
		The GLM is evaluated with statmodelfit OLS, the result is saved in a subject specific pickle file."""

		event_kernel = stats.gamma.pdf(np.linspace(kernel_time_interval[0], kernel_time_interval[1], (kernel_time_interval[1]-kernel_time_interval[0]) * analysis_sample_rate), 
										a = irf_kernel_params['a_event'], loc = irf_kernel_params['loc_event'], scale = irf_kernel_params['scale_event'])
		reward_kernel = stats.gamma.pdf(np.linspace(kernel_time_interval[0], kernel_time_interval[1], (kernel_time_interval[1]-kernel_time_interval[0]) * analysis_sample_rate), 
										a = irf_kernel_params['a_reward'], loc = irf_kernel_params['loc_reward'], scale = irf_kernel_params['scale_reward'])
		
		#plot probability density function of gamma fit:   
		timepoints = np.linspace(0, 7.5, event_kernel.size)
		pl.figure()
		pl.plot(timepoints, event_kernel, 'r', timepoints, reward_kernel, 'b')
		pl.legend(['event_kernel', 'reward_kernel'])
		pl.title('probability density function of event and reward kernel')
		pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials + '_PDF_event_reward_kernel.pdf'))

		if input_data_type == 'residuals':
			with pd.get_store(self.ho.inputObject) as h5_file:									   
				try:																			   
					input_data = h5_file.get("/%s/%s"%('standard_deconvolve_keypress_%s'%str(microsaccades_added), 'residuals')) 
				except (IOError, NoSuchNodeError):
					self.logger.error("no residuals present")	
				finally:
					self.events_and_signals_in_time(data_type = 'pupil_bp_zscore', requested_eye= requested_eye)
			
		else:
			self.events_and_signals_in_time(data_type = input_data_type, requested_eye = requested_eye) 
			input_data = self.pupil_data

		# create timepoints for events
		total_nr_regressors = (self.sound_times.shape[0] + self.colour_times.shape[0]) * 2  # for two types of IRF shapes per event, 4 regressors per trial
		raw_design_matrix = np.zeros((total_nr_regressors,len(input_data)))
		 
		# fill in sound times regressor events		
		sound_sample_indices = np.array(np.round(self.sound_times * analysis_sample_rate), dtype = int)
		colour_sample_indices = np.array(np.round(self.colour_times * analysis_sample_rate), dtype = int)

		convolved_design_matrix = np.zeros(raw_design_matrix.shape)
		for kernel, event_indices, shift in zip([event_kernel, event_kernel, reward_kernel, reward_kernel], 
												[sound_sample_indices, colour_sample_indices, sound_sample_indices, colour_sample_indices], 
												np.arange(4) * self.sound_times.shape[0]):
			for i,t in enumerate(event_indices):		# fast event kernel regressors for sound events
				raw_design_matrix[i + shift,t] = 1;	
				convolved_design_matrix[i + shift] = fftconvolve(raw_design_matrix[i + shift], kernel, 'full')[:convolved_design_matrix[i+shift].shape[0]] # implicit padding here, done by indexing

		#demean all regressors 
		self.convolved_design_matrix = (convolved_design_matrix.T - convolved_design_matrix.mean(axis=1)).T 		

		#multiple regression using statmodelfit 
		X = sm.add_constant(convolved_design_matrix[:,::subsample_ratio].T) #take every 5th element of convolved_design_matrix to speed up calculation
		model = sm.OLS(input_data[::subsample_ratio], X)
		results = model.fit()
		self.logger.info(results.summary())
		results.save(os.path.join(self.base_directory, 'processed', self.subject.initials + '_single_trial_GLM_dual_gamma_kernel_demeaned_regressors_%s.pickle'%input_data_type))

	
	def single_trial_GLM_dual_results_behav(self, input_data_type='residuals', data_type='pupil_bp_zscore', requested_eye='L', interval=0.5, analysis_sample_rate=25): 
		"""single_trial_GLM_dual_results_behav analyses the GLM results of single_trial_GLM_dual_kernel_behav. Only  correct runs from events_and_signals_in_time_behav are used to order the beta values into 
		meaningfull trial type categories (PRPE, NRPE, HPHR, LPLR)""" 
		
		self.events_and_signals_in_time_behav(data_type = data_type, requested_eye = requested_eye)
		 
		##Single trial GLM results
		GLM_results = np.load(self.base_directory + '/processed/%s_single_trial_GLM_dual_gamma_kernel_demeaned_regressors_%s.pickle'%(self.subject.initials, input_data_type))		
		betas = GLM_results.params[1:] #all beta values - constant
		split_betas = np.split(betas, 4)
		beta_sound_event=np.array(split_betas[0])
		beta_colour_event=np.array(split_betas[1])
		beta_sound_reward=np.array(split_betas[2])
		beta_colour_reward=np.array(split_betas[3])

		#Beta values of correct runs 
		correct_beta_sound_event = np.concatenate(np.array([beta_sound_event[i[0]:i[1]] for i in self.run_trial_limits])[self.correct_indices])
		correct_beta_colour_event = np.concatenate(np.array([beta_colour_event[i[0]:i[1]] for i in self.run_trial_limits])[self.correct_indices])
		correct_beta_sound_reward = np.concatenate(np.array([beta_sound_reward[i[0]:i[1]] for i in self.run_trial_limits])[self.correct_indices])
		correct_beta_colour_reward = np.concatenate(np.array([beta_colour_reward[i[0]:i[1]] for i in self.run_trial_limits])[self.correct_indices])


		#Reversal indices of correct runs 
		timepoints = np.linspace(0, 85, self.correct_domain_time.size) 
		dts_per_run = [self.correct_domain_time[i[0]:i[1]] for i in self.run_trial_limits_corrected]
		self.correct_reversal_indices = np.array([np.append(0, block) for block in self.correct_reversal_blocks])
		#append the last trial to the last block to let cumulative_correct_indices run to the last trial 
		
		if self.subject.initials == 'ta': 
			self.correct_reversal_indices[-1] = np.r_[self.correct_reversal_indices[-1], self.last_trial_per_run_corrected[-1]]
		else:
			self.correct_reversal_indices[-1] = np.hstack((self.correct_reversal_indices[-1], self.last_trial_per_run_corrected[-1]))

		cum_correct_indices = [] 
		counter = 0 
		for i in range(len(self.correct_reversal_indices)): 
			cum_correct_indices.append(self.correct_reversal_indices[i] + counter) 
			counter = self.run_trial_limits_corrected[i][1]	 

		#Cumulative correct reversal block indices and limits 
		block_indices = np.unique(np.concatenate(cum_correct_indices)) 	
		block_limits = np.array([block_indices[:-1],block_indices[1:]]).T  	

		#Relevant trial type information of correct runs 
		colours = self.real_reward_probability * self.hue_indices
		green = colours[0]; purple = colours[1]	
		ls_colour, rw_colour = self.reward_prob_indices[0], self.reward_prob_indices[1]	
		ls_sound, rw_sound = self.sound_indices[0], self.sound_indices[1]
		green_correct_runs = np.array([green[i[0]:i[1]] for i in self.run_trial_limits])[self.correct_indices]
		purple_correct_runs = np.array([purple[i[0]:i[1]] for i in self.run_trial_limits])[self.correct_indices]
	
		#Correct prediction error trials 
		PRPE= ls_colour * rw_sound; NRPE= rw_colour * ls_sound
		PRPE_correct_runs = np.concatenate(np.array([PRPE[i[0]:i[1]] for i in self.run_trial_limits])[self.correct_indices])
		NRPE_correct_runs = np.concatenate(np.array([NRPE[i[0]:i[1]] for i in self.run_trial_limits])[self.correct_indices])
		#Correct predicted trials 
		LPNR = ls_colour * ls_sound; HPHR= rw_colour * rw_sound
		LPNR_correct_runs = np.concatenate(np.array([LPNR[i[0]:i[1]] for i in self.run_trial_limits])[self.correct_indices]) 
		HPHR_correct_runs = np.concatenate(np.array([HPHR[i[0]:i[1]] for i in self.run_trial_limits])[self.correct_indices])

		#Give each trial an index within its reversal block 
		block_indexed_trials = np.concatenate([np.arange(0,bl[1]-bl[0]) for bl in block_limits]) 
		
		#Give all reward trial types an index within its reversal block 
		prpe_order = np.argsort(block_indexed_trials[PRPE_correct_runs]) #order trials on position in reversal block (early --> late)
		nrpe_order = np.argsort(block_indexed_trials[NRPE_correct_runs])
		hphr_order = np.argsort(block_indexed_trials[HPHR_correct_runs]) 
		lplr_order = np.argsort(block_indexed_trials[LPNR_correct_runs])

		#demeaned and standardised difference scores of reward kernel fitted colour and sound beta values 
		z_correct_beta_sound_reward = (correct_beta_sound_reward - correct_beta_sound_reward.mean())/correct_beta_sound_reward.std() 
		z_correct_beta_colour_reward = (correct_beta_colour_reward - correct_beta_colour_reward.mean()) / correct_beta_colour_reward.std()
		
		#calculate slope 
		prpe_betas = correct_beta_sound_reward[PRPE_correct_runs][prpe_order]
		time = np.arange(len(prpe_betas))
		slope, intercept, r_value, p_value, std_err=  sp.stats.linregress(prpe_betas,time)

		folder_name = 'betas_correct_runs'
		#Bar and line plots 
		pe_sw, nope_sw = 3.0, 3.0 # which_parts used for binning bars
		for name, these_betas in zip(['sound_event_kernel','sound_reward_kernel', 'zscored_beta_sound_reward'], #'colour_event_kernel','colour_reward_kernel',, 'zscored_beta_colour_reward'
									[ correct_beta_sound_event, correct_beta_sound_reward, z_correct_beta_sound_reward]): # correct_beta_colour_event, correct_beta_colour_reward,, z_correct_beta_colour_reward
			f = pl.figure(figsize=(9,6)) 
			ax1 = f.add_subplot(221)
			ax1.set_title(name + '\nprediction error trials \ncorrect runs')			
			prpe_median_start_end_data = [these_betas[PRPE_correct_runs][prpe_order][:int(PRPE_correct_runs.sum()/pe_sw)], these_betas[PRPE_correct_runs][prpe_order][int(pe_sw-1)*int(PRPE_correct_runs.sum()/pe_sw):]]
			nrpe_median_start_end_data = [these_betas[NRPE_correct_runs][nrpe_order][:int(NRPE_correct_runs.sum()/pe_sw)], these_betas[NRPE_correct_runs][nrpe_order][int(pe_sw-1)*int(NRPE_correct_runs.sum()/pe_sw):]]
			pl.bar([0,1], map(np.median, prpe_median_start_end_data), yerr = map(np.std, prpe_median_start_end_data)/np.sqrt(PRPE_correct_runs.sum()/pe_sw), color = 'r', width = 0.2, ecolor = 'k' )
			pl.bar([0.3,1.3], map(np.median, nrpe_median_start_end_data), yerr = map(np.std, nrpe_median_start_end_data)/np.sqrt(NRPE_correct_runs.sum()/pe_sw), color = 'b', width = 0.2, ecolor = 'k' )
			pl.legend(['prpe', 'nrpe'], loc = 'best')
			simpleaxis(ax1)
			spine_shift(ax1)
			pl.ylabel('beta values')
			pl.axhline(0, color = 'k', linewidth = 0.25)
			ax1.set_xticks([0.25, 1.25])
			ax1.set_xticklabels(['early', 'late'])

			ax2 = f.add_subplot(222, sharey=ax1)
			ax2.set_title(name + '\nno prediction error trials \ncorrect runs')
			hphr_median_start_end_data = [these_betas[HPHR_correct_runs][hphr_order][:int(HPHR_correct_runs.sum()/nope_sw)], these_betas[HPHR_correct_runs][hphr_order][int(nope_sw-1)*int(HPHR_correct_runs.sum()/nope_sw):]]			
			lplr_median_start_end_data = [these_betas[LPNR_correct_runs][lplr_order][:int(LPNR_correct_runs.sum()/nope_sw)], these_betas[LPNR_correct_runs][lplr_order][int(nope_sw-1)*int(LPNR_correct_runs.sum()/nope_sw):]]
			pl.bar([0,1], map(np.median, hphr_median_start_end_data), yerr = map(np.std, hphr_median_start_end_data)/np.sqrt(HPHR_correct_runs.sum()/nope_sw), color = 'r', width = 0.2, ecolor = 'k', alpha = 0.5 )
			pl.bar([0.3,1.3], map(np.median, lplr_median_start_end_data), yerr = map(np.std, lplr_median_start_end_data)/np.sqrt(LPNR_correct_runs.sum()/nope_sw), color = 'b', width = 0.2, ecolor = 'k' , alpha = 0.5)
			pl.legend(['hphr', 'lplr'], loc = 'best')			
			simpleaxis(ax2)
			spine_shift(ax2)
			pl.ylabel('beta values')	
			pl.axhline(0, color = 'k', linewidth = 0.25)
			ax2.set_xticks([0.25, 1.25])
			ax2.set_xticklabels(['early', 'late'])

			ax3 = f.add_subplot(223)
			ax3.set_title(name + '\nprediction error trials \ncorrect runs')
			pl.plot(pd.stats.moments.rolling_mean(these_betas[PRPE_correct_runs][prpe_order], 5), 'r',  label='prpe')
			pl.plot(pd.stats.moments.rolling_mean(these_betas[NRPE_correct_runs][nrpe_order], 5), 'b',  label='nrpe')
			pl.legend(['prpe', 'nrpe'])
			simpleaxis(ax3)
			spine_shift(ax3)
			pl.ylabel('beta values')
			pl.xlabel('ordered trial # ')

			ax4 = f.add_subplot(224, sharey=ax3)
			ax4.set_title(name + '\nno prediction error trials \ncorrect runs')
			pl.plot(pd.stats.moments.rolling_median(these_betas[HPHR_correct_runs][hphr_order], 5), 'r', alpha = 0.5, label='hphr')
			pl.plot(pd.stats.moments.rolling_median(these_betas[LPNR_correct_runs][lplr_order], 5), 'b', alpha = 0.5, label='lpnr')
			pl.legend(['hphr', 'lpnr'])
			simpleaxis(ax4)
			spine_shift(ax4)	 	
			pl.ylabel('beta values')
			pl.xlabel('ordered trial # ')
			pl.tight_layout()
			pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials + '_' + name + '_GLM_bar_line_correct_runs.pdf'))

			with pd.get_store(self.ho.inputObject) as h5_file:
				h5_file.put("/%s/%s"%(folder_name, 'prpe_median_start_end_data_'+name), pd.DataFrame(prpe_median_start_end_data))	
				h5_file.put("/%s/%s"%(folder_name, 'nrpe_median_start_end_data_'+name), pd.DataFrame(nrpe_median_start_end_data))	
				h5_file.put("/%s/%s"%(folder_name, 'hphr_median_start_end_data_'+name), pd.DataFrame(hphr_median_start_end_data))	
				h5_file.put("/%s/%s"%(folder_name, 'lplr_median_start_end_data_'+name), pd.DataFrame(lplr_median_start_end_data))	


	def individual_IRF(self, data_type = 'pupil_bp_zscore', analysis_sample_rate=25, subsample_ratio=5): 
		""" individual IRF uses function pupil_IRF to estimate whether using the IRF and its derivative explain more variance in pupil data than the original
		deconvolution of events without inpulse response function """

		self.events_and_signals_in_time_behav(data_type=data_type)
		 
		#IRF input: 
		IRF_len=3.0 #seconds
		timepoints = np.linspace(0, IRF_len, IRF_len * analysis_sample_rate)
		
		#calculation IRF and its derivative		
		IRF, IRF_prime = myfuncs.pupil_IRF(timepoints=timepoints,s=1.0/(10**26), n=10.1, tmax=0.93) 
		
		#time series info 
		input_signal = sp.signal.decimate(self.pupil_data, int(self.sample_rate / analysis_sample_rate))
		duration = np.array(np.round(len(input_signal)/analysis_sample_rate), dtype=int) # ~ 5000 seconds
		sound_sample_indices = np.array(np.round(self.sound_times * analysis_sample_rate), dtype=int)
		colour_sample_indices =  np.array(np.round(self.colour_times * analysis_sample_rate), dtype=int)

		#make regressor matrix
		total_nr_regressors = 4  #IRF & IRF_prime for colour and sound events 
		raw_design_matrix = np.zeros((total_nr_regressors,len(input_signal)))  
		convolved_design_matrix = np.zeros(raw_design_matrix.shape)
		
		#convolve events with IRF and IRF prime 
		for kernel, event_indices, index in zip([IRF, IRF_prime, IRF, IRF_prime], 
										[colour_sample_indices, colour_sample_indices, sound_sample_indices, sound_sample_indices],
										range(4)):			
			raw_design_matrix[index,event_indices] = 1;	
			convolved_design_matrix[index] = signal.fftconvolve(raw_design_matrix[index], kernel, 'full')[:convolved_design_matrix[index].shape[0]] # implicit padding here, done by indexing
			#demean regressors 
			convolved_design_matrix[index] = (convolved_design_matrix[index] - convolved_design_matrix[index].mean()) / convolved_design_matrix[index].std()

		#multiple regression on pupil data with convolved regressors
		X = convolved_design_matrix[:,::subsample_ratio].T #take every 5th element of convolved_design_matrix to speed up calculation
		model = sm.OLS(input_signal[::subsample_ratio], X)
		results = model.fit()
		self.logger.info(results.summary())
		results.save(os.path.join(self.base_directory, 'processed', self.subject.initials + '_individual_IRF.pickle'))

		beta_colour = np.mean((results.params[0], results.params[1]), axis=0) 
		beta_colour_no_dt = results.params[0]
		beta_sound =  np.mean((results.params[2], results.params[3]), axis=0) 
		beta_sound_no_dt = results.params[2]

		#plot original and weighted IRF kernel for colour and sound events
		f = pl.figure() 
		ax = f.add_subplot(211)
		pl.plot(timepoints, IRF, 'r') 
		pl.title('original IRF')
		pl.ylabel('au')
		pl.xlabel('time(s)')
		ax = f.add_subplot(212)
		pl.plot(timepoints,(IRF * beta_sound), 'g', timepoints,(IRF * beta_sound_no_dt), 'r', timepoints,(IRF * beta_colour), 'b', timepoints,(IRF * beta_colour_no_dt), 'y')
		pl.legend(['IRF_sound_dt $beta$:%1.3f'%beta_sound, 'IRF_sound_no_dt $beta$:%1.3f'%beta_sound_no_dt, 'IRF_colour_dt $beta$:%1.3f'%beta_colour, 'IRF_colour_no_dt $beta$:%1.3f'%beta_colour_no_dt])
		pl.title('Individual IRF of colour and sound events after multiple regression')
		pl.ylabel('au')
		pl.xlabel('time(s)')
		pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials +'_individual_irf.pdf'))


	def event_related_average(self, data_type='pupil_bp_zscore', analysis_sample_rate=10, 
							 use_domain='full', fixation_duration=0.5, interesting_dur = 3.0): 
		"""calculate the average event response per participant for events of the type: colour, sound, 
		cue low, cue high, reward low, high , hphr, hpnr, lphr,  lpnr for different time domains """ 
		
		self.events_and_signals_in_time(data_type=data_type)

		interesting_period = int(interesting_dur * analysis_sample_rate)
		time_points = np.linspace(-interesting_period / analysis_sample_rate, interesting_period / analysis_sample_rate, interesting_period*2)
		
		#import cleaned up pupil signal 
		with pd.get_store(self.ho.inputObject) as h5_file:
			try:
				input_signal = np.array(h5_file.get("/%s/%s"%('deconvolve_nuisance_%s_Hz'%str(analysis_sample_rate), 'residuals')))				
			except (IOError, NoSuchNodeError):
				self.logger.error("no residuals present")

		#get domain indices
		domain_indices_used_now = self.select_domain_indices(use_domain=use_domain)
		
		cue_indices =    [
			self.reward_prob_indices[0] * domain_indices_used_now, 	#cue Low
			self.reward_prob_indices[1] * domain_indices_used_now,	#cue High
		]
		cue_times = [self.colour_times[ci] for ci in cue_indices]

		reward_indices =    [
			self.sound_indices[0] * domain_indices_used_now, 	#loss
			self.sound_indices[1] * domain_indices_used_now,	#win 
		]
		reward_times = [self.sound_times[ri] for ri in reward_indices]

		reward_event_indices = [
			self.reward_prob_indices[0] * self.sound_indices[0] * domain_indices_used_now,  #LP NR  
			self.reward_prob_indices[0] * self.sound_indices[1] * domain_indices_used_now,  #LP HR  
			self.reward_prob_indices[1] * self.sound_indices[0] * domain_indices_used_now,  #HP LR  
			self.reward_prob_indices[1] * self.sound_indices[1] * domain_indices_used_now,  #HP HR 			
		]	
		reward_event_times = [self.sound_times[ri] for ri in reward_event_indices]
		
		
		#baseline correct all cue and sound events using mean_pupil_at_fixation 
		fix_samples = int(fixation_duration * analysis_sample_rate)
		fix_start_idx = (self.fix_times * analysis_sample_rate).astype(int)
		mean_pupil_at_fixation = np.mean([input_signal[fix:fix+fix_samples] for fix in fix_start_idx], axis=1)
		 
		#event start indices 
		colour_start_idx = (self.colour_times * analysis_sample_rate).astype(int)
		sound_start_idx = (self.sound_times * analysis_sample_rate).astype(int)
		cue_l_h_start_idx = [(cue_times[type] * analysis_sample_rate).astype(int) for type in range(len(cue_times))]
		reward_l_h_start_idx = [(reward_times[type] * analysis_sample_rate).astype(int) for type in range(len(reward_times))]
		reward_event_start_idx = [(reward_event_times[type] * analysis_sample_rate).astype(int) for type in range(len(reward_event_times))]
		
		labels = ['colour', 'sound', 'cue_low', 'cue_high', 'sound_low', 'sound_high', 'LPNR', 'LPHR', 'HPNR', 'HPHR']
		#select epochs (first and last trial are removed)
		epoched_data = [] 
		mean_pupil_event_responses = []
		for event in [colour_start_idx, sound_start_idx] + cue_l_h_start_idx + reward_l_h_start_idx + reward_event_start_idx: 
			event_epochs = np.array([input_signal[x-interesting_period:x+interesting_period]-i for i, x in zip(mean_pupil_at_fixation[1:-1], event[1:-1])])
			mean_event = np.mean(event_epochs, axis=0)
			epoched_data.append(event_epochs)
			mean_pupil_event_responses.append(mean_event)
		epoched_data=np.array(epoched_data)
		mean_pupil_event_responses=np.array(mean_pupil_event_responses)

		folder_name = 'epoched_data_%s_domain_%s_Hz'%(use_domain, analysis_sample_rate)
		with pd.get_store(self.ho.inputObject) as h5_file:
			h5_file.put("/%s/%s"%(folder_name, 'time_points'), pd.Series(np.array(time_points)))
			h5_file.put("/%s/%s"%(folder_name, 'epoched_pupil_data'), pd.DataFrame(np.squeeze(epoched_data)))
			h5_file.put("/%s/%s"%(folder_name, 'mean_pupil_event_responses'), pd.DataFrame(np.squeeze(mean_pupil_event_responses)))
			h5_file.put("/%s/%s"%(folder_name, 'labels'), pd.Series(np.array(labels)))


	def extract_event_epochs(self, event_times, analysis_sample_rate, input_signal, baseline_correction, t_before=0.5, t_after=4.0): 
		""" input event start times and interval of interest to select pupil values of interest  """
		#convert list of events to numpy nd array 
		event_times = np.array(event_times)

		#number of samples in a chunk 
		nsamples = int(t_before + t_after) * analysis_sample_rate
		sample_idx = np.arange(nsamples)

		#get index of the first sample, subtract t_before and convert from seconds to samples		
		event_start_indices = [((event_times[event] - t_before)*analysis_sample_rate).astype(int) for event in range(len(event_times))]
		
		#Extract event epochs of [t_before, t_after] from input signal. If the epoch extends input signal, 
		#take last sample of input signal as the last index. 
		epoched_events=[]
		for event_type in event_start_indices: 
			nr_trials = len(event_type)    
			empty_data_array = np.ones((nr_trials, nsamples)) * np.nan #trials x samples 
			for i in np.arange(nr_trials):
				final_sample = np.min([(event_type[i] + nsamples), input_signal.shape[0]]) #calculate the shortest sample for each trial, use this as sample to calculate trial dur 
				trial_dur = final_sample - event_type[i] #end timepoint - begin timepoint for duration of trial 
				empty_data_array[i,0:trial_dur] = input_signal[event_type[i]:event_type[i]+trial_dur]
			epoched_events.append(empty_data_array)
		epoched_events = np.array(epoched_events)
		

 
	def TD_states(self, data_type = 'pupil_bp_zscore', do_zoom='zoom', do_sim=False, do_plot=False, best_sim_params='average_TD_params'): 
		"""TD_behaviour takes state and reward information and calculates two state values for every trial using TD learning (TD.py). From the state values, a difference value score 
		of the two states is calculated. Whenever the difference value score flips sign, this can be interpreted as a reversal in state values according tho the model """

		self.events_and_signals_in_time_TD(data_type=data_type)
		
		#set parameter values 
		if do_zoom == 'no_zoom': 
			alphas = np.linspace(0.0,1.0,30)
			gammas = np.linspace(0.0,1.0,30)
			lambdas = np.linspace(0.0,1.0,30)
		elif do_zoom == 'zoom': 
			alphas = np.linspace(0.0, 0.3, 30)
			gammas = np.linspace(0.9, 1.0, 30)
			lambdas = np.linspace(0.7, 1.0, 30)		

		#get event information 
		trials_per_run = np.split(self.trial_indices, np.where(self.trial_indices == 0.)[0][1:])
		self.last_trial_per_run = np.array([np.max(t_p_b) for t_p_b in trials_per_run]).astype(int)		
		state_green = self.hue_indices[0]
		state_purple = self.hue_indices[1]
		states = self.hue_indices[0].astype(int) 	#0 is green, 1 is purple		
		rewards = self.sound_indices[1].astype(int) #0 is loss, 1 is reward 

		self.rewards_per_run = [rewards[i[0]:i[1]] for i in self.run_trial_limits] #rewards 0 and 1 		
		self.states_per_run = [states[i[0]:i[1]] for i in self.run_trial_limits]
		self.green_per_block = [state_green[i[0]:i[1]] for i in self.run_trial_limits]
		self.purple_per_block = [state_purple[i[0]:i[1]] for i in self.run_trial_limits]
		parameters = [alphas, gammas, lambdas]
		parameter_combi = list(itertools.product(*parameters))
		 
		## get participant's reversal_keypresses 
		with pd.get_store(self.ho.inputObject) as h5_file:
			try:
				self.reversal_keypresses = h5_file.get("/%s/%s"%('TD', 'reversal_keypresses')).as_matrix()					
			except (IOError, NoSuchNodeError):
				self.logger.error("no reversal_keypresses present")

		#### START TD SIMULATIONS ####
		#initialise DelayCalculator and get keypress distances 
		td_delay = DelayCalculator(reversal_positions=self.reversal_positions, run_trial_limits=self.run_trial_limits, reversal_keypresses=self.reversal_keypresses)
		self.behav_distance = np.concatenate(td_delay.calculate_reversal_distance(bps = self.reversal_keypresses)) 
		
		#initialise TD_runner and run simulations
		td_runner = TD_runner(states_per_run=self.states_per_run, rewards_per_run=self.rewards_per_run, reversal_positions=self.reversal_positions, 
								run_trial_limits=self.run_trial_limits, behav_distance=self.behav_distance, reversal_keypresses=self.reversal_keypresses) 		
		
		if do_sim: #if true, simulations are run
			self.logger.info('starting TD simulations for participant %s. Zooming in on on parameter values: %s ' %(self.subject.initials, do_zoom))
			td_runner.simulate_TD_for_loop(alphas=alphas, gammas=gammas, lambdas=lambdas, init_val= 5.0)

		#save results of each TD_runner simulation 
		sim_dist_file = os.path.join(os.path.split(self.ho.inputObject)[0], 'TD_model_behav_distance_%s.npz'%do_zoom)  #name of new simulation where model takes reversal keypresses instead of reversal points as distance criterion
		
		if do_sim:
			td_runner.save_simulation_results(filename=sim_dist_file)
			self.logger.info('TD simulation results for participant %s are being saved as: %s ' %(self.subject.initials, sim_dist_file))
	
		
		#### LOAD TD SIMULATIONS ####
		data = np.load(sim_dist_file)
		sim_dist = data['arr_0'].astype('float32')#array containing alpha,gamma,lambda,summed distances, summed reversals for each parameter combination 		
		summed_ds = sim_dist[:,:,:,-1] #summed distances 
		block_reversals = sim_dist[:,:,:,:-1] #block reversals for all parameter combinations
		 
		#check if model simulations have at least one reversal per block.
		check_block_reversals=[]
		for a, alpha in enumerate(alphas): 
			for g, gamma in enumerate(gammas): 
				for l, lamb in enumerate(lambdas): 
					check = all(x > 0 for x in block_reversals[a,g,l,0:len(self.reversal_positions)])
					check_block_reversals.append(check) #evaluates to True if all blocks have at least 1 reversal

		#find the index and value of parameter combination with the smallest model-behaviour distance AND at least one reversal per block		
		smallest_distance, smallest_idx = td_delay.find_smallest_distance(summed_distance=summed_ds, check_block_reversals=check_block_reversals)
		best_param_combi = parameter_combi[smallest_idx]
		 
		## save participant and model reversal distances in HDF5
		with pd.get_store(self.ho.inputObject) as h5_file:
			h5_file.put("/%s/%s"%('TD', 'behav_distance_%s'%do_zoom), pd.Series(np.array(self.behav_distance)))
			h5_file.put("/%s/%s"%('TD', 'parameter_combinations_%s'%do_zoom), pd.DataFrame(np.array(parameter_combi), columns=['alpha', 'gamma', 'lambda']))
			h5_file.put("/%s/%s"%('TD', 'best_parameter_combination_%s'%do_zoom), pd.Series(np.array(best_param_combi)))
			h5_file.put("/%s/%s"%('TD', 'best_TD_fit_values_%s'%do_zoom), pd.Series(np.array([smallest_distance, smallest_idx])))

		
		average_TD_params = np.load(os.path.join(os.path.split(self.base_directory)[0], 'group_level/data/av_best_param_combi.npy')) 
		
		## optimal state values and prediction erros per trial 
		if best_sim_params == 'individual_TD_params': 
			self.logger.info('Calculating TD state values and prediction error using individual TD parameter values for participant %s. Zoom: %s ' %(self.subject.initials, do_zoom))
			self.best_statevalues, self.raw_signed_prediction_error = td_runner.simulate_best_TD_timecourse(alpha=best_param_combi[0], gamma=best_param_combi[1], lamb=best_param_combi[2], init_val=5.0)
			#save state values and RPE with individual TD parameter values
			with pd.get_store(self.ho.inputObject) as h5_file:
				h5_file.put("/%s/%s"%('TD', 'individual_TD_params_statevalues_%s'%do_zoom), pd.DataFrame(np.array(self.best_statevalues)))
				h5_file.put("/%s/%s"%('TD', 'individual_TD_params_raw_signed_prediction_error_%s'%do_zoom), pd.DataFrame(np.array(self.raw_signed_prediction_error)))

		elif best_sim_params == 'average_TD_params': 
			self.logger.info('Calculating TD state values and prediction error using average TD parameter values for participant %s. Zoom: %s ' %(self.subject.initials, do_zoom))
			self.best_statevalues, self.raw_signed_prediction_error = td_runner.simulate_best_TD_timecourse(alpha=average_TD_params[0], gamma=average_TD_params[1], lamb=average_TD_params[2], init_val=5.0)
			#save state values and RPE with individual TD parameter values
			with pd.get_store(self.ho.inputObject) as h5_file:
				h5_file.put("/%s/%s"%('TD', 'average_TD_params_statevalues_%s'%do_zoom), pd.DataFrame(np.array(self.best_statevalues)))
				h5_file.put("/%s/%s"%('TD', 'average_TD_params_raw_signed_prediction_error_%s'%do_zoom), pd.DataFrame(np.array(self.raw_signed_prediction_error)))
		else: 
			self.logger.info('No TD params found for participant %s ' %self.subject.initials)


		## TD z-scored signed and unsigned prediction error regressors
		self.signed_prediction_error = np.copy(self.raw_signed_prediction_error)
		self.abs_prediction_error = np.abs(self.signed_prediction_error)		
		self.z_scored_signed_prediction_error = (self.signed_prediction_error - np.mean(self.signed_prediction_error, axis=0))/self.signed_prediction_error.std()
		self.z_scored_unsigned_prediction_error= (self.abs_prediction_error - np.mean(self.abs_prediction_error, axis=0))/self.abs_prediction_error.std()
		#cor_regressors = sp.stats.spearmanr(a=self.signed_prediction_error, b=self.abs_prediction_error)
				 
		green_statevalues = [self.best_statevalues[:,0][i[0]:i[1]] for i in self.run_trial_limits]
		purple_statevalues = [self.best_statevalues[:,1][i[0]:i[1]] for i in self.run_trial_limits]
		diff_statevalues = [self.best_statevalues[:,2][i[0]:i[1]] for i in self.run_trial_limits] 
		best_modelreversals = td_delay.calculate_model_reversals(state_values=self.best_statevalues, trial_nr=10) 
		best_modeldistance = np.concatenate(td_delay.calculate_model_distance(mps=best_modelreversals))

		
		if do_plot==True: 

			# f = pl.figure() 
			# ax = f.add_subplot(211)
			# pl.hist(self.signed_prediction_error, alpha=0.5)
			# pl.hist(self.abs_prediction_error, alpha=0.5)
			# pl.title('raw signed and unsigned RPE')
			# pl.legend(['signed RPE', 'unsigned RPE'])
			# ax = f.add_subplot(212)
			# pl.hist(self.z_scored_signed_prediction_error, alpha=0.5)
			# pl.hist(self.z_scored_unsigned_prediction_error, alpha=0.5)
			# pl.title('zscored signed and unsigned RPE')
			# pl.legend(['signed RPE', 'unsigned RPE'])
			# pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials +'_signed_unsigned_histogram.pdf'))


			## raw and demeaned prediction error
			# f = pl.figure(figsize=(12,5))
			# ax = f.add_subplot(211) 
			# pl.plot(self.raw_signed_prediction_error[:100], 'r', self.z_scored_signed_prediction_error[:100], 'b', alpha=0.5)
			# pl.title('Prediction error on every trial')
			# pl.legend(['raw signed prediction error', 'z-scored signed prediction_error'])
			# pl.ylabel('prediction error')
			# ax = f.add_subplot(212)
			# pl.plot(self.z_scored_unsigned_prediction_error[:100], 'g', alpha=0.5)
			# pl.legend(['z-scored unsigned prediction error'])
			# pl.xlabel('trial #')
			# pl.ylabel('prediction error')
			# pl.tight_layout()
			# pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials +'_prediction_error_%s.pdf'%do_zoom))

			# #histogram of all pvalues --> check to see how p-values are distributed and adjust the plot accordingly 
			# f = pl.figure()
			# pl.hist(summed_ds.flatten(), 1000, fc='k', ec='k') 
			# ax = f.add_subplot(111)
			# pl.title('histogram of distance between participant and model reversals')
			# pl.xlabel('distance')
			# pl.ylabel('frequency')
			# pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials +'_summed_distance_histogram_%s.pdf'%do_zoom))

			# #clip the simulation to the minimum and maximum distance values  
			# flattened_max, flattened_min = np.nanmax(summed_ds.flatten()), np.min(summed_ds.flatten()[np.nonzero(summed_ds.flatten())])
			# simulation_to_plot = np.linspace(0,29,9).astype(int)
			
			# #simulation subplots
			# fig, axes = pl.subplots(nrows=3, ncols=3, sharey=True, sharex=True, figsize = (10,7))		
			# for i, ax in enumerate(axes.flat):
			# 	im = ax.imshow(summed_ds[:,:,simulation_to_plot[i]], cmap='YlGnBu')
			# 	ax.set_title('lambda:%s'%simulation_to_plot[i],fontsize=7)				
			# im.set_clim(flattened_min, flattened_max)
			# im.set_interpolation('bicubic')		
			# fig.text(0.5, 0.02, 'alpha', ha='center', va='center', fontsize='medium')
			# fig.text(0.08, 0.5, 'gamma', ha='center', va='center', rotation='vertical', fontsize='medium')
			# fig.text(0.5, 0.98, 'Distance between model and participant reversals for each TD simulation parameter combination %s'%do_zoom, ha='center', va='center', fontsize='large')
			# #color bar 
			# fig.subplots_adjust(right=0.8)
			# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
			# fig.text(0.88, 0.88, 'summed distance', ha='center', va='center', fontsize='medium')
			# fig.colorbar(im, cax=cbar_ax)
			# pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials +'_model_behaviour_distance_imshow_%s.pdf'%do_zoom))

			vert_offset=1
			button_press_trial = [np.zeros(len(run))*np.nan for run in trials_per_run] 
			model_press_trial = [np.zeros(len(run))*np.nan for run in trials_per_run]
			dts_per_run = [self.domain_time[i[0]:i[1]] for i in self.run_trial_limits] 
			for i in range(len(button_press_trial)): 
				button_press_trial[i][self.reversal_keypresses[i]] = 1.1 
				model_press_trial[i][best_modelreversals[i]] = 1.3

			# #adjust amount of plots according to length of blocks 
			if len(trials_per_run) == 7: 
				nrows, ncols = 7,1
			elif len(trials_per_run) == 8: 
				nrows, ncols = 4,2
			elif len(trials_per_run) ==9: 
				nrows, ncols = 3,3 
			elif len(trials_per_run) == 10: 			
				nrows, ncols = 5,2
			
			fig, axes = pl.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, figsize = (12,7))		
			for i, ax in enumerate(axes.flat):			
				ax.set_title('run %s'%str(i+1), {'fontsize':8})
				ax.plot(trials_per_run[i], diff_statevalues[i], alpha = 0.8 )
				ax.plot(np.arange(len(trials_per_run[i]))[self.rewards_per_run[i] * self.green_per_block[i] == 1], (self.rewards_per_run[i] * self.green_per_block[i])[self.rewards_per_run[i] * self.green_per_block[i] == 1] - vert_offset, 'go', alpha = 0.25)
				ax.plot(np.arange(len(trials_per_run[i]))[self.rewards_per_run[i] * self.purple_per_block[i] == 1], (self.rewards_per_run[i] * self.purple_per_block[i])[self.rewards_per_run[i] * self.purple_per_block[i] == 1] - vert_offset, 'mo', alpha = 0.25)
				ax.plot(button_press_trial[i], 'k*', lw=12, alpha=0.7) 
				ax.plot(model_press_trial[i], 'y*', lw=12, alpha=0.7) 
				ax.plot(dts_per_run[i], 'r', alpha=0.4)
				ax.axhline(0, lw=0.25, alpha=0.8, color = 'k')
				#ax.set_ylim(-1.5,1.5)
			fig.text(0.5, 0.02, 'trial #', ha='center', va='center', fontsize='medium')
			fig.text(0.08, 0.5, 'state value', ha='center', va='center', rotation='vertical', fontsize='medium')
			fig.text(0.5, 0.98, 'Best TD model simulation', ha='center', va='center', fontsize='large')
			pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials +'_TD_timecourse_smallest_distance_%s_%s.pdf'%(do_zoom, best_sim_params)))
			 
			# ## plot individual responses and best fitting TD simulation  
		# 	behav_pdf = stats.kde.gaussian_kde(self.behav_distance) 
		# 	model_pdf = stats.kde.gaussian_kde(best_modeldistance)
		# 	f = pl.figure()
		# 	x = np.linspace(-50,50,100)
		# 	best_param_string = ' '.join(map(str, (best_param_combi)))
		# 	ax = f.add_subplot(111)
		# 	pl.plot(x, behav_pdf(x),'royalblue', lw=2.0, alpha=0.7) # distribution function
		# 	pl.plot(x, model_pdf(x), 'indianred', lw=2.0, alpha=0.7)
		# 	pl.hist(self.behav_distance,normed=1,alpha=.3, color='royalblue') #response histogram
		# 	pl.hist(best_modeldistance, normed=1, alpha=.3, color='indianred')	
		# 	pl.legend(['participant responses', 'model responses'])
		# 	pl.text(15, 0.055,'smallest summed distance: %s'%smallest_distance, horizontalalignment='left', verticalalignment='baseline', fontsize=8)
		# 	pl.text(15, 0.045, 'parameter combi: %s'%best_param_string, horizontalalignment='left', verticalalignment='baseline', fontsize=8)
		# 	pl.title('Reversal response times of participant and best fitting model simulation')
		# 	ax.set_xlim([-50,50])
		# 	pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials +'_TD_model_smallest_distance_%s.pdf'%do_zoom))
		# else: 
		# 	pass 

	def regress_TD_prediction_error(self, data_type = 'pupil_bp_zscore', analysis_sample_rate=25, t_before=0.5, t_after=3.5, fix_dur=0.5, standard_deconvolution='sound', microsaccades_added=False, 
									do_zscore='z_scored'): 
		"""deconvolve_with_TD_prediction_error takes pupil signals and prediction errors acquired from TD_states to evaluate whether pupil timecourses 
		can be explained by the model prediction errors. It evalueates both signed and unsigned prediction error signals using a glm approach. """ 
		
		self.events_and_signals_in_time_TD(data_type=data_type)
		self.TD_states(data_type=data_type)

		if standard_deconvolution == 'sound': #residual signal sound response deconvolved 
			data_folder='standard_deconvolve_keypress_%s'%str(microsaccades_added)
		elif standard_deconvolution == 'no_sound': #residual signal sound response not deconvolved
			data_folder='standard_deconvolve_keypress_no_sound'

		#get z-scored or raw prediction error signals 
		if do_zscore == 'z_scored': 
			signed_prediction_error = self.z_scored_signed_prediction_error
			abs_prediction_error = self.z_scored_unsigned_prediction_error
		elif do_zscore == 'raw': 
			signed_prediction_error = self.signed_prediction_error
			abs_prediction_error = self.abs_prediction_error

		#get residuals from standard deconvolution 
		with pd.get_store(self.ho.inputObject) as h5_file:
			try:
				residuals_standard_deconvolve_keypress = h5_file.get("/%s/%s"%(data_folder, 'residuals'))					
			except (IOError, NoSuchNodeError):
				self.logger.error("no residuals present")


		pupil_sound_sample_limits=[] 
		trial_regressors_s=[]
		trial_regressors_us=[]
		
		#get event start samples 
		#sound_start_idx = np.around(self.sound_times*analysis_sample_rate).astype(int) #sound start times in analysis sampling rate samples
		  
		sound_start_idx = ((self.sound_times - t_before) * analysis_sample_rate).astype(int) #0.5 sec before sound onset
		fix_start_idx = np.around(self.fix_times*analysis_sample_rate).astype(int) 
		regress_period = int(t_before + t_after) * analysis_sample_rate #interval [-0.5 - 3.5s], 100 samples 
		fix_period = int(fix_dur * analysis_sample_rate)
		pupil_sound_samples = np.ones((len(self.sound_times), regress_period))*np.nan
		# pupil_sound_samples_demeaned = np.ones((len(self.sound_times), regress_period))*np.nan
		# pupil_sound_samples_demeaned_full = np.ones((len(self.sound_times), regress_period))*np.nan
		# pupil_sound_samples_full = np.ones((len(self.sound_times), regress_period))*np.nan


		#get pupil signals 
		input_signal = np.array(residuals_standard_deconvolve_keypress)	#residual pupil signal 
		#input_signal_full = np.array(sp.signal.decimate(self.pupil_data, int(self.sample_rate / analysis_sample_rate)))
		ds_pupil_baseline = sp.signal.decimate(self.pupil_baseline_data, int(self.sample_rate / analysis_sample_rate)) #downsampled pupil_baseline_zscore signal		
		pupil_baseline_values_fix = np.array([ds_pupil_baseline[fix:fix+fix_period].mean(axis=0) for fix in fix_start_idx])
		pupil_baseline_values_presound = np.array([ds_pupil_baseline[sound-fix_period:sound].mean() for sound in sound_start_idx])
		
		#get regressors and trial samples 
		for i in range(len(self.sound_times)): #loop over trials 
			pupil_sound_sample_limits.append([sound_start_idx[i], sound_start_idx[i] + regress_period]) 
			final_sample = np.min([sound_start_idx[i] + regress_period, input_signal.shape[0]])
			trial_dur = final_sample - sound_start_idx[i] #end sample - begin sample 
			pupil_sound_samples[i,0:trial_dur] = (input_signal[sound_start_idx[i]:sound_start_idx[i]+trial_dur])#-pupil_baseline_values_fix[i]
			# pupil_sound_samples_demeaned[i,0:trial_dur] = (input_signal[sound_start_idx[i]:sound_start_idx[i]+trial_dur])-pupil_baseline_values_fix[i]
			# pupil_sound_samples_demeaned_full[i,0:trial_dur] = (input_signal_full[sound_start_idx[i]:sound_start_idx[i]+trial_dur])-pupil_baseline_values_fix[i]
			# pupil_sound_samples_full[i,0:trial_dur] = (input_signal_full[sound_start_idx[i]:sound_start_idx[i]+trial_dur])

			#trial_regressors.append([1, self.signed_prediction_error[i], self.abs_prediction_error[i], pupil_baseline_values[i]]) #constant, signed, unsigned, baseline  	
			trial_regressors_s.append([1, signed_prediction_error[i], pupil_baseline_values_fix[i]])   	
			trial_regressors_us.append([1, abs_prediction_error[i], pupil_baseline_values_fix[i]]) 

		trial_regressors_s=np.array(trial_regressors_s)	
		trial_regressors_us=np.array(trial_regressors_us)

		#linear regression signed prediction error 
		clf_s = linear_model.LinearRegression(fit_intercept=False)
		clf_s.fit(trial_regressors_s[:-1], pupil_sound_samples[:-1]) 				#fit linear model 
		regres_coeff_s = clf_s.coef_ 								   				#get beta coefficients 
		prediction_s = clf_s.predict(trial_regressors_s[:-1])	   				    #predict pupil values using linear model regressors 
		u = ((pupil_sound_samples[:-1] - prediction_s)**2).sum()	   				#regression sum of squares 
		v = ((pupil_sound_samples[:-1] - pupil_sound_samples[:-1].mean())**2).sum() #residual sum of squares 
		r_squared_s = (1 - u/v)

		#linear regression unsigned prediction error 
		clf_us = linear_model.LinearRegression(fit_intercept=False)
		clf_us.fit(trial_regressors_us[:-1], pupil_sound_samples[:-1]) 				#fit linear model 
		regres_coeff_us = clf_us.coef_ 								   				#get beta coefficients 
		prediction_us = clf_us.predict(trial_regressors_us[:-1])	   				#predict pupil values using linear model regressors 
		u = ((pupil_sound_samples[:-1] - prediction_us)**2).sum()	   				#regression sum of squares 
		v = ((pupil_sound_samples[:-1] - pupil_sound_samples[:-1].mean())**2).sum() #residual sum of squares 
		r_squared_us = (1 - u/v)

		with pd.get_store(self.ho.inputObject) as h5_file:
			h5_file.put("/%s/%s"%('TD_regression', 'regres_coeff_us_%s_%s'%(standard_deconvolution, do_zscore)), pd.DataFrame(np.array(regres_coeff_us)))
			h5_file.put("/%s/%s"%('TD_regression', 'regres_coeff_s_%s_%s'%(standard_deconvolution, do_zscore)), pd.DataFrame(np.array(regres_coeff_s)))
			h5_file.put("/%s/%s"%('TD_regression', 'rsquared_us_%s_%s'%(standard_deconvolution, do_zscore)), pd.Series(r_squared_us))
			h5_file.put("/%s/%s"%('TD_regression', 'rsquared_s_%s_%s'%(standard_deconvolution, do_zscore)), pd.Series(r_squared_s))

		
		# # #plot pupil timecourse and estimated timecourse with linear model 
		# f = pl.figure()
		# pl.plot(np.mean(pupil_sound_samples[:-1], axis=1), 'r', np.mean(prediction_s, axis=1), 'b', alpha=0.5)
		# pl.legend(['per trial pupil signal', 'per trial pupil prediction']) 
		# pl.text(550, -3, 'rsquared: %s'%r_squared_s, fontsize=10)
		# pl.title('Pupil timecourse and estimated timecourse using linear model (signed RPE)')
		# pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials +'_pupil_timecoruse_linear_regression_rpe.pdf'))

		# # ##plot results 
		timepoints = np.linspace(-0.5, (t_before + t_after), regress_period)
		# f = pl.figure() 
		# ax = f.add_subplot(121)
		# pl.plot(timepoints, regres_coeff_us[:,0], 'k', timepoints, regres_coeff_us[:,1], 'b', timepoints, regres_coeff_us[:,2], 'r')
		# pl.legend(['constant', 'unsigned prediction error', 'pupil baseline'])
		# pl.xlabel('time (s)')
		# pl.ylabel('beta value')
		# pl.axvline(0, lw=0.25, alpha=0.5, color='k')
		# pl.axhline(0, lw=0.25, alpha=0.5, color='k')
		# pl.title('Linear regression of TD model values on pupil sound response')
		# ax = f.add_subplot(122)
		# pl.plot(timepoints, regres_coeff_s[:,0], 'k', timepoints, regres_coeff_s[:,1], 'b', timepoints, regres_coeff_s[:,2], 'r')
		# pl.legend(['constant', 'signed prediction error', 'pupil baseline'])
		# pl.axvline(0, lw=0.25, alpha=0.5, color='k')
		# pl.axhline(0, lw=0.25, alpha=0.5, color='k')
		# # pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials +'_TD_linear_regression_raw.pdf'))


