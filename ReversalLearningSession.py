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
matplotlib.use('Agg') 
import matplotlib.pyplot as pl
import pandas as pd
import numpy.linalg as LA
import bottleneck as bn
import glob
import seaborn as sn
import scipy.signal as signal
import sympy

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy import optimize, polyval, polyfit
from scipy.linalg import sqrtm, inv
#from scipy.signal import fftconvolve
from tables import NoSuchNodeError
import matplotlib.lines as mlines
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn import linear_model

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
from Tools.other_scripts import savitzky_golay as savitzky_golay

from fir import FIRDeconvolution

from IPython import embed as shell

class ReversalLearningSession(object):
	"""ReversalLearningSession"""
	def __init__(self, subject, experiment_name, project_directory, version, aliases, pupil_hp, loggingLevel = logging.DEBUG):
		self.subject = subject
		self.experiment_name = experiment_name
		self.aliases = aliases
		self.version = version
		self.pupil_hp = pupil_hp 

		try:
			os.mkdir(os.path.join( project_directory, experiment_name ))
			os.mkdir(os.path.join( project_directory, experiment_name, self.subject.initials ))
		except OSError:
			pass
		self.project_directory = project_directory
		self.base_directory = os.path.join( self.project_directory, self.experiment_name, self.subject.initials )
		
		self.create_folder_hierarchy()
		self.hdf5_filename = os.path.join(self.base_directory, 'processed', self.subject.initials + '.hdf5')
		self.ho = HDFEyeOperator.HDFEyeOperator(self.hdf5_filename)
		
		# add logging for this session
		# sessions create their own logging file handler
		self.loggingLevel = loggingLevel
		self.logger = logging.getLogger( self.__class__.__name__ )
		self.logger.setLevel(self.loggingLevel)
		addLoggingHandler( logging.handlers.TimedRotatingFileHandler( os.path.join(self.base_directory, 'log', 'sessionLogFile.log'), when = 'H', delay = 2, backupCount = 10), loggingLevel = self.loggingLevel )
		loggingLevelSetup()
		for handler in logging_handlers:
			self.logger.addHandler(handler)
		self.logger.info('starting analysis in ' + self.base_directory)
	
	def create_folder_hierarchy(self):
		"""createFolderHierarchy does... guess what."""
		this_dir = self.project_directory
		for d in [self.experiment_name, self.subject.initials]:
			try:
				this_dir = os.path.join(this_dir, d)
				os.mkdir(this_dir)
			except OSError:
				pass
		for p in ['raw','processed','figs','log']:
			try:
				os.mkdir(os.path.join(self.base_directory, p))
			except OSError:
				pass
	
	def import_raw_data(self, edf_files, aliases):
		"""import_raw_data loops across edf_files and their respective aliases and copies and renames them into the raw directory."""
		for edf_file, alias in zip(edf_files, aliases):
			self.logger.info('importing file ' + edf_file + ' as ' + alias)
			ExecCommandLine('cp "' + edf_file + '" "' + os.path.join(self.base_directory, 'raw', alias + '.edf"') )

	def import_msg_data(self, aliases):
		"""import_all_data loops across the aliases of the sessions and converts the respective edf files, adds them to the self.ho's hdf5 file. """
		for alias in aliases:
			self.ho.add_edf_file(os.path.join(self.base_directory, 'raw', alias + '.edf'))
			self.ho.edf_message_data_to_hdf(alias = alias)

	def import_gaze_data(self, aliases):
		"""import_all_data loops across the aliases of the sessions and converts the respective edf files, adds them to the self.ho's hdf5 file. """
		for alias in aliases:
			self.ho.add_edf_file(os.path.join(self.base_directory, 'raw', alias + '.edf'))			 
			self.ho.edf_gaze_data_to_hdf(alias = alias, 
										 pupil_hp = self.pupil_hp, 
										 pupil_lp = 4, 
										 maximal_frequency_filterbank = 0.05, 
										 minimal_frequency_filterbank = 0.002, 
										 nr_freq_bins_filterbank=20, 
										 tf_decomposition_filterbank='lp_butterworth')		
	
	def baseline_scalars(self, timeseries, event_times):
		data_in_sensible_range = (np.arange(len(timeseries))>event_times[0]-500)
		baseline = savitzky_golay.savitzky_golay(timeseries[data_in_sensible_range], 100001, 3)
			
		baseline_full = np.zeros(len(timeseries))
		baseline_full[data_in_sensible_range] = baseline 
		
		baseline_scalars = np.array([np.mean(baseline_full[event-500:event]) for event in event_times]) 
		
		return baseline_scalars

		
	def remove_HDF5(self):
		os.system('rm ' + os.path.join(self.base_directory, 'processed', self.subject.initials + '.hdf5') )
		

		
	def prepocessing_report(self, eye = 'L', downsample_rate=20):
		for alias in self.aliases:
			# load times per session:
			trial_times = self.ho.read_session_data(alias, 'trials')
			trial_phase_times = self.ho.read_session_data(alias, 'trial_phases')
			session_start_EL_time = np.array(trial_times['trial_start_EL_timestamp'])[0]
			session_stop_EL_time = np.array(trial_times['trial_end_EL_timestamp'])[-1]

			sample_rate = self.ho.sample_rate_during_period([session_start_EL_time, session_stop_EL_time], alias)

			pupil_raw = np.squeeze(self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], alias = alias, signal = 'pupil', requested_eye = eye))
			pupil_int = np.squeeze(self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], alias = alias, signal = 'pupil_int', requested_eye = eye))

			pupil_bp = np.squeeze(self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], alias = alias, signal = 'pupil_bp', requested_eye = eye))
			pupil_lp = np.squeeze(self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], alias = alias, signal = 'pupil_lp', requested_eye = eye))
			pupil_hp = np.squeeze(self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], alias = alias, signal = 'pupil_hp', requested_eye = eye))

			x = sp.signal.decimate(np.arange(len(pupil_raw)) / float(sample_rate), downsample_rate, 1)
			pup_raw_dec = sp.signal.decimate(pupil_raw, downsample_rate, 1)
			pup_int_dec = sp.signal.decimate(pupil_int, downsample_rate, 1)

			pupil_bp_dec = sp.signal.decimate(pupil_bp, downsample_rate, 1)
			pupil_lp_dec = sp.signal.decimate(pupil_lp, downsample_rate, 1)
			pupil_hp_dec = sp.signal.decimate(pupil_hp, downsample_rate, 1)

			# plot interpolated pupil:
			fig = pl.figure(figsize = (24,9))
			s = fig.add_subplot(311)
			pl.plot(x, pup_raw_dec, 'b'); pl.plot(x, pup_int_dec, 'g')
			pl.ylabel('pupil size'); pl.xlabel('time (s)')
			pl.legend(['raw pupil', 'blink interpolated pupil'])
			s.set_title(self.subject.initials)

			ymin = pupil_raw.min(); ymax = pupil_raw.max()
			tps = (list(trial_phase_times[trial_phase_times['trial_phase_index'] == 2]['trial_phase_EL_timestamp']) - session_start_EL_time, list(trial_phase_times[trial_phase_times['trial_phase_index'] == 3]['trial_phase_EL_timestamp']) - session_start_EL_time)
			for i in range(tps[0].shape[0]):
				pl.axvline(x = tps[0][i] / float(sample_rate), ymin = ymin, ymax = ymax, color = 'r')
				pl.axvline(x = tps[1][i] / float(sample_rate), ymin = ymin, ymax = ymax, color = 'k')
			s.set_ylim(ymin = pup_int_dec.min()-100, ymax = pup_int_dec.max()+100)
			s.set_xlim(xmin = tps[0][0] / float(sample_rate), xmax = tps[1][-1] / float(sample_rate))

			s = fig.add_subplot(312)
			pl.plot(x, pupil_bp_dec, 'b'); pl.plot(x, pupil_lp_dec, 'g');
			pl.ylabel('pupil size'); pl.xlabel('time (s)')
			pl.legend(['band_passed', 'lowpass'])
			s.set_title(self.subject.initials)

			ymin = pupil_raw.min(); ymax = pupil_raw.max()
			tps = (list(trial_phase_times[trial_phase_times['trial_phase_index'] == 2]['trial_phase_EL_timestamp']) - session_start_EL_time, list(trial_phase_times[trial_phase_times['trial_phase_index'] == 3]['trial_phase_EL_timestamp']) - session_start_EL_time)
			for i in range(tps[0].shape[0]):
				pl.axvline(x = tps[0][i] / float(sample_rate), ymin = ymin, ymax = ymax, color = 'r')
				pl.axvline(x = tps[1][i] / float(sample_rate), ymin = ymin, ymax = ymax, color = 'k')
			# s.set_ylim(ymin = pup_int_dec.min()-100, ymax = pup_int_dec.max()+100)
			s.set_xlim(xmin = tps[0][0] / float(sample_rate), xmax = tps[1][-1] / float(sample_rate))

			s = fig.add_subplot(313)
			pl.plot(x, pupil_bp_dec, 'b'); pl.plot(x, pupil_hp_dec, 'b');
			pl.ylabel('pupil size'); pl.xlabel('time (s)')
			pl.legend(['band_passed', 'highpass'])
			s.set_title(self.subject.initials)

			ymin = pupil_raw.min(); ymax = pupil_raw.max()
			tps = (list(trial_phase_times[trial_phase_times['trial_phase_index'] == 2]['trial_phase_EL_timestamp']) - session_start_EL_time, list(trial_phase_times[trial_phase_times['trial_phase_index'] == 3]['trial_phase_EL_timestamp']) - session_start_EL_time)
			for i in range(tps[0].shape[0]):
				pl.axvline(x = tps[0][i] / float(sample_rate), ymin = ymin, ymax = ymax, color = 'r')
				pl.axvline(x = tps[1][i] / float(sample_rate), ymin = ymin, ymax = ymax, color = 'k')
			# s.set_ylim(ymin = pup_int_dec.min()-100, ymax = pup_int_dec.max()+100)
			s.set_xlim(xmin = tps[0][0] / float(sample_rate), xmax = tps[1][-1] / float(sample_rate))
			pl.savefig(os.path.join(self.base_directory, 'figs', alias + '.pdf'))
			pl.close() 



	def events_and_signals_in_time(self, data_type = 'pupil_bp', requested_eye = 'L', saccade_duration_ll = 0.020, do_plot=False):
		"""events_and_signals_in_time takes all aliases' data from the hdf5 file.
		This results in variables self.colour_times, self.sound_times that designate occurrences in seconds time, 
		in the time as useful for the variable self.pupil_data, which contains z-scored data_type type data and
		is still sampled at the original sample_rate. Note: the assumption is that all aliases are sampled at the same frequency. 
		events_and_signals_in_time further creates self.colour_indices and self.sound_indices variables that 
		index which trials (corresponding to _times indices) correspond to which sounds and which reward probabilities.
		"""
		reward_prob_indices = [] # an array that holds the different colour onsets as a boolean array on trials, ordered by their reward probability
		sound_indices = [] # an array that holds the different sound onsets as a boolean array on trials, ordered by their reward value
		hue_indices = [] # an array that holds the different colour onsets as a boolean array on trials, ordered by their hue
		colour_times = []
		fix_times=[]
		color_sound_delay=[]
		anticipation_times = []
		blink_times = []
		nr_blinks=[]
		sound_times = []
		trial_indices = []
		trial_durations = [] 
		pupil_data = []
		padded_pupil_data=[]
		padded_pupil_baseline_z_data=[]
		pupil_baseline_data=[]
		pupil_baseline_data_z=[]
		total_time = []
		which_sounds=[]
		domain_time=[]
		block_reversals=[]
		total_reversals=[]
		samples_per_run = [0]
		samples_per_trial = [0]
		trial_start_sample=[]
		trial_end_sample=[]
		trials_per_run = [0]
		end_time_trial=[]
		last_trial_per_run=[]
		real_reward_probability=[]
		reversals=[]
		reversal_indices=[0]
		reversal_indices_split=[]
		reversal_positions=[]
		fix1, fix2 = [], []
		dx_data = []
		microsaccade_times =[]
		ms_events=[]
		saccade_times=[]
		all_padded_sound_times=[]	
		colour_phase_duration=[]
		sound_phase_duration = [] 
		
		

		if int(self.project_directory.split('/')[-3]) == 3: # first experiment
			which_sounds = [0,3]
		elif int(self.project_directory.split('/')[-3]) == 4: # second experiment
			if self.subject.initials[0] == 'b': 	# in 2nd experiment, the initials determine which sound is rewarded
				which_sounds = [-4,-5]				#piano = reward tone
			elif self.subject.initials[0] == 'a':
				which_sounds = [-5,-4]				#violin = reward tone 
		elif int(self.project_directory.split('/')[-3]) == 5: # third experiment, noise and pling tones
			which_sounds = [0,1]

		counter = 0
		session_time = 0
		padding_time = 60 #seconds  		
		 
		for idx, alias in enumerate(self.aliases):
			trial_times = self.ho.read_session_data(alias, 'trials')
			trial_phase_times = self.ho.read_session_data(alias, 'trial_phases')
			session_start_EL_time = np.array(trial_phase_times[trial_phase_times['trial_phase_index'] == 1]['trial_phase_EL_timestamp'])[0] # np.array(trial_times['trial_start_EL_timestamp'])[0]#
			session_stop_EL_time = np.array(trial_times['trial_end_EL_timestamp'])[-1]			
			total_time = np.array(((session_stop_EL_time - session_start_EL_time)/1000)/60) #total time in minutes
			
			self.sample_rate = self.ho.sample_rate_during_period([session_start_EL_time, session_stop_EL_time], alias)
			
			trial_start_EL_time = np.array(trial_times['trial_start_EL_timestamp'])
			trial_end_EL_time = np.array(trial_times['trial_end_EL_timestamp'])

			trial_parameters = self.ho.read_session_data(alias, 'parameters')			
			reward_prob_indices.append(np.array([(trial_parameters['reward_probability'] == i) for i in [0.2,0.8]])) 
			real_reward_probability.append(np.array([(trial_parameters['reward_probability'])]))
			hue_indices.append(np.array([(trial_parameters['hue'] == i) for i in [0.25, 0.75]]))  #green=0.25, purple=0.75
			sound_indices.append(np.array([(trial_parameters['sound'] == i) for i in which_sounds])) # 0=loss=True 		 
			trial_indices.append(trial_parameters['trial_nr'])
			trials_per_run.append(len(trial_parameters['trial_nr']))


			fix_times.append((np.array(trial_phase_times[trial_phase_times['trial_phase_index']==1]['trial_phase_EL_timestamp']) - session_start_EL_time + session_time) / self.sample_rate)
			colour_times.append((np.array(trial_phase_times[trial_phase_times['trial_phase_index'] == 2]['trial_phase_EL_timestamp']) - session_start_EL_time + session_time) / self.sample_rate )
			sound_times.append((np.array(trial_phase_times[trial_phase_times['trial_phase_index'] == 3]['trial_phase_EL_timestamp']) - session_start_EL_time + session_time) / self.sample_rate )
			end_time_trial.append((np.array(trial_times['trial_end_EL_timestamp']) - session_start_EL_time + session_time) / self.sample_rate)
			 
			colour_phase_duration.append(sound_times[idx] - colour_times[idx])
			sound_phase_duration.append(end_time_trial[idx] - sound_times[idx])
			trial_durations.append((end_time_trial[idx] - fix_times[idx]))

			#load in blink data
			eyelink_blink_data = self.ho.read_session_data(alias, 'blinks_from_message_file')
			eyelink_blink_data_L = eyelink_blink_data[eyelink_blink_data['eye'] == requested_eye] #only select data from left eye
			b_start_times = np.array(eyelink_blink_data_L.start_timestamp)
			b_end_times = np.array(eyelink_blink_data_L.end_timestamp)

			#evaluate only blinks that occur after start and before end experiment
			b_indices = (b_start_times>session_start_EL_time)*(b_end_times<session_stop_EL_time) 
			b_start_times_t = (b_start_times[b_indices] - session_start_EL_time) #valid blinks (start times) 
			b_end_times_t = (b_end_times[b_indices] - session_start_EL_time) 
			blinks = np.array(b_start_times_t)			
			blink_times.append(((blinks + session_time) / self.sample_rate ))

			#load saccade data 
			eyelink_saccade_data = self.ho.read_session_data(alias, 'saccades_from_message_file')
			eyelink_saccade_data_L = eyelink_saccade_data[eyelink_saccade_data['eye'] == requested_eye]
			sac_start_times = np.array(eyelink_saccade_data_L.start_timestamp)
			sac_end_times = np.array(eyelink_saccade_data_L.end_timestamp)

			sac_indices = (sac_start_times>session_start_EL_time)*(sac_end_times < session_stop_EL_time)
			sac_start_times_t = (sac_start_times[sac_indices] - session_start_EL_time)
			sac_end_times_t = (sac_end_times[sac_indices] - session_start_EL_time)
			saccade_times.append((np.array(sac_start_times_t + session_time) / self.sample_rate))
			
			pupil = np.squeeze(self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], alias = alias, signal = 'pupil_bp', requested_eye = requested_eye))
			pupil_data.append((pupil - np.median(pupil))/ pupil.std())

			#Zero padding of pupil signal of signal data_type
			padded_pupil = np.zeros((pupil.shape[0] + 2*(padding_time * self.sample_rate)))
			padded_pupil[padding_time*self.sample_rate:-padding_time*self.sample_rate] = pupil
			padded_pupil_data.append(padded_pupil)

			#baseline and baseline z-scored pupil signal 
			pupil_baseline = np.squeeze(self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], alias = alias, signal = 'pupil_baseline', requested_eye = requested_eye)) 
			pupil_baseline_data.append(pupil_baseline)
			pupil_baseline_z = ((pupil_baseline - np.mean(pupil_baseline)) / pupil_baseline.std())
			pupil_baseline_data_z.append(pupil_baseline_z)
			
			samples_per_run.append(len(pupil))
			samples_per_trial = np.array(trial_phase_times[trial_phase_times['trial_phase_index'] == 1]['trial_phase_EL_timestamp']) - session_start_EL_time  + session_time # np.array(trial_times['trial_start_EL_timestamp'])[0]#
			samples_sound_start = np.array(trial_phase_times[trial_phase_times['trial_phase_index'] == 3]['trial_phase_EL_timestamp']) - session_start_EL_time  + session_time
			
			###eye jitter data ### 
			xy_data = self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], alias = alias, signal = 'gaze_x_int', requested_eye = 'L')
			vel_data = self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], alias = alias, signal = 'vel', requested_eye = 'L')
			x = np.squeeze(xy_data['L_gaze_x_int'])	# get xposition gaze data for eye jitter based estimation
			x = (x-np.median(x)) / x.std() #z-score data 
			dx = np.r_[0, np.diff(x)] # calculate derivative of eye movement --> velocity 
			dx_data.append((dx - dx.mean()) / dx.std())			
			 
			domain_time.append(np.array(trial_parameters['domain']))
			block_reversals = [np.array(np.sum(np.abs(np.diff(dt)))) for dt in domain_time]
			reversals = [np.array(np.diff(dt)) for dt in domain_time]	
			last_trial_per_run = trials_per_run[1:]	 
			
			#reversal positions	
			reversal_positions = [np.array(np.where(np.any([rev !=0], axis=0 )))[-1][:] for rev in reversals]	
			reversal_indices = [np.append(0,[(np.append(reversal_positions[i], last_trial_per_run[i]))]).astype(int) for i in range(len(last_trial_per_run))]
			# reversal_discard_end = [np.delete(reversal_indices[i], -1) for i in range(len(reversal_indices))]
			# reversal_discard_start = [np.delete(reversal_indices[i], 0) for i in range(len(reversal_indices))]
			# first_domain_start = reversal_discard_end
			
			# #split halve reversal positions
			# reversal_indices_2split = [np.round(np.diff(reversal_indices[i])/2).astype(int) for i in range(len(last_trial_per_run))]
			# second_domain_start_2split = [first_domain_start[i] + reversal_indices_2split[i] for i in range(len(last_trial_per_run))]
			
			# #split in thirds reversal positions
			# reversal_indices_3split =[np.round(np.diff(reversal_indices[i])/3).astype(int) for i in range(len(last_trial_per_run))]			
			# first_domain_end_3split = [first_domain_start[i] + reversal_indices_3split[i] for i in range(len(last_trial_per_run))]
			# third_domain_start_3split = [reversal_discard_start[i] - reversal_indices_3split[i] for i in range(len(last_trial_per_run))]
			# third_domain_end_3split = [third_domain_start_3split[i] + reversal_indices_3split[i] for i in range(len(last_trial_per_run))]

			# #Zero-padding of event times
			padded_sound_times = padding_time + sound_times[idx] + ((2*idx)*padding_time) 
			all_padded_sound_times.append(padded_sound_times)

						
			session_time += session_stop_EL_time - session_start_EL_time
		
		
		#concatenate all blinks 
		nr_blinks.append(np.array([len(blink_times[x]) for x in range(len(blink_times))]) )
		blink_rate = np.array(nr_blinks/total_time)  #blink rate per minute 
		
		with pd.get_store(self.ho.inputObject) as h5_file: 
			h5_file.put("/%s"%('blink_rate/blink_rate'), pd.Series(np.squeeze(np.array(blink_rate)))) 
			h5_file.put("/%s"%('domain/domain'), pd.Series(np.squeeze(np.array(domain_time))))	
			h5_file.put("/%s"%('domain/block_reversals'), pd.Series(np.squeeze(np.array(block_reversals))))

		self.end_time_trial =  np.concatenate(end_time_trial)
		self.fix_times = np.concatenate(fix_times)
		self.colour_times = np.concatenate(colour_times)
		self.sound_times = np.concatenate(sound_times)
		self.padded_sound_times = np.concatenate(all_padded_sound_times)
		self.blink_times = np.concatenate(blink_times)
		self.dx_data = np.concatenate(dx_data)			
		self.pupil_data = np.hstack(pupil_data)
		self.padded_pupil_data = np.hstack(padded_pupil_data)
		self.padded_sound_times = np.concatenate(all_padded_sound_times)
		self.pupil_baseline_data_z = np.hstack(pupil_baseline_data_z)
		self.pupil_baseline_data = np.hstack(pupil_baseline_data)
		self.trial_indices = np.hstack(trial_indices)
		self.trial_durations = np.hstack(trial_durations)
		self.reward_prob_indices = np.hstack(reward_prob_indices)
		self.real_reward_probability = np.hstack(real_reward_probability)
		self.sound_indices = np.hstack(sound_indices)
		self.hue_indices = np.hstack(hue_indices)
		self.domain_time = np.hstack(domain_time)
		self.reversal_indices = np.hstack(reversal_indices)
		self.saccade_times = np.concatenate(saccade_times)
		self.run_sample_limits = np.array([np.cumsum(samples_per_run)[:-1],np.cumsum(samples_per_run)[1:]]).T
		self.trial_sound_sample_limits = np.array([np.cumsum(samples_sound_start)[:-1], np.cumsum(samples_sound_start)[1:]]).T
		self.run_trial_limits = np.array([np.cumsum(trials_per_run)[:-1],np.cumsum(trials_per_run)[1:]]).T
		self.samples_per_run = np.hstack(samples_per_run)
		self.colour_phase_duration = np.hstack(colour_phase_duration)
		self.sound_phase_duration = np.hstack(sound_phase_duration)

	
		folder_name = 'pupil_data'
		with pd.get_store(self.ho.inputObject) as h5_file:
			h5_file.put("/%s/%s"%(folder_name, 'pupil_data'), pd.Series(np.array(self.pupil_data)))
			h5_file.put("/%s/%s"%(folder_name, 'sample_rate'), pd.Series(self.sample_rate))

		


		# #split half domain_indices: true = first half of reversal block, false = second half of reversal block
		# #split in thirds domain indices, select the first and third part as domain indices 		
		# for i in range(len(first_domain_start)):
		# 		new_f.append(first_domain_start[i] + counter)
		# 		new_s.append(second_domain_start_2split[i] + counter)				
		# 		first_end_3split.append(first_domain_end_3split[i] + counter)
		# 		third_start_3split.append(third_domain_start_3split[i] + counter)
		# 		third_end_3split.append(third_domain_end_3split[i] + counter)
		# 		counter = self.run_trial_limits[i][1]
		

		# self.domain_indices= np.array([(np.arange(self.run_trial_limits[-1][-1])<x[1]) * (np.arange(self.run_trial_limits[-1][-1])>=x[0]) for x in zip(np.hstack(new_f), np.hstack(new_s))]).sum(axis =0, dtype = bool) #select trials on basis of first and second domain indices
		# #domain indices for reversal blocks split in three. First domain = first third of trials, Second domain = last third of trials
		# self.first_domain_indices_3split = np.array([(np.arange(self.run_trial_limits[-1][-1])<x[1]) * (np.arange(self.run_trial_limits[-1][-1])>=x[0]) for x in zip(np.hstack(new_f), np.hstack(first_end_3split))]).sum(axis =0, dtype = bool) #select trials on basis of first and second domain indices
		# self.third_domain_indices_3split = np.array([(np.arange(self.run_trial_limits[-1][-1])<x[1]) * (np.arange(self.run_trial_limits[-1][-1])>=x[0]) for x in zip(np.hstack(third_start_3split), np.hstack(third_end_3split))]).sum(axis =0, dtype = bool) #select trials on basis of first and second domain indices
		
		# #domains per run
		# dts_per_run = [self.domain_time[i[0]:i[1]] for i in self.run_trial_limits]
		# #colours per run
		# colours = self.real_reward_probability * self.hue_indices
		# green, purple = colours[0], colours[1]
		# green_per_run = [green[i[0]:i[1]] for i in self.run_trial_limits]
		# purple_per_run = [purple[i[0]:i[1]] for i in self.run_trial_limits]
		# lp_colour, hp_colour = self.reward_prob_indices[0].astype(int), self.reward_prob_indices[1].astype(int)
		# #sounds per run 
		# sound_indices_per_run = [self.sound_indices[0,i[0]:i[1]] for i in self.run_trial_limits]
		# sound_times_per_run = [self.sound_times[i[0]:i[1]] for i in self.run_trial_limits]
		# low_rw_sound, high_rw_sound = self.sound_indices[0].astype(int), self.sound_indices[1].astype(int)
		# #colour sound probabilities per run 
		# LP_NR = np.ma.masked_equal(lp_colour * low_rw_sound,0)+.1 
		# HP_HR = np.ma.masked_equal(hp_colour * high_rw_sound,0)+.1 
		# PRPE= np.ma.masked_equal(lp_colour * high_rw_sound,0)+.1 
		# NRPE= np.ma.masked_equal(hp_colour * low_rw_sound,0)+.1 
		# LP_NR_per_run, HP_HR_per_run = [LP_NR[i[0]:i[1]]for i in self.run_trial_limits],[HP_HR[i[0]:i[1]]for i in self.run_trial_limits]
		# PRPE_per_run, NRPE_per_run = [PRPE[i[0]:i[1]]for i in self.run_trial_limits],[NRPE[i[0]:i[1]]for i in self.run_trial_limits]
		

		# #plot all events in time per block 
		# if do_plot == True: 
		# 	fig = pl.figure(figsize = (12,14))
		# 	for i in range(len(self.run_trial_limits)):
		# 		s = fig.add_subplot(len(self.run_trial_limits),1,i+1)
		# 		pl.plot(dts_per_run[i],'r')
		# 		pl.plot(green_per_run[i], 'g')
		# 		pl.plot(purple_per_run[i], 'm')
		# 		PRPE, = pl.plot(PRPE_per_run[i], 'g*')
		# 		NRPE, = pl.plot(NRPE_per_run[i], 'r*')
		# 		LPNR, = pl.plot(LP_NR_per_run[i], 'ko')
		# 		HPHR, = pl.plot(HP_HR_per_run[i], 'k*')
		# 		s.set_ylim([-0.1,1.2])
		# 		simpleaxis(s)
		# 		spine_shift(s)		
		# 	pl.tight_layout()		
		# 	pl.savefig(os.path.join(self.base_directory, 'figs' ,'blocks_per_run.pdf'))	
		# 	pl.close()
	
	def check_each_trial_condition(self): 

		domains=[]
		hue=[]
		reward_probability=[]
		reward_outcome=[]
		reward_condition=[]
		reward_prob_indices=[]
		sound_indices=[]
		original_reward_conditions=[]
		hue_prob_consistensy=[]
				
		for i, alias in enumerate(self.aliases):
			trial_params = self.ho.read_session_data(alias, 'parameters') 
			reward_probability.append(np.array(trial_params['reward_probability']))
			reward_outcome.append(np.array(trial_params['sound']))
			hue.append(np.array(trial_params['hue']))
			domains.append(np.array(trial_params['domain']))

			reward_prob_indices.append(np.array([(trial_params['reward_probability'] == i) for i in [0.2,0.8]])) 
			sound_indices.append(np.array([(trial_params['sound'] == i) for i in [0,1]])) # 0=loss=True 		

		self.low_reward_prob_indices = np.hstack(reward_prob_indices)[0]
		self.low_sound_indices = np.hstack(sound_indices)[0]
		self.reward_probability = np.hstack(reward_probability)
		self.reward_outcome = np.hstack(reward_outcome)
		self.hue = np.hstack(hue)
		self.domains = np.hstack(domains)

		for  trial in range(self.hue.shape[0]): 
			#check the condition of each trial 
			if (self.reward_probability[trial] == 0.2) and (self.reward_outcome[trial] == 0.0): #LPNR
				reward_condition.append([1.0, trial])
			elif (self.reward_probability[trial] == 0.2) and (self.reward_outcome[trial] == 1.0): #LPHR
				reward_condition.append([2.0, trial])
			elif (self.reward_probability[trial] == 0.8) and (self.reward_outcome[trial] == 0.0): #HPNR
				reward_condition.append([3.0, trial])
			else: #HPHR
				reward_condition.append([4.0, trial])	
			
			#compare to old boolean logic scheme 
			if (self.low_reward_prob_indices[trial] == True) and (self.low_sound_indices[trial] == True): #LPNR
				original_reward_conditions.append([1.0, trial])
			elif (self.low_reward_prob_indices[trial] == True) and (self.low_sound_indices[trial] == False): #LPHR
				original_reward_conditions.append([2.0, trial])
			elif (self.low_reward_prob_indices[trial] == False) and (self.low_sound_indices[trial] == True): #HPNR
				original_reward_conditions.append([3.0, trial])
			elif (self.low_reward_prob_indices[trial] == False) and (self.low_sound_indices[trial] == False): #HPHR
				original_reward_conditions.append([4.0, trial])
		
		reward_condition = np.array(reward_condition)
		original_reward_conditions = np.array(original_reward_conditions)
		compare_approaches = [reward_condition[:,0] == original_reward_conditions[:,0]] #similar 

		#compare hue expectation consistency
		dom_rev = np.diff(self.domains)
		rev_pos = np.r_[0, np.where(dom_rev !=0)[0], len(self.domains)]
		rev_limits = [rev_pos[:-1], rev_pos[1:]]
	
		#select start and other hue 
		start_hue = [hue[i][0] for i in range(len(domains))]
		start_prob = [reward_probability[i][0] for i in range(len(domains))]		
	
		#iterate over runs 
		for i, domain in enumerate(domains): 
			#set each run's start hue and probability 
			start_h, start_p = start_hue[i], start_prob[i]
			if start_h == 0.25: 
				other_h = 0.75 
			else: 
				other_h = 0.25 

			if start_p == 0.2: 
				other_p = 0.8
			else: 
				other_p = 0.2  			
			#check each trial in a run 
			for trial in range(len(domain)): 
				if (hue[i][trial] == start_h) and (reward_probability[i][trial] == start_p): #high prob color (start block)
					hue_prob_consistensy.append(0.0)
				elif (hue[i][trial] == other_h) and (reward_probability[i][trial] == other_p): #low prob color (start block) 
					hue_prob_consistensy.append(0.0)
				elif (hue[i][trial] == start_h) and (reward_probability[i][trial] == other_p): #green & low (flip block) 
					hue_prob_consistensy.append(1.0)
				elif (hue[i][trial] == other_h) and (reward_probability[i][trial] == start_p): #purple & high (flip block)
					hue_prob_consistensy.append(1.0)
				
		hue_prob_consistensy = np.array(hue_prob_consistensy)
		compare_domains = [self.domains == hue_prob_consistensy] #similar as to original


	def filter_bank_pupil_signals(self, data_type='pupil_bp', requested_eye='L', subsample_ratio=50, do_plot=False, zscore=True): 
		"""get_filter_bank_signals extracts the filter_bank signals per run and concatenates them in a dictionary of n_frequencies x n_samples  """
		
		self.filter_bank = {} #will contain filtered, z-scored pupil signal ranging in frequency from lower than self.pupil_hp (0.1Hz) until the lowest possible frequency in a run (0.002Hz)
		session_time = 0 	
		f = pl.figure() 
		for alias in self.aliases: 

			trial_times = self.ho.read_session_data(alias, 'trials')
			trial_phase_times = self.ho.read_session_data(alias, 'trial_phases')
			session_start_EL_time = np.array(trial_phase_times[trial_phase_times['trial_phase_index'] == 1]['trial_phase_EL_timestamp'])[0] # np.array(trial_times['trial_start_EL_timestamp'])[0]#
			session_stop_EL_time = np.array(trial_times['trial_end_EL_timestamp'])[-1]	
			self.sample_rate = self.ho.sample_rate_during_period([session_start_EL_time, session_stop_EL_time], alias)						
			
			#Extract filterbank signals from dataframe	
			all_pupil_signals = self.ho.data_from_time_period(time_period = [session_start_EL_time, session_stop_EL_time], alias = alias, columns=None)
			assert filter(lambda k: 'filterbank' in k, all_pupil_signals.keys()), 'No filterbank signals in run %s, please run import_gaze_data()'%alias
			filterbank_keys = filter(lambda k: 'filterbank' in k, all_pupil_signals.keys())
			filterbank_freqs = [filterbank_keys[i].rpartition('_')[-1] for i in range(len(filterbank_keys))]
			filterbank_string = filterbank_keys[0].rpartition('_')[0] 			
			for i, freq in enumerate(filterbank_freqs): 
				filtered_pupil_signal = self.ho.data_from_time_period(time_period = [session_start_EL_time, session_stop_EL_time], alias = alias, columns=filterbank_string+'_'+freq)
				filtered_pupil_signal_z = ((filtered_pupil_signal - np.mean(filtered_pupil_signal))/filtered_pupil_signal.std())
				#option to put zscored or non-zscored signals into self.filter_bank
				if zscore==True: 
					#make new key:value pair when freq does not yet exist
					if not freq in self.filter_bank: 
						self.filter_bank[freq] = filtered_pupil_signal_z
					else:
						self.filter_bank[freq] = np.hstack([self.filter_bank[freq],filtered_pupil_signal_z]) 
				else: 
					if not freq in self.filter_bank: 
						self.filter_bank[freq] = filtered_pupil_signal
					else:
						self.filter_bank[freq] = np.hstack([self.filter_bank[freq],filtered_pupil_signal]) 

			self.logger.info('added filtered signals of block %s to filter_bank'%alias)
			session_time += session_stop_EL_time - session_start_EL_time		
		
	
		if do_plot == True: 
			time_points = np.linspace(0,len(self.filter_bank.values()[0])/self.sample_rate/60, len(self.filter_bank.values()[0]))
			f = pl.figure(figsize=(12,8))
			f.text(0.5, 0.98, '3rd order Butterworth-filtered pupil signal,  %s Hz < frequencies < %s Hz'%(filterbank_freqs[-1], filterbank_freqs[0]), fontsize=10, ha='center')
			for i, signal in enumerate(self.filter_bank.values()[::2]): #plot each second frequency of the whole frequency bank 
				s = f.add_subplot(int(len(self.filter_bank)/4),2,i+1)
				pl.plot(time_points, signal)
				s.set_title(self.filter_bank.keys()[::2][i] + 'Hz', fontsize=9)
				pl.xlabel('time (minutes)')
				pl.ylabel('Z')
				simpleaxis(s)
				pl.tight_layout()
			pl.savefig(os.path.join(self.base_directory, 'figs' , self.subject.initials + '_filter_bank_butter.pdf'))
			pl.close()	
		else: 
			pass 


	def events_and_signals_in_time_TD(self, data_type = 'pupil_bp', requested_eye = 'L'):
		""" events in time needed for Temporal Difference Learning is a shortened version of events_and_signals_in_time, with only those variables that are needed in TD learning 
		(to speed up calculations) """

		hue_indices=[]
		fix_times=[]
		sound_indices=[]
		reward_prob_indices=[]
		sound_times=[]
		trial_indices=[]
		trials_per_run=[0]
		domain_time=[]
		pupil_data=[]
		pupil_baseline_data=[]


		if int(self.project_directory.split('/')[-3]) == 3: # first experiment
			which_sounds = [0,3]
		elif int(self.project_directory.split('/')[-3]) == 4: # second experiment
			if self.subject.initials[0] == 'b': 	# in 2nd experiment, the initials determine which sound is rewarded
				which_sounds = [-4,-5]				#piano = reward tone
			elif self.subject.initials[0] == 'a':
				which_sounds = [-5,-4]				#violin = reward tone 
		elif int(self.project_directory.split('/')[-3]) == 5: # third experiment, noise and pling tones
			which_sounds = [0,1]

		session_time = 0
		 
		for alias in self.aliases:

			trial_times = self.ho.read_session_data(alias, 'trials')
			trial_phase_times = self.ho.read_session_data(alias, 'trial_phases')
			session_start_EL_time = np.array(trial_phase_times[trial_phase_times['trial_phase_index'] == 1]['trial_phase_EL_timestamp'])[0] # np.array(trial_times['trial_start_EL_timestamp'])[0]#
			session_stop_EL_time = np.array(trial_times['trial_end_EL_timestamp'])[-1]	

			self.sample_rate = self.ho.sample_rate_during_period([session_start_EL_time, session_stop_EL_time], alias)

			trial_parameters = self.ho.read_session_data(alias, 'parameters')						
			reward_prob_indices.append(np.array([(trial_parameters['reward_probability'] == i) for i in [0.2,0.8]])) 
			hue_indices.append(np.array([(trial_parameters['hue'] == i) for i in [0.25, 0.75]]))  #green=0.25, purple=0.75
			fix_times.append((np.array(trial_phase_times[trial_phase_times['trial_phase_index']==1]['trial_phase_EL_timestamp']) - session_start_EL_time + session_time) / self.sample_rate)
			sound_times.append((np.array(trial_phase_times[trial_phase_times['trial_phase_index'] == 3]['trial_phase_EL_timestamp']) - session_start_EL_time + session_time) / self.sample_rate )
			sound_indices.append(np.array([(trial_parameters['sound'] == i) for i in which_sounds])) # 0=loss=True 	
			trial_indices.append(trial_parameters['trial_nr'])
			trials_per_run.append(len(trial_parameters['trial_nr']))

			domain_time.append(np.array(trial_parameters['domain']))
			reversals = [np.array(np.diff(dt)) for dt in domain_time]
			reversal_positions = [np.array(np.where(np.any([rev !=0], axis=0 )))[-1][:] for rev in reversals]	

			pupil = np.squeeze(self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], alias = alias, signal = data_type, requested_eye = requested_eye))
			pupil_data.append((pupil - np.median(pupil))/ pupil.std()) 

			pupil_baseline = np.squeeze(self.ho.signal_during_period(time_period = [session_start_EL_time, session_stop_EL_time], alias = alias, signal = 'pupil_baseline_zscore', requested_eye = requested_eye)) 
			pupil_baseline_data.append(pupil_baseline)

			session_time += session_stop_EL_time - session_start_EL_time

		self.reversal_positions = np.array(reversal_positions)
		self.trial_indices = np.hstack(trial_indices)
		self.fix_times = np.concatenate(fix_times)
		self.sound_times = np.concatenate(sound_times)
		self.sound_indices = np.hstack(sound_indices)
		self.hue_indices = np.hstack(hue_indices)
		self.reward_prob_indices = np.hstack(reward_prob_indices)		
		self.run_trial_limits = np.array([np.cumsum(trials_per_run)[:-1],np.cumsum(trials_per_run)[1:]]).T
		self.domain_time = np.hstack(domain_time)
		self.pupil_data = np.hstack(pupil_data)
		self.pupil_baseline_data = np.hstack(pupil_baseline_data)

	def get_domain_time_and_indices(self, data_type = 'pupil_bp_zscore', requested_eye = 'L'):
		""" get reversal information to use in domain calculations """
		
		trials_per_run=[0]
		domain_time=[]
		reversals=[]
		reversal_indices=[0]
		reversal_indices_split=[]
		reversal_positions=[]
		new_f, new_s = [],[]
		first_end_3split, third_start_3split, third_end_3split = [],[],[]


		if int(self.project_directory.split('/')[-3]) == 3: # first experiment
			which_sounds = [0,3]
		elif int(self.project_directory.split('/')[-3]) == 4: # second experiment
			if self.subject.initials[0] == 'b': 	# in 2nd experiment, the initials determine which sound is rewarded
				which_sounds = [-4,-5]				#piano = reward tone
			elif self.subject.initials[0] == 'a':
				which_sounds = [-5,-4]				#violin = reward tone 
		elif int(self.project_directory.split('/')[-3]) == 5: # third experiment, noise and pling tones
			which_sounds = [0,1]

		session_time = 0
		counter = 0 
		 
		for alias in self.aliases:

			trial_times = self.ho.read_session_data(alias, 'trials')
			trial_phase_times = self.ho.read_session_data(alias, 'trial_phases')
			session_start_EL_time = np.array(trial_phase_times[trial_phase_times['trial_phase_index'] == 1]['trial_phase_EL_timestamp'])[0] # np.array(trial_times['trial_start_EL_timestamp'])[0]#
			session_stop_EL_time = np.array(trial_times['trial_end_EL_timestamp'])[-1]	

			self.sample_rate = self.ho.sample_rate_during_period([session_start_EL_time, session_stop_EL_time], alias)

			trial_parameters = self.ho.read_session_data(alias, 'parameters')						
			trials_per_run.append(len(trial_parameters['trial_nr']))

			#calculate the amount of reversals per block 
			domain_time.append(np.array(trial_parameters['domain']))
			block_reversals = [np.array(np.sum(np.abs(np.diff(dt)))) for dt in domain_time]
			reversals = [np.array(np.diff(dt)) for dt in domain_time]	
			last_trial_per_run = trials_per_run[1:]	 
			
			#reversal positions	
			reversal_positions = [np.array(np.where(np.any([rev !=0], axis=0 )))[-1][:] for rev in reversals]	
			reversal_indices = [np.append(0,[(np.append(reversal_positions[i], last_trial_per_run[i]))]).astype(int) for i in range(len(last_trial_per_run))]
			reversal_discard_end = [np.delete(reversal_indices[i], -1) for i in range(len(reversal_indices))]
			reversal_discard_start = [np.delete(reversal_indices[i], 0) for i in range(len(reversal_indices))]
			first_domain_start = reversal_discard_end
			
			#split halve reversal positions
			reversal_indices_2split = [np.round(np.diff(reversal_indices[i])/2).astype(int) for i in range(len(last_trial_per_run))]
			second_domain_start_2split = [first_domain_start[i] + reversal_indices_2split[i] for i in range(len(last_trial_per_run))]
			
			#split in thirds reversal positions
			reversal_indices_3split =[np.round(np.diff(reversal_indices[i])/3).astype(int) for i in range(len(last_trial_per_run))]			
			first_domain_end_3split = [first_domain_start[i] + reversal_indices_3split[i] for i in range(len(last_trial_per_run))]
			third_domain_start_3split = [reversal_discard_start[i] - reversal_indices_3split[i] for i in range(len(last_trial_per_run))]
			third_domain_end_3split = [third_domain_start_3split[i] + reversal_indices_3split[i] for i in range(len(last_trial_per_run))]

			session_time += session_stop_EL_time - session_start_EL_time

		self.run_trial_limits = np.array([np.cumsum(trials_per_run)[:-1],np.cumsum(trials_per_run)[1:]]).T
		self.domain_time = np.hstack(domain_time)
		self.reversal_indices = np.hstack(reversal_indices)
		self.run_trial_limits = np.array([np.cumsum(trials_per_run)[:-1],np.cumsum(trials_per_run)[1:]]).T

		#split half domain_indices: true = first half of reversal block, false = second half of reversal block
		#split in thirds domain indices, select the first and third part as domain indices 		
		
		for i in range(len(first_domain_start)):
				new_f.append(first_domain_start[i] + counter)
				new_s.append(second_domain_start_2split[i] + counter)				
				first_end_3split.append(first_domain_end_3split[i] + counter)
				third_start_3split.append(third_domain_start_3split[i] + counter)
				third_end_3split.append(third_domain_end_3split[i] + counter)
				counter = self.run_trial_limits[i][1]		

		self.domain_indices= np.array([(np.arange(self.run_trial_limits[-1][-1])<x[1]) * (np.arange(self.run_trial_limits[-1][-1])>=x[0]) for x in zip(np.hstack(new_f), np.hstack(new_s))]).sum(axis =0, dtype = bool) #select trials on basis of first and second domain indices
		#domain indices for reversal blocks split in three. First domain = first third of trials, Second domain = last third of trials
		self.first_domain_indices_3split = np.array([(np.arange(self.run_trial_limits[-1][-1])<x[1]) * (np.arange(self.run_trial_limits[-1][-1])>=x[0]) for x in zip(np.hstack(new_f), np.hstack(first_end_3split))]).sum(axis =0, dtype = bool) #select trials on basis of first and second domain indices
		self.third_domain_indices_3split = np.array([(np.arange(self.run_trial_limits[-1][-1])<x[1]) * (np.arange(self.run_trial_limits[-1][-1])>=x[0]) for x in zip(np.hstack(third_start_3split), np.hstack(third_end_3split))]).sum(axis =0, dtype = bool) #select trials on basis of first and second domain indices



	def deconvolve_colour_sounds_nuisance(self, analysis_sample_rate = 20, 
										interval = [-0.5,5.0],  
										data_type = 'pupil_bp_zscore', 
										requested_eye = 'L', 
										deconvolution = 'nuisance_standard_events_rewards'): #'nuisance', 'nuisance_standard_events',
		"""raw deconvolution, to see what happens to pupil size when the fixation colour changes, 
		and when the sound chimes."""

		self.logger.info('starting basic pupil deconvolution with data of type %s and sample_rate of %i Hz in the interval %s, ' % (data_type, analysis_sample_rate, str(interval)))

		if not hasattr(self, 'pupil_data'): # we assume that we'll grab the same events and data whatever the present deconvolve_reward_probabilities method does
			self.events_and_signals_in_time(data_type = data_type, requested_eye= requested_eye)
		subsample_ratio = int(self.sample_rate / analysis_sample_rate)
		input_signal = self.pupil_data[::subsample_ratio]

		with pd.get_store(self.ho.inputObject) as h5_file:									   
			try:																			   
				keypress_times = np.array(h5_file.get("/%s/%s"%('keypresses', 'all_keypresses'))) 
			except (IOError, NoSuchNodeError):
				self.logger.error("no keypresses found participant %s "%self.subject.initials)		 		
		
		if deconvolution == 'nuisance': 
			events = [self.blink_times + interval[0], self.saccade_times + interval [0], keypress_times + interval[0]]
			folder_name = 'deconvolve_nuisance_%s_Hz'%str(analysis_sample_rate)
		elif deconvolution == 'nuisance_standard_events': 
			events = [self.blink_times + interval[0], self.saccade_times + interval [0], keypress_times + interval[0], self.colour_times + interval[0], self.sound_times + interval[0] ]
			folder_name = 'deconvolve_colour_sound_nuisance_%s_Hz'%str(analysis_sample_rate)
		elif deconvolution == 'nuisance_standard_events_rewards': 
			events = [self.blink_times + interval[0], self.saccade_times + interval [0], keypress_times + interval[0], self.colour_times + interval[0]]
			reward_event_times = [self.sound_times[reward] for reward in self.sound_indices]
			events.append(reward_event_times[0] + interval[0]) #loss sound
			events.append(reward_event_times[1] + interval[0]) #reward sound
			folder_name = 'deconvolve_colour_reward_sounds_nuisance_%s_Hz'%str(analysis_sample_rate)	 

		#run deconvolution 
		doNN = ArrayOperator.DeconvolutionOperator( inputObject = input_signal,
							eventObject = events, TR = 1.0/analysis_sample_rate, deconvolutionSampleDuration = 1.0/analysis_sample_rate, 
							deconvolutionInterval = interval[1] - interval[0], run = True )
		doNN.residuals()		
		
		self.logger.info('explained variance (r^sq) %1.4f, analysis_sample_rate = %i'%((1.0 -(np.sum(np.array(doNN.residuals)**2) / np.sum(input_signal**2))), analysis_sample_rate))

		time_points = np.linspace(interval[0], interval[1], np.squeeze(doNN.deconvolvedTimeCoursesPerEventType).shape[1])

		with pd.get_store(self.ho.inputObject) as h5_file:
			h5_file.put("/%s/%s"%(folder_name, 'residuals'), pd.Series(np.squeeze(np.array(doNN.residuals))))
			h5_file.put("/%s/%s"%(folder_name, 'time_points'), pd.Series(time_points))
			h5_file.put("/%s/%s"%(folder_name, 'dec_time_course'), pd.DataFrame(np.squeeze(doNN.deconvolvedTimeCoursesPerEventType).T))
		self.logger.info('Saved nuisance deconvolution %iHz of subject %s in folder %s '%(analysis_sample_rate, self.subject.initials, folder_name))


	def select_domain_indices(self, use_domain='full', requested_eye='L', data_type='pupil_bp_zscore'): 

		self.get_domain_time_and_indices(data_type = data_type, requested_eye = requested_eye)

		#select domain to analyse 
		if use_domain == 'first':
			domain_indices_used_now = self.domain_indices
		elif use_domain == 'second':
			domain_indices_used_now = -self.domain_indices
		elif use_domain == 'full':
			domain_indices_used_now = np.ones(self.domain_indices.shape, dtype = bool)
		elif use_domain == 'first_3split': 							#'first_3split': reversal block is split up in 3 parts, the first part is selected
			domain_indices_used_now = self.first_domain_indices_3split
		elif use_domain == 'third_3split': 							#'second_3split': reversal block is split up in 3 parts, the third part is selected
			domain_indices_used_now = self.third_domain_indices_3split
		# elif use_domain == 'clear': 
		# 	domain_indices_used_now = -self.clarity_indices  #evaluate the periods of time where the participant knows the reward probabilities 
		# elif use_domain == 'unclear': 
		# 	domain_indices_used_now = self.clarity_indices   #evaluate the periods of time where the participant does not know the reward probabilities 



		return domain_indices_used_now

	def deconvolve_full_FIR(self, 
							analysis_sample_rate=10, 
							interval =[-4.0, 4.0], 
							data_type='pupil_bp_zscore', 
							requested_eye='L', 
							use_domain = 'first', 
							):
	 	"""deconvolve_full_FIR uses FIR deconvolution and ridge regression to deconvolve all nuisance, standard and reward prediction error events simultaneously"""

		self.events_and_signals_in_time(data_type = data_type, requested_eye = requested_eye)

		with pd.get_store(self.ho.inputObject) as h5_file:									   
			try:																			   
				all_keypresstimes = np.array(h5_file.get("/%s/%s"%('keypresses', 'all_keypresses'))) 
			except (IOError, NoSuchNodeError):
				self.logger.error("no keypresses found participant %s "%self.subject.initials)

		subsample_ratio = int(self.sample_rate/analysis_sample_rate)
		input_signal = self.pupil_data[::subsample_ratio]

		domain_indices_used_now = self.select_domain_indices(use_domain=use_domain)
	
		self.logger.info('starting deconvolution for subject %s of data_type %s in interval %s using %s domain'%(self.subject.initials, data_type, str(interval), use_domain))
		
		#select regressors 
		blink_times =    [self.blink_times]						
		keypress_times = [all_keypresstimes]
		saccade_times =  [self.saccade_times] 	
		colour_times =   [self.colour_times]
		sound_times =    [self.sound_times]	
		cue_indices =    [
			self.reward_prob_indices[0] * domain_indices_used_now, 
			self.reward_prob_indices[1] * domain_indices_used_now,
		]
		cue_times =      [self.colour_times[ci] for ci in cue_indices]
		
		reward_event_indices = [
			self.reward_prob_indices[0] * self.sound_indices[0] * domain_indices_used_now, #LP NR 
			self.reward_prob_indices[0] * self.sound_indices[1] * domain_indices_used_now, #LP HR  
			self.reward_prob_indices[1] * self.sound_indices[0] * domain_indices_used_now, #HP LR  
			self.reward_prob_indices[1] * self.sound_indices[1] * domain_indices_used_now, #HP HR  			
		]	
		reward_event_times = [self.sound_times[re_i] for re_i in reward_event_indices]

		events=[]							 	#regressor events: 
		events.extend(blink_times) 			 	#[0]
		events.extend(keypress_times)		 	#[1]
		events.extend(saccade_times)		 	#[2]
		events.extend(colour_times)			 	#[3]
		events.extend(sound_times)			 	#[4]
		events.extend([cue_times[0]])			#[5]LP
		events.extend([cue_times[1]])			#[6]HP
		events.extend([reward_event_times[0]])	#LP_NR[7]
		events.extend([reward_event_times[1]]) 	#LP_HR[8] 
		events.extend([reward_event_times[2]]) 	#HP_LR[9]
		events.extend([reward_event_times[3]])	#HP_HR[10] 

		#anticipation covariate 
		anticipation_durations = self.sound_times - self.colour_times
		anticipation_durations_l = anticipation_durations[self.reward_prob_indices[0] * domain_indices_used_now]
		anticipation_durations_h = anticipation_durations[self.reward_prob_indices[1] * domain_indices_used_now]

		#fill for all other stick regressors durations = one sample 
		stick_durations=[]
		for regressor_idx in range(len(events)): 
			stick_durations.append(np.ones((len(events[regressor_idx])))/analysis_sample_rate)
		stick_durations=np.array(stick_durations)

		covariates = {
			'blink.gain': np.ones(len(events[0])), 
			'keypress.gain': np.ones(len(events[1])),
			'saccade.gain': np.ones(len(events[2])),
			'colour.gain': np.ones(len(events[3])), 
			'sound.gain': np.ones(len(events[4])), 	
			'cue_low.gain': np.ones(len(events[5])), 						
			'cue_high.gain': np.ones(len(events[6])), 
			'LPNR.gain': np.ones(len(events[7])), 	
			'LPHR.gain': np.ones(len(events[8])), 
			'HPNR.gain': np.ones(len(events[9])), 
			'HPHR.gain': np.ones(len(events[10]))	
					}

		fd = FIRDeconvolution(
			signal = input_signal, 
			events = events,
			event_names = ['blink', 'keypress', 'saccade','colour','sound','cue_low','cue_high', 'LPNR', 'LPHR', 'HPNR', 'HPHR'], 
			durations = {'blink': stick_durations[0], 'keypress': stick_durations[1], 'saccade': stick_durations[2], 
						'colour': stick_durations[3], 'sound': stick_durations[4], 'cue_low': anticipation_durations_l, 'cue_high': anticipation_durations_h,
						'LPNR': stick_durations[7], 'LPHR': stick_durations[8], 'HPNR': stick_durations[9], 'HPHR': stick_durations[10]},
			sample_frequency = analysis_sample_rate, 
			deconvolution_interval = interval, 
			deconvolution_frequency = analysis_sample_rate,
			covariates = covariates,
			) 

		fd.create_design_matrix()
		# plot_time = 5000
		# f = pl.figure()
		# s = f.add_subplot(111)		
		# s.set_title('design matrix (%i Hz)'%analysis_sample_rate)
		# pl.imshow(fd.design_matrix[:,:plot_time], aspect = 0.075 * plot_time/fd.deconvolution_interval_size, interpolation = 'nearest', rasterized = True)
		# sn.despine(offset=10)

		fd.ridge_regress(cv=5, alphas=np.linspace(0,200,20)) 
		fd.calculate_rsq()
		self.logger.info('Rsquared for participant %s: %f' %(self.subject.initials, fd.rsq))

		#pre-allocate betas matrix  
		betas = pd.DataFrame(np.zeros((fd.deconvolution_interval_size, len(fd.covariates))), columns=fd.covariates.keys())
		for i,b in enumerate(fd.covariates.keys()): 
			betas[b] = np.squeeze(fd.betas_for_cov(covariate=b))
		prediction_error = np.mean([betas['LPHR.gain'], betas['HPNR.gain']], axis=0)

		#save all important variables in hdf5
		folder_name = 'deconvolve_full_FIR_%s_domain'%use_domain
		#folder_name = 'deconvolve_full_FIR_%s_domain_long_interval'%use_domain
		with pd.get_store(self.ho.inputObject) as h5_file:
			h5_file.put("/%s/%s"%(folder_name, 'time_points'), pd.Series(fd.deconvolution_interval_timepoints))
			h5_file.put("/%s/%s"%(folder_name, 'deconvolved_pupil_timecourses'), betas)
			h5_file.put("/%s/%s"%(folder_name, 'prediction_error_kernel'), pd.Series(prediction_error))
			h5_file.put("/%s/%s"%(folder_name, 'rsquared'), pd.Series(fd.rsq))
			h5_file.put("/%s/%s"%(folder_name, 'keys'), pd.DataFrame(fd.covariates.keys()))
			h5_file.put("/%s/%s"%(folder_name, 'alpha'), pd.Series(fd.rcv.alpha_))
		self.logger.info('Saved ridge regression %s for participant %s in folder %s' %(use_domain, self.subject.initials, folder_name))


	
	def deconvolve_full(self, analysis_sample_rate=20, interval =[-0.5, 5.5], data_type='pupil_bp_zscore', requested_eye='L', use_domain = 'first'):
		"""deconvolve full takes the raw pupil data and deconvolves blinks, cue events, reward events, microsaccades and key presses """

		self.events_and_signals_in_time(data_type = data_type, requested_eye = requested_eye)

		#get keypresses
		with pd.get_store(self.ho.inputObject) as h5_file:									   
			try:																			   
				all_keypresstimes = np.array(h5_file.get("/%s/%s"%('keypresses', 'all_keypresses'))) 
			except (IOError, NoSuchNodeError):
				self.logger.error("no keypresses found participant %s "%self.subject.initials)

		subsample_ratio = int(self.sample_rate/analysis_sample_rate)
		input_signal = self.pupil_data[::subsample_ratio]
		#select domain to analyse 
		domain_indices_used_now = self.select_domain_indices(use_domain=use_domain)
		
		self.logger.info('starting deconvolution of %s in interval %i using %s domain'%(data_type, interval, use_domain))
		
		#select regressors 
		blink_times = [self.blink_times + interval[0]]						
		keypress_times = [all_keypresstimes + interval[0]]
		saccade_times = [self.saccade_times + interval[0]] 	
		colour_times = [self.colour_times + interval[0]]
		sound_times = [self.sound_times + interval[0]]	
		cue_indices = [
			self.reward_prob_indices[0] * domain_indices_used_now, 
			self.reward_prob_indices[1] * domain_indices_used_now,
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


		do = ArrayOperator.DeconvolutionOperator( inputObject =  input_signal, 
							eventObject = events, TR = 1.0/analysis_sample_rate, deconvolutionSampleDuration = 1.0/analysis_sample_rate, deconvolutionInterval = interval[1] - interval[0], run = True )
		time_points = np.linspace(interval[0], interval[1], np.squeeze(do.deconvolvedTimeCoursesPerEventType).shape[1])
		do.residuals()
		
		#all deconvolved regressor timecourses 
		timecourses = np.squeeze(do.deconvolvedTimeCoursesPerEventType) 
		nuisance_timecourses = timecourses[:3,:]
		standard_response_timecourses = np.r_[[timecourses[3], timecourses[6]]]
		cue_high_low_timecourses = np.r_[[timecourses[4], timecourses[5]]]
		reward_timecourses = np.r_[[timecourses[7], timecourses[8], timecourses[9], timecourses[10]]]
		rpe_timecourses = np.r_[[timecourses[8] - timecourses[10], timecourses[9] - timecourses[7]]] #PRPE: LPHR - HPHR, NRPE: HPNR - LPNR 
		pe_timecourses = np.r_[[timecourses[10] - timecourses[7], timecourses[8] - timecourses[9]]]  #Predicted reward - predicted loss, Unpredicted reward - unpredicted los 
		
		#plot all regressor timecourses 
		f = pl.figure(figsize = (10,10))		
		ax = f.add_subplot(321)
		colours=['k','b','y']
		alphas = np.ones(len(timecourses)) * 1.0 
		lws = np.ones(len(timecourses)) * 1.0 
		for x in range(len(nuisance_timecourses)): 
			pl.plot(time_points, nuisance_timecourses[x], colours[x], alpha=alphas[x], linewidth=lws[x])
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		ax.set_xlim(xmin=interval[0], xmax=interval[1])
		pl.legend(['blinks', 'keypresses', 'saccades'], loc='best')
		simpleaxis(ax)
		spine_shift(ax)
		pl.title('Effect of blinks, keypresses and saccades (%s)'%use_domain)
		pl.ylabel('Z')	
		pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials + '_deconvolve_full_%s_domain.pdf'%use_domain))
	
		folder_name = 'deconvolve_full_%s_domain'%use_domain
		with pd.get_store(self.ho.inputObject) as h5_file:
			h5_file.put("/%s/%s"%(folder_name, 'residuals'), pd.Series(np.squeeze(np.array(do.residuals))))
			h5_file.put("/%s/%s"%(folder_name, 'time_points'), pd.Series(time_points))
			h5_file.put("/%s/%s"%(folder_name, 'dec_time_course'), pd.DataFrame(np.squeeze(do.deconvolvedTimeCoursesPerEventType).T))		

	


	def pupil_baselines_phasic_amplitude_correlations(self, 
													  data_type = 'pupil_bp_zscore', 
													  requested_eye='L', 
													  analysis_sample_rate=10, 
													  fix_dur=0.5, 
													  peak=2): 
		""" correlate phasic baseline and all filter_bank baselines with the amplitude of the sound residuals (physical effect sound is regressed)"""

		#get event times and filter_bank signals
		self.events_and_signals_in_time() 
		self.filter_bank_pupil_signals(do_plot=False, zscore=True)
		subsample_ratio=int(self.sample_rate/analysis_sample_rate)
		ds_pupil_data = self.pupil_data[::subsample_ratio]

		#downsample filter_bank signals and extract per trial baseline 
		fix_start_idx = np.around(self.fix_times*analysis_sample_rate).astype(int)
		fix_samples = int(fix_dur * analysis_sample_rate)
		filter_bank_per_trial_bl = {}
		filter_bank_per_trial_diff = {}
		for key, signal in self.filter_bank.items(): 
			ds_filter_bank_signal = self.filter_bank[key][::subsample_ratio]
			this_filter_baseline = np.array([ds_filter_bank_signal[fix:fix+fix_samples].mean(axis=0) for  fix in fix_start_idx])
			filter_bank_per_trial_bl[key] = this_filter_baseline 
			filter_bank_per_trial_diff[key] = np.diff(this_filter_baseline)			 

		with pd.get_store(self.ho.inputObject) as h5_file:	
			try:
				ds_residuals = h5_file.get("/%s/%s"%('deconvolve_colour_reward_sounds_nuisance_20_Hz', 'residuals'))								   
				#prediction_error_kernel = h5_file.get("/%s/%s"%('deconvolve_full_FIR_full_domain','prediction_error_kernel')) #deconvolved at 10 hz
				time_points = h5_file.get("/%s/%s"%('deconvolve_full_FIR_full_domain','time_points'))										
			except (IOError, NoSuchNodeError):
				self.logger.error("no data present")

		#blow up sound_times by the analysis_sample_rate and use as index in the phasic signal  
		sound_start_idx = np.around(self.sound_times*analysis_sample_rate).astype(int)
		sound_peak = int(peak) * analysis_sample_rate
		 
		selected_sound_intervals = [ds_residuals[sound:sound+sound_peak] for sound in sound_start_idx]

		#place sound_intervals in matrix 
		selected_sound_interval_matrix = np.zeros((len(sound_start_idx), sound_peak))
		for i, val in enumerate(selected_sound_intervals): 
			selected_sound_interval_matrix[i,0:len(val)] = val
		
		#select different trial baselines 
		phasic_trial_baseline = np.array([ds_residuals[fix:fix+fix_samples].mean(axis=0) for fix in fix_start_idx])
		phasic_sound_baseline = selected_sound_interval_matrix[:,0]
		
		#calculate each trial's base to peak change as a measure of amplitude 
		base_to_peak_change = selected_sound_interval_matrix[:,-1] - phasic_sound_baseline

		#ordered frequencies 
		freqs = np.array([float(key) for key in filter_bank_per_trial_bl.keys()])
		freq_order = np.argsort(freqs)	
		keys = filter_bank_per_trial_bl.keys()
		
		# #all ordered baseline-frequency correlations
		freq_corrs_simple = np.array([[keys[i], sp.stats.spearmanr(base_to_peak_change, filter_bank_per_trial_bl[keys[i]])[0]] for i in freq_order]).T
		freq_corrs_simple = np.r_[freq_corrs_simple.T, [[0.15, sp.stats.spearmanr(base_to_peak_change, phasic_trial_baseline)[0]], [0.2, sp.stats.spearmanr(base_to_peak_change, phasic_sound_baseline)[0]]]].T
		pval_corrs_simple = np.array([[keys[i], sp.stats.spearmanr(base_to_peak_change, filter_bank_per_trial_bl[keys[i]])[1]] for i in freq_order]).T
		pval_corrs_simple = np.r_[pval_corrs_simple.T, [[0.15, sp.stats.spearmanr(base_to_peak_change, phasic_trial_baseline)[1]], [0.2, sp.stats.spearmanr(base_to_peak_change, phasic_sound_baseline)[1]]]].T
		freq_corrs_simple_diff = np.array([[keys[i], sp.stats.spearmanr(base_to_peak_change[:-1], filter_bank_per_trial_diff[keys[i]])[0]] for i in freq_order]).T

		######## use prediction error kernels to project sound responses onto 
		# baseline_times = time_points < 0 
		# prediction_error_kernel_s = myfuncs.smooth(prediction_error_kernel.as_matrix(), window_len=10) - prediction_error_kernel[baseline_times].mean() 
		# prediction_error_kernel_s = prediction_error_kernel_s[peak_period[0]*analysis_sample_rate:peak_period[1]*analysis_sample_rate]	
			
		##demean using event baseline 
		# sound_trial_responses_demeaned = np.array([(selected_sound_interval_matrix[i] - p_s_b ) for i, p_s_b in enumerate(phasic_sound_baseline)])
		
		##projection of sound events on sound kernel 
		#sound_amplitude = np.array([np.dot(prediction_error_kernel_s, sound_trial) for sound_trial in sound_trial_responses_demeaned]/(np.linalg.norm(prediction_error_kernel_s)**2)) 

		# freq_corrs_projection = np.array([[keys[i], sp.stats.spearmanr(sound_amplitude, filter_bank_per_trial_bl[keys[i]])[0]] for i in freq_order]).T
		# freq_corrs_projection = np.r_[freq_corrs_projection.T, [[0.15, sp.stats.spearmanr(sound_amplitude, phasic_trial_baseline)[0]], [0.2, sp.stats.spearmanr(sound_amplitude, phasic_sound_baseline)[0]]]].T
		# freq_corrs_projection_diff = np.array([[keys[i], sp.stats.spearmanr(sound_amplitude[:-1], filter_bank_per_trial_diff[keys[i]])[0]] for i in freq_order]).T
		# average_prediction_error_kernel = np.mean(np.load(os.path.join(os.path.split(self.base_directory)[0], 'group_level/data/prediction_error_kernel.npy')), axis=0) 
		# average_prediction_error_kernel = average_prediction_error_kernel[peak_period[0]*analysis_sample_rate:peak_period[1]*analysis_sample_rate]
		
		# sound_amplitude_av_pe_kernel = np.array([np.dot(average_prediction_error_kernel, sound_trial) for sound_trial in sound_trial_responses_demeaned]/(np.linalg.norm(average_prediction_error_kernel)**2)) 

		# freq_corrs_av_projection = np.array([[keys[i], sp.stats.spearmanr(sound_amplitude_av_pe_kernel, filter_bank_per_trial_bl[keys[i]])[0]] for i in freq_order]).T
		# freq_corrs_av_projection = np.r_[freq_corrs_av_projection.T, [[0.15, sp.stats.spearmanr(sound_amplitude_av_pe_kernel, phasic_trial_baseline)[0]], [0.2, sp.stats.spearmanr(sound_amplitude_av_pe_kernel, phasic_sound_baseline)[0]]]].T
		# freq_corrs_av_projection_diff = np.array([[keys[i], sp.stats.spearmanr(sound_amplitude_av_pe_kernel[:-1], filter_bank_per_trial_diff[keys[i]])[0]] for i in freq_order]).T	

		#save all relevant baseline correlation variables in HDF5 file
		folder_name = 'pupil_all_baselines_phasic'
		with pd.get_store(self.ho.inputObject) as h5_file:
			h5_file.put("/%s/%s"%(folder_name, 'pupil_baseline_correlations_simple'), pd.DataFrame(freq_corrs_simple.T, columns = pd.Series(['frequencies','correlations'])))
			h5_file.put("/%s/%s"%(folder_name, 'pupil_baseline_diff_correlations_simple'), pd.DataFrame(freq_corrs_simple_diff.T, columns = pd.Series(['frequencies','correlations'])))
			h5_file.put("/%s/%s"%(folder_name, 'pupil_baseline_pvals_simple'), pd.DataFrame(pval_corrs_simple.T, columns = pd.Series(['frequencies','pvalues'])))			
			# h5_file.put("/%s/%s"%(folder_name, 'pupil_baseline_correlations_projection'), pd.DataFrame(freq_corrs_projection.T, columns = pd.Series(['frequencies','correlations'])))
			# h5_file.put("/%s/%s"%(folder_name, 'pupil_baseline_correlations_projection_diff'), pd.DataFrame(freq_corrs_projection_diff.T, columns = pd.Series(['frequencies','correlations'])))
			# h5_file.put("/%s/%s"%(folder_name, 'pupil_baseline_correlations_av_projection'), pd.DataFrame(freq_corrs_av_projection.T, columns = pd.Series(['frequencies','correlations'])))
			# h5_file.put("/%s/%s"%(folder_name, 'pupil_baseline_correlations_av_projection_diff'), pd.DataFrame(freq_corrs_av_projection_diff.T, columns = pd.Series(['frequencies','correlations'])))


	def pupil_baseline_amplitude(self, data_type='pupil_bp_zscore', requested_eye='L', analysis_sample_rate=10, fix_dur=0.5, signal_for_baseline='tonic_at_fixation', 		event_time_samples = 80):

		#get downsampled pupil data and fixation intervals from downsampled_pupil_and_events
		self.events_and_signals_in_time(data_type='pupil_bp_zscore')
		subsample_ratio = self.sample_rate/analysis_sample_rate

		if signal_for_baseline == 'phasic_at_fixation': 
			ds_pupil = self.pupil_data[::subsample_ratio] #downsample to 20Hz
		elif signal_for_baseline == 'tonic_at_fixation': 
			ds_pupil = self.pupil_baseline_data_z[::subsample_ratio]
		
		#get deconvolved timecouses of colour and sound to make template 		
		with pd.get_store(self.ho.inputObject) as h5_file:									   
			try:																			   
				#deconvolved_responses = h5_file.get("/%s/%s"%('deconvolve_colour_sound_no_eye_jitter', 'dec_time_course'))	
				deconvolved_responses = h5_file.get("/%s/%s"%('deconvolve_colour_sound_nuisance_%s_Hz'%str(analysis_sample_rate), 'dec_time_course'))	
			except (IOError, NoSuchNodeError):
				self.logger.error("no deconvolved timecourses present")	

		#80 samples per deconvolved event type (~4 sec)
		event_time_samples = int(len(deconvolved_responses[:80]))		

		#sound and colour pupil start times 
		sound_idx = np.array(self.sound_times * analysis_sample_rate).astype(int)
		colour_idx =  np.array(self.colour_times * analysis_sample_rate).astype(int)		
		sound_trial_responses = np.array([ds_pupil[si:si+event_time_samples] for si in sound_idx])[:-1] #remove last trial due to too little samples (51)
		colour_trial_responses = np.array([ds_pupil[ci:ci+event_time_samples] for ci in colour_idx])

		fix_start_idx = np.around(self.fix_times*analysis_sample_rate).astype(int) 
		fix_duration = fix_dur * analysis_sample_rate
		pupil_during_fix = np.array([ds_pupil[fix:fix+fix_duration].mean(axis=0) for fix in fix_start_idx])

		#demean timecourses pupil_at_fixation
		sound_trial_responses_demeaned = np.array([(sound_trial_responses[i] - pupil_fix ) for i, pupil_fix in enumerate(pupil_during_fix[:-1])])
		colour_trial_responses_demeaned = np.array([(colour_trial_responses[i] - pupil_fix ) for i, pupil_fix in enumerate(pupil_during_fix)])

		#event templates
		colour_template = np.array(deconvolved_responses[1][:event_time_samples]) #kernel length = 100 samples (~4 sec)
		sound_template = np.array(deconvolved_responses[2][:event_time_samples])
		
		#amplitude calculation using pupil_at_fixation demeaning
		sound_amp = np.array([np.dot(sound_template, sptd) for sptd in sound_trial_responses_demeaned]/(np.linalg.norm(sound_template)**2)) 
		colour_amp = np.array([np.dot(colour_template, cptd) for cptd in colour_trial_responses_demeaned] /(np.linalg.norm(colour_template)**2))

		#spearman correlation  
		corr_pupil_fix_sound_amp = sp.stats.spearmanr(pupil_during_fix[:-1], sound_amp)  
		corr_pupil_fix_colour_amp = sp.stats.spearmanr(pupil_during_fix, colour_amp) 		
	
		#save all relevant baseline correlation variables in HDF5 file
		#folder_name = 'pupil_baseline_amplitude_%s'%signal_for_baseline
		folder_name = 'pupil_baseline_amplitude_%s_%s_Hz'%(signal_for_baseline, analysis_sample_rate)
		with pd.get_store(self.ho.inputObject) as h5_file:
			h5_file.put("/%s/%s"%(folder_name, 'fix_pupil'), pd.Series(pupil_during_fix))
			h5_file.put("/%s/%s"%(folder_name, 'sound_amp'), pd.Series(sound_amp))
			h5_file.put("/%s/%s"%(folder_name, 'colour_amp'), pd.Series(colour_amp))
			h5_file.put("/%s/%s"%(folder_name, 'corr_pupil_fix_sound_amp'), pd.Series(corr_pupil_fix_sound_amp))
			h5_file.put("/%s/%s"%(folder_name, 'corr_pupil_fix_colour_amp'), pd.Series(corr_pupil_fix_colour_amp))


	
	def kernel_fit_gamma(self): 
		"""fit of event and reward IRF kernel (De Gee, 2014)"""
		#load IRF kernels 
		
		event_kernel = np.load('/home/shared/reward_pupil/5/data/reward_prediction_error/kernels/kernel_event.npy')
		event_kernel = event_kernel[25:] #cut first 0.5 seconds from response 
		reward_kernel = np.load('/home/shared/reward_pupil/5/data/reward_prediction_error/kernels/kernel_reward_diff.npy')
		reward_kernel = reward_kernel[25:] 

		timepoints = np.linspace(0, 7.0, event_kernel.size)

		fitfunc = lambda p, t: p[0] * stats.gamma.pdf(t, a = p[1], loc = p[2], scale  = p[3])   #p = parameters to fit [t=input, a=shape, loc=location, scale=scale],
		errfunc = lambda p, t, y: fitfunc (p,t) - y  				   						    #distance between data and fit 
		p0 = [1, 2, 1, 2]  #initial guess for parameters
		p1_ev, success_ev = sp.optimize.leastsq(errfunc, p0[:], args=(timepoints, event_kernel))	#fit procedure
		pl.figure()
		pl.plot(timepoints, event_kernel, "bo", timepoints, fitfunc(p1_ev, timepoints), "b-") 	#plot data and fit 

		p1_rew, success_rew = sp.optimize.leastsq(errfunc, p0[:], args=(timepoints, reward_kernel)) #fit procedure
		pl.plot(timepoints, reward_kernel, "ro", timepoints, fitfunc(p1_rew, timepoints), "r-")  #plot data and fit 
		pl.title('kernel fit Gamma')
		pl.legend(['data_event', 'fit_event', 'data_rew', 'fit_rew'])		
		pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials + '_gamma_kernel_fit.pdf'))


	def single_trial_GLM_one_gamma_kernel(self, 
								input_data_type = 'residuals', 
								irf_kernel_params={'a_reward':2.13286891, 'loc_reward': 1.03410114,'scale_reward': 1.11341402,
												   'a_event': 2.66242099, 'loc_event': 0.34845018,'scale_event': 0.4001978},
								kernel_time_interval=[0,10.0], 
								analysis_sample_rate=25, 
								requested_eye='L', 
								subsample_ratio=5,
								which_kernel='event'): 
		"""single_trial_GLM_one_gamma_kernel performs single trial GLM analysis to fit one IRF gamma kernel [reward kernel] on input_data. Per trial, two event types [cue and sound] are fitted on this kernel, which results in 
		approx. 800 trials x 2 events = 1600 regressor beta values per participant. The GLM is evaluated using statmodelfit OLS and saved in a pickle file """

		if which_kernel == 'reward': 
			kernel = stats.gamma.pdf(np.linspace(kernel_time_interval[0], kernel_time_interval[1], (kernel_time_interval[1]-kernel_time_interval[0]) * analysis_sample_rate), 
										a = irf_kernel_params['a_reward'], loc = irf_kernel_params['loc_reward'], scale = irf_kernel_params['scale_reward'])
		if which_kernel == 'event': 
			kernel = stats.gamma.pdf(np.linspace(kernel_time_interval[0], kernel_time_interval[1], (kernel_time_interval[1]-kernel_time_interval[0]) * analysis_sample_rate), 
								a = irf_kernel_params['a_event'], loc = irf_kernel_params['loc_event'], scale = irf_kernel_params['scale_event'])
		 
		if input_data_type == 'residuals':
			with pd.get_store(self.ho.inputObject) as h5_file:									   
				try:																			   
					input_data = h5_file.get("/%s/%s"%('deconvolve_colour_sound', 'residuals')) 
				except (IOError, NoSuchNodeError):
					self.logger.error("no residuals present")	
				finally:
					self.events_and_signals_in_time(data_type = 'pupil_bp_zscore', requested_eye= requested_eye)
		else:	
			self.events_and_signals_in_time(data_type = input_data_type, requested_eye = requested_eye) 
			input_data = self.pupil_data

		# create timepoints for events
		total_nr_regressors = (self.sound_times.shape[0] + self.colour_times.shape[0])   # one type of IRF per event, 2 events --> 2 regressors per trial
		raw_design_matrix = np.zeros((total_nr_regressors,len(input_data)))

		# fill in regressor events		
		sound_sample_indices = np.array(np.round(self.sound_times * analysis_sample_rate), dtype = int)
		colour_sample_indices = np.array(np.round(self.colour_times * analysis_sample_rate), dtype = int)
		
		convolved_design_matrix = np.zeros(raw_design_matrix.shape)
		for this_kernel, event_indices, shift in zip([kernel, kernel], 
												[sound_sample_indices, colour_sample_indices], 
												np.arange(2) * self.sound_times.shape[0]):
			for i,t in enumerate(event_indices):		# slow reward kernel regressors for colour and sound events
				raw_design_matrix[i + shift,t] = 1;	
				convolved_design_matrix[i + shift] = fftconvolve(raw_design_matrix[i + shift], this_kernel, 'full')[:convolved_design_matrix[i+shift].shape[0]] # implicit padding here, done by indexing
		
		#demean  regressors 
		convolved_design_matrix = (convolved_design_matrix.T - convolved_design_matrix.mean(axis=1)).T 

		#multiple regression using statmodelfit 
		X = sm.add_constant(convolved_design_matrix[:,::subsample_ratio].T) #take every 5th element of convolved_design_matrix to speed up calculation
		model = sm.OLS(input_data[::subsample_ratio], X)
		results = model.fit()
		self.logger.info(results.summary())
		if which_kernel == 'reward': 
			results.save(os.path.join(self.base_directory, 'processed', self.subject.initials + '_single_trial_GLM_one_gamma_kernel_demeaned_regressors_%s.pickle'%input_data_type))
		if which_kernel == 'event': 
			results.save(os.path.join(self.base_directory, 'processed', self.subject.initials + '_single_trial_GLM_event_gamma_kernel_demeaned_regressors_%s.pickle'%input_data_type))



	def single_trial_GLM_dual_gamma_kernel(self, 
								input_data_type = 'residuals',
								irf_kernel_params = {'a_event': 2.66242099, 'loc_event': 0.34845018,'scale_event': 0.4001978,
													'a_reward':2.13286891, 'loc_reward': 1.03410114,'scale_reward': 1.11341402},
								kernel_time_interval = [0,10.0], 
								analysis_sample_rate = 25,
								requested_eye = 'L',
								subsample_ratio = 5
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
					input_data = h5_file.get("/%s/%s"%('deconvolve_colour_sound', 'residuals')) 
				except (IOError, NoSuchNodeError):
					self.logger.error("no residuals present")	
				finally:
					self.events_and_signals_in_time(data_type = 'pupil_bp_zscore', requested_eye= requested_eye)
			
		else:	# for example, 'pupil_bp_zscore'
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

		# #multiple regression 
		# convolved_design_matrix = np.mat(convolved_design_matrix[:,::subsample_ratio]).T  #take every 5th element of convolved_design_matrix to speed up calculation
		# betas = ((convolved_design_matrix.T * convolved_design_matrix).I * convolved_design_matrix.T) * np.mat(input_data[:,::subsample_ratio]).T
		# betas = np.array(betas)
	
		#multiple regression using statmodelfit 
		X = sm.add_constant(convolved_design_matrix[:,::subsample_ratio].T) #take every 5th element of convolved_design_matrix to speed up calculation
		model = sm.OLS(input_data[::subsample_ratio], X)
		results = model.fit()
		self.logger.info(results.summary())
		results.save(os.path.join(self.base_directory, 'processed', self.subject.initials + '_single_trial_GLM_dual_gamma_kernel_demeaned_regressors_%s.pickle'%input_data_type))

	def single_trial_GLM_one_gamma_kernel_results(self, input_data_type='residuals', data_type='pupil_lp_zscore', requested_eye='L', interval=0.5, analysis_sample_rate=25, which_kernel='event'):
		"""inspect resuls from single_trial_GLM_one_gamma_kernel """ 

		self.events_and_signals_in_time(data_type = data_type, requested_eye = requested_eye) 
		self.downsampled_pupil_and_events(data_type=data_type, requested_eye=requested_eye)

		if which_kernel == 'reward': 
			GLM_results = np.load(self.base_directory + '/processed/%s_single_trial_GLM_one_gamma_kernel_demeaned_regressors_%s.pickle'%(self.subject.initials, input_data_type))
		if which_kernel == 'event': 
			GLM_results = np.load(self.base_directory + '/processed/%s_single_trial_GLM_event_gamma_kernel_demeaned_regressors_%s.pickle'%(self.subject.initials, input_data_type))
		

		betas = GLM_results.params[1:] #all beta values - constant
		split_betas = np.split(betas, 2)
	
		beta_sound=np.array(split_betas[0])
		beta_colour=np.array(split_betas[1])
		diff_beta_sound_colour = beta_sound - beta_colour
		zscored_beta_sound = (beta_sound - beta_sound.mean())/beta_sound.std() 
		zscored_beta_colour = (beta_colour - beta_colour.mean()) / beta_colour.std()

		#linear regression pupil baseline & beta values
		#parameters
		n=len(self.fix_pupil_mean)
		x= np.copy(self.fix_pupil_mean)
		f=pl.figure()
		for i in range(len(split_betas)):
			y= np.copy(split_betas[i])
			#fitted slope and y-intercept 
			(ar,br)=polyfit(x, y, 1)
			xr=polyval([ar,br],x)
			#mean square error 
			err=sp.sqrt(sum((xr-y)**2)/n)
			#plot regression for all beta conditions
			ax= f.add_subplot(2,2,i+1)
			pl.plot(x,y, 'ro', alpha=0.5)
			pl.plot(x,xr, 'g-')
			pl.title('split_betas %s'%i)
			pl.xlabel('pupil baseline')
			pl.ylabel('beta value')
			pl.legend(['original', 'regression: r=%.2f' %ar] )
			print('Linear regression using polyfit')
			print('parameters: x=fix_pupil_mean y=split_betas_%s \nregression: a=%.2f b=%.2f, ms error= %.3f' % (i,ar,br,err))			
			pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials + 'pupil_baseline_GLM_%s_kernel_beta_correlations.pdf'%which_kernel))


		#reversal indices to be able to sort beta values 
		raw_reversal_indices = np.split(self.reversal_indices, np.where(self.reversal_indices[:]== 0)[0][1:])
		cum_reversal_indices = [] 
		counter = 0 
		for i in range(len(raw_reversal_indices)):
			cum_reversal_indices.append(raw_reversal_indices[i] + counter) 
			counter = self.run_trial_limits[i][1]

		#condition indices 
		low_rw_sound, high_rw_sound = self.sound_indices[0], self.sound_indices[1]
		lp_colour, hp_colour = self.reward_prob_indices[0], self.reward_prob_indices[1]

		#reversal block indices and limits 
		block_indices = np.unique(np.concatenate(cum_reversal_indices)) 	
		block_limits = np.array([block_indices[:-1],block_indices[1:]]).T 

		#conditions to mask 
		prpe_trials_mask = high_rw_sound * lp_colour 
		nrpe_trials_mask = low_rw_sound * hp_colour
		hphr_trials_mask = high_rw_sound * hp_colour 
		lplr_trials_mask = low_rw_sound * lp_colour

		#index trials on the basis of their occurrence in a reversal block 
		block_indexed_trials = np.concatenate([np.arange(0,bl[1]-bl[0]) for bl in block_limits])

		#order condition trials on position in reversal block (early --> late)
		prpe_order = np.argsort(block_indexed_trials[prpe_trials_mask]) 
		nrpe_order = np.argsort(block_indexed_trials[nrpe_trials_mask])
		hphr_order = np.argsort(block_indexed_trials[hphr_trials_mask]) 
		lplr_order = np.argsort(block_indexed_trials[lplr_trials_mask])

		zscored_diff_beta_sound_colour = np.concatenate([(diff_beta_sound_colour[bi[0]:bi[1]]-diff_beta_sound_colour[bi[0]:bi[1]].mean())/diff_beta_sound_colour[bi[0]:bi[1]].std()  for bi in block_limits])
		
		#barplot early and late beta values for different trial conditions 		
		folder_name = 'betas_%s_gamma_kernel'%which_kernel
		pe_sw, nope_sw = 4.0, 4.0 # which_parts used for binning
		for name, these_betas in zip(['diff_betas_%s_kernel'%which_kernel, 'zscored_diff_betas_%s_kernel'%which_kernel,'colour_%s_kernel'%which_kernel,'sound_%s_kernel'%which_kernel, 'zscored_beta_sound_%s'%which_kernel, 'zscored_beta_colour_%s' %which_kernel],
									 [ diff_beta_sound_colour, zscored_diff_beta_sound_colour, beta_colour, beta_sound, zscored_beta_sound, zscored_beta_colour]):
			f = pl.figure(figsize=(6,4)) 
			ax1 = f.add_subplot(121)
			ax1.set_title(name + '\nprediction error trials')
			pl.ylabel('beta values')
			pl.xlabel('time')			
			prpe_median_start_end_data = [these_betas[prpe_trials_mask][prpe_order][:int(prpe_trials_mask.sum()/pe_sw)], these_betas[prpe_trials_mask][prpe_order][int(pe_sw-1)*int(prpe_trials_mask.sum()/pe_sw):]]
			nrpe_median_start_end_data = [these_betas[nrpe_trials_mask][nrpe_order][:int(nrpe_trials_mask.sum()/pe_sw)], these_betas[nrpe_trials_mask][nrpe_order][int(pe_sw-1)*int(nrpe_trials_mask.sum()/pe_sw):]]
			pl.bar([0,1], map(np.median, prpe_median_start_end_data), yerr = map(np.std, prpe_median_start_end_data)/np.sqrt(prpe_trials_mask.sum()/pe_sw), color = 'r', width = 0.2, ecolor = 'k' )
			pl.bar([0.3,1.3], map(np.median, nrpe_median_start_end_data), yerr = map(np.std, nrpe_median_start_end_data)/np.sqrt(nrpe_trials_mask.sum()/pe_sw), color = 'b', width = 0.2, ecolor = 'k' )
			pl.legend(['prpe', 'nrpe'], loc = 'best')
			simpleaxis(ax1)
			spine_shift(ax1)
			pl.axhline(0, color = 'k', linewidth = 0.25)
			ax1.set_xticks([0.25, 1.25])
			ax1.set_xticklabels(['early', 'late'])

			ax2 = f.add_subplot(122, sharey=ax1)
			ax2.set_title(name + '\nno prediction error trials')
			pl.ylabel('beta values')
			pl.xlabel('time')				
			hphr_median_start_end_data = [these_betas[hphr_trials_mask][hphr_order][:int(hphr_trials_mask.sum()/nope_sw)], these_betas[hphr_trials_mask][hphr_order][int(nope_sw-1)*int(hphr_trials_mask.sum()/nope_sw):]]			
			lplr_median_start_end_data = [these_betas[lplr_trials_mask][lplr_order][:int(lplr_trials_mask.sum()/nope_sw)], these_betas[lplr_trials_mask][lplr_order][int(nope_sw-1)*int(lplr_trials_mask.sum()/nope_sw):]]
			pl.bar([0,1], map(np.median, hphr_median_start_end_data), yerr = map(np.std, hphr_median_start_end_data)/np.sqrt(hphr_trials_mask.sum()/nope_sw), color = 'r', width = 0.2, ecolor = 'k', alpha = 0.5 )
			pl.bar([0.3,1.3], map(np.median, lplr_median_start_end_data), yerr = map(np.std, lplr_median_start_end_data)/np.sqrt(lplr_trials_mask.sum()/nope_sw), color = 'b', width = 0.2, ecolor = 'k' , alpha = 0.5)
			pl.legend(['hphr', 'lplr'], loc = 'best')			
			simpleaxis(ax2)
			spine_shift(ax2)
			pl.axhline(0, color = 'k', linewidth = 0.25)
			ax2.set_xticks([0.25, 1.25])
			ax2.set_xticklabels(['early', 'late'])
			pl.tight_layout()			
			pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials + '_' + name + '_GLM_time_bar_%s_kernel.pdf' %which_kernel))			

			#save in HDF5
			with pd.get_store(self.ho.inputObject) as h5_file:
				h5_file.put("/%s/%s"%(folder_name, 'prpe_median_start_end_data_'+name), pd.DataFrame(prpe_median_start_end_data))	
				h5_file.put("/%s/%s"%(folder_name, 'nrpe_median_start_end_data_'+name), pd.DataFrame(nrpe_median_start_end_data))	
				h5_file.put("/%s/%s"%(folder_name, 'hphr_median_start_end_data_'+name), pd.DataFrame(hphr_median_start_end_data))	
				h5_file.put("/%s/%s"%(folder_name, 'lplr_median_start_end_data_'+name), pd.DataFrame(lplr_median_start_end_data))	



	def single_trial_GLM_dual_gamma_kernel_results(self, input_data_type='residuals', data_type='pupil_lp_zscore', requested_eye='L', interval=0.5, analysis_sample_rate=25): 
		"""inspect results from single_trial_GLM_dual_gamma_kernel"""
	
		#self.single_trial_GLM_dual_gamma_kernel()
		self.events_and_signals_in_time(data_type = data_type, requested_eye = requested_eye) 
		self.downsampled_pupil_and_events(data_type=data_type, requested_eye=requested_eye)

		GLM_results = np.load(self.base_directory + '/processed/%s_single_trial_GLM_dual_gamma_kernel_demeaned_regressors_%s.pickle'%(self.subject.initials, input_data_type))
		
		betas = GLM_results.params[1:] #all beta values - constant
		split_betas = np.split(betas, 4)
		beta_sound_event=np.array(split_betas[0])
		beta_colour_event=np.array(split_betas[1])
		beta_sound_reward=np.array(split_betas[2])
		beta_colour_reward=np.array(split_betas[3])

		diff_beta_sound_colour_reward = beta_sound_reward - beta_colour_reward
		diff_beta_sound_colour_event = beta_sound_event - beta_colour_event

		corr_sr_cr = sp.stats.pearsonr(beta_sound_reward, beta_colour_reward) 		
		corr_se_re = sp.stats.pearsonr(beta_sound_event, beta_colour_event) 
		
		# #scatter pupil baseline & beta values
		# f = pl.figure()
		# for i in range(len(split_betas)):			
		# 	ax = f.add_subplot(2,2,i+1)
		# 	pl.scatter(self.fix_pupil_mean, split_betas[i])
		# 	pl.title('split_betas %s'%i)
		# 	pl.ylabel('beta value')
		# 	pl.xlabel('pupil baseline')

		# #linear regression pupil baseline & beta values
		# #parameters
		# n=len(self.fix_pupil_mean)
		# x= np.copy(self.fix_pupil_mean)
		# f=pl.figure()
		# for i in range(len(split_betas)):
		# 	y= np.copy(split_betas[i])
		# 	#fitted slope and y-intercept 
		# 	(ar,br)=polyfit(x, y, 1)
		# 	xr=polyval([ar,br],x)
		# 	#mean square error 
		# 	err=sp.sqrt(sum((xr-y)**2)/n)
		# 	#plot regression for all beta conditions
		# 	ax= f.add_subplot(2,2,i+1)
		# 	pl.plot(x,y, 'ro', alpha=0.5)
		# 	pl.plot(x,xr, 'g-')
		# 	pl.title('split_betas %s'%i)
		# 	pl.xlabel('pupil baseline')
		# 	pl.ylabel('beta value')
		# 	pl.legend(['original', 'regression: r=%.2f' %ar] )
		# 	print('Linear regression using polyfit')
		# 	print('parameters: x=fix_pupil_mean y=split_betas_%s \nregression: a=%.2f b=%.2f, ms error= %.3f' % (i,ar,br,err))
		# #pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials + 'pupil_baseline_GLM_beta_correlations.pdf'))
		
		# #correlation plots of beta conditions
		# f = pl.figure()	
		# ax = f.add_subplot(211)
		# ax = sns.regplot(beta_sound_reward, beta_colour_reward)		
		# pl.title('correlation GLM beta values with reward kernel')
		# pl.xlabel('beta sound')
		# pl.ylabel('beta colour')
		# pl.legend(['r=%.2f, p=%.6f' %(corr_sr_cr[0], corr_sr_cr[1])])
		# ax = f.add_subplot(212)
		# ax = sns.regplot(beta_sound_event, beta_colour_event)	
		# pl.title('correlation GLM beta values with event kernel')
		# pl.xlabel('beta sound')
		# pl.ylabel('beta colour')
		# pl.legend(['r=%.2f, p=%.6f' %(corr_se_re[0], corr_se_re[1])])
		# #pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials + '_GLM_beta_correlations.pdf'))

		#reversal indices 
		timepoints = np.linspace(0, 85, self.trial_indices.size)
		dts_per_run = [self.domain_time[i[0]:i[1]] for i in self.run_trial_limits]
		raw_reversal_indices = np.split(self.reversal_indices, np.where(self.reversal_indices[:]== 0)[0][1:])
		cum_reversal_indices = [] 
		counter = 0 
		for i in range(len(raw_reversal_indices)):
			cum_reversal_indices.append(raw_reversal_indices[i] + counter) 
			counter = self.run_trial_limits[i][1]	

		green, purple = self.hue_indices[0], self.hue_indices[1]
		low_rw_sound, high_rw_sound = self.sound_indices[0], self.sound_indices[1]
		lp_colour, hp_colour = self.reward_prob_indices[0], self.reward_prob_indices[1]
		
		#reversal block indices and limits 
		block_indices = np.unique(np.concatenate(cum_reversal_indices)) 	
		block_limits = np.array([block_indices[:-1],block_indices[1:]]).T  	
		#conditions
		prpe_trials_mask = high_rw_sound * lp_colour 
		nrpe_trials_mask = low_rw_sound * hp_colour
		hphr_trials_mask = high_rw_sound * hp_colour 
		lplr_trials_mask = low_rw_sound * lp_colour

		block_indexed_trials = np.concatenate([np.arange(0,bl[1]-bl[0]) for bl in block_limits])

		prpe_order = np.argsort(block_indexed_trials[prpe_trials_mask]) #order trials on position in reversal block (early --> late)
		nrpe_order = np.argsort(block_indexed_trials[nrpe_trials_mask])
		hphr_order = np.argsort(block_indexed_trials[hphr_trials_mask]) 
		lplr_order = np.argsort(block_indexed_trials[lplr_trials_mask])
		
		#demeaned and standardised difference scores of reward kernel fitted colour and sound beta values 
		zscored_diff_beta_sound_colour_reward = np.concatenate([(diff_beta_sound_colour_reward[bi[0]:bi[1]]-diff_beta_sound_colour_reward[bi[0]:bi[1]].mean())/diff_beta_sound_colour_reward[bi[0]:bi[1]].std()  for bi in block_limits])
		zscored_beta_sound_reward = (beta_sound_reward - beta_sound_reward.mean())/beta_sound_reward.std() 
		zscored_beta_colour_reward = (beta_colour_reward - beta_colour_reward.mean()) / beta_colour_reward.std()
		folder_name = 'betas'
		#Reversal block bar plots using the first and last [pe_sw / nope_sw] of reversal blocks to visualise pupil responses over time 
		pe_sw, nope_sw = 4.0, 4.0 # which_parts used for binning
		for name, these_betas in zip(['diff_betas_event_kernel', 'diff_betas_reward_kernel', 'zscored_diff_betas_reward_kernel','colour_event_kernel','colour_reward_kernel','sound_event_kernel','sound_reward_kernel', 'zscored_beta_sound_reward', 'zscored_beta_colour_reward'],
									[diff_beta_sound_colour_event, diff_beta_sound_colour_reward, zscored_diff_beta_sound_colour_reward, beta_colour_event, beta_colour_reward, beta_sound_event, beta_sound_reward, zscored_beta_sound_reward, zscored_beta_colour_reward]):
			f = pl.figure(figsize=(6,4)) 
			ax1 = f.add_subplot(121)
			ax1.set_title(name + '\nprediction error trials')
			pl.ylabel('beta values')
			pl.xlabel('time')			
			prpe_median_start_end_data = [these_betas[prpe_trials_mask][prpe_order][:int(prpe_trials_mask.sum()/pe_sw)], these_betas[prpe_trials_mask][prpe_order][int(pe_sw-1)*int(prpe_trials_mask.sum()/pe_sw):]]
			nrpe_median_start_end_data = [these_betas[nrpe_trials_mask][nrpe_order][:int(nrpe_trials_mask.sum()/pe_sw)], these_betas[nrpe_trials_mask][nrpe_order][int(pe_sw-1)*int(nrpe_trials_mask.sum()/pe_sw):]]
			pl.bar([0,1], map(np.median, prpe_median_start_end_data), yerr = map(np.std, prpe_median_start_end_data)/np.sqrt(prpe_trials_mask.sum()/pe_sw), color = 'r', width = 0.2, ecolor = 'k' )
			pl.bar([0.3,1.3], map(np.median, nrpe_median_start_end_data), yerr = map(np.std, nrpe_median_start_end_data)/np.sqrt(nrpe_trials_mask.sum()/pe_sw), color = 'b', width = 0.2, ecolor = 'k' )
			pl.legend(['prpe', 'nrpe'], loc = 'best')
			simpleaxis(ax1)
			spine_shift(ax1)
			pl.axhline(0, color = 'k', linewidth = 0.25)
			ax1.set_xticks([0.25, 1.25])
			ax1.set_xticklabels(['early', 'late'])

			ax2 = f.add_subplot(122, sharey=ax1)
			ax2.set_title(name + '\nno prediction error trials')
			pl.ylabel('beta values')
			pl.xlabel('time')				
			hphr_median_start_end_data = [these_betas[hphr_trials_mask][hphr_order][:int(hphr_trials_mask.sum()/nope_sw)], these_betas[hphr_trials_mask][hphr_order][int(nope_sw-1)*int(hphr_trials_mask.sum()/nope_sw):]]			
			lplr_median_start_end_data = [these_betas[lplr_trials_mask][lplr_order][:int(lplr_trials_mask.sum()/nope_sw)], these_betas[lplr_trials_mask][lplr_order][int(nope_sw-1)*int(lplr_trials_mask.sum()/nope_sw):]]
			pl.bar([0,1], map(np.median, hphr_median_start_end_data), yerr = map(np.std, hphr_median_start_end_data)/np.sqrt(hphr_trials_mask.sum()/nope_sw), color = 'r', width = 0.2, ecolor = 'k', alpha = 0.5 )
			pl.bar([0.3,1.3], map(np.median, lplr_median_start_end_data), yerr = map(np.std, lplr_median_start_end_data)/np.sqrt(lplr_trials_mask.sum()/nope_sw), color = 'b', width = 0.2, ecolor = 'k' , alpha = 0.5)
			pl.legend(['hphr', 'lplr'], loc = 'best')			
			simpleaxis(ax2)
			spine_shift(ax2)
			pl.axhline(0, color = 'k', linewidth = 0.25)
			ax2.set_xticks([0.25, 1.25])
			ax2.set_xticklabels(['early', 'late'])
			pl.tight_layout()
			#pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials + '_' + name + '_GLM_time_bar.pdf'))
			with pd.get_store(self.ho.inputObject) as h5_file:
				h5_file.put("/%s/%s"%(folder_name, 'prpe_median_start_end_data_'+name), pd.DataFrame(prpe_median_start_end_data))	
				h5_file.put("/%s/%s"%(folder_name, 'nrpe_median_start_end_data_'+name), pd.DataFrame(nrpe_median_start_end_data))	
				h5_file.put("/%s/%s"%(folder_name, 'hphr_median_start_end_data_'+name), pd.DataFrame(hphr_median_start_end_data))	
				h5_file.put("/%s/%s"%(folder_name, 'lplr_median_start_end_data_'+name), pd.DataFrame(lplr_median_start_end_data))	




	# def deconvolve_reward_probabilities(self, analysis_sample_rate=10, interval=[-0.5, 4.5], data_type='pupil_bp_zscore', requested_eye='L', use_domain = 'first', standard_deconvolution = 'nuisance', microsaccades_added=False):
	# 	"""deconvolve on resididuals deconvolve_diff_colours to evaluate reward prediction effects"""
		
	# 	labeling_schema = ['LP_NR', 'LP_HR', 'HP_NR', 'HP_HR']

	# 	if not hasattr(self, 'pupil_data'): # we assume that we'll grab the same events and data whatever the present deconvolve_reward_probabilities method does
	# 		self.events_and_signals_in_time(data_type = data_type, requested_eye= requested_eye, microsaccades_added=microsaccades_added)
		
	# 	#residuals standard deconvolution analysis with or without eye jitter regressor
	# 	if standard_deconvolution == 'basic': 
	# 		data_folder = 'deconvolve_colour_sound_no_eye_jitter_microsaccades_%s' %str(microsaccades_added)
	# 	elif standard_deconvolution == 'nuisance': 
	# 		data_folder = 'deconvolve_colour_sound_eye_jitter_microsaccades_%s' %str(microsaccades_added)

	# 	with pd.get_store(self.ho.inputObject) as h5_file:
	# 		try:
	# 			#residuals_diff_cols = h5_file.get("/%s/%s"%('deconvolve_diff_colours_full_domain', 'residuals'))
	# 			#residuals_deconvolve_colour_sound = h5_file.get("/%s/%s"%('deconvolve_colour_sound', 'residuals'))
	# 			residuals_deconvolve_colour_sound = h5_file.get("/%s/%s"%(data_folder, 'residuals'))					
	# 		except (IOError, NoSuchNodeError):
	# 			self.logger.error("no residuals present")


	# 	#domains that can be analysed
	# 	if use_domain == 'second':
	# 		domain_indices_used_now = -self.domain_indices
	# 	elif use_domain == 'first':
	# 		domain_indices_used_now = self.domain_indices
	# 	elif use_domain == 'full':
	# 		domain_indices_used_now = np.ones(self.domain_indices.shape, dtype = bool)

	# 	event_indices = [
	# 		self.reward_prob_indices[0] * self.sound_indices[0] * domain_indices_used_now,
	# 		self.reward_prob_indices[0] * self.sound_indices[1] * domain_indices_used_now,
	# 		self.reward_prob_indices[1] * self.sound_indices[0] * domain_indices_used_now,
	# 		self.reward_prob_indices[1] * self.sound_indices[1] * domain_indices_used_now,
	# 	]
	# 	events = [self.sound_times[ev_i] + interval[0] for ev_i in event_indices]
		
	# 	input_signal = sp.signal.decimate(self.pupil_data, int(self.sample_rate / analysis_sample_rate))	

	# 	self.logger.info('starting reward pupil deconvolution with data of type %s and sample_rate of %i Hz in the interval %s. Currently the %s domain is analysed using residuals of the %s standard deconvolution' % (data_type, analysis_sample_rate, str(interval), use_domain, standard_deconvolution))

	# 	do = ArrayOperator.DeconvolutionOperator( inputObject = np.array(residuals_deconvolve_colour_sound),
	# 						eventObject = events, TR = 1.0/analysis_sample_rate, deconvolutionSampleDuration = 1.0/analysis_sample_rate, deconvolutionInterval = interval[1] - interval[0], run = True )
	# 	time_points = np.linspace(interval[0], interval[1], np.squeeze(do.deconvolvedTimeCoursesPerEventType).shape[1])
	# 	do.residuals()

	# 	#compare rsquared after using residuals of different basic deconvolutionn
	# 	rsquared_full_input_signal = 1.0 -(np.sum(np.array(do.residuals)**2) / np.sum(input_signal**2))
	# 	rsquared_on_residuals = 1.0 -(np.sum(np.array(do.residuals)**2) / np.sum(residuals_deconvolve_colour_sound**2))
	
	# 	self.logger.info('after reward deconvolution on %s deconvolution %i Hz sampling rate, microsaccades_added= %s, explained variance (on full input_signal)(r^sq) = %1.4f'%(standard_deconvolution, analysis_sample_rate, str(microsaccades_added), rsquared_full_input_signal))
	# 	self.logger.info('after reward deconvolution on %s deconvolution, %i Hz sampling rate, microsaccades_added = %s, explained variance (on residuals of input_signal)(r^sq) = %1.4f'%(standard_deconvolution, analysis_sample_rate, str(microsaccades_added), rsquared_on_residuals))
		

	# 	timecourses = np.squeeze(do.deconvolvedTimeCoursesPerEventType)
	# 	new_event_timecourses = np.r_[[timecourses[1] - timecourses[0]], [timecourses[2] - timecourses[3]]]
			
	# 	#baseline correction
	# 	baseline_times = time_points < 0 
	# 	corrected_timecourses = np.array([timecourses[i,:] - timecourses[i,baseline_times].mean() for i in range(timecourses.shape[0])]) #.transpose(1,0)
	# 	predicted_timecourses = np.mean(np.array([corrected_timecourses[0,:], corrected_timecourses[3,:]]), axis=0)
	# 	#corrected_new_event_timecourses = np.r_[[corrected_timecourses[1] - predicted_timecourses], [corrected_timecourses[2] - predicted_timecourses]]
	# 	corrected_new_event_timecourses = np.r_[[corrected_timecourses[1] - corrected_timecourses[3], corrected_timecourses[2]-corrected_timecourses[0]]]
		
	# 	#plot probabillities 
	# 	f = pl.figure()
	# 	ax = f.add_subplot(211)
	# 	colours = ['b','b','r','r']
	# 	alphas = np.array([0.5,1,0.5,1])
	# 	lws = np.ones(len(events)) * 2.5 
	# 	for x in range(len(events)):
	# 		pl.plot(time_points, corrected_timecourses[x], colours[x], alpha = alphas[x], linewidth = lws[x]) 
	# 	pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
	# 	pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
	# 	ax.set_xlim(xmin=interval[0], xmax=interval[1])	
	# 	ax.set_ylim(ymin=-0.5, ymax=0.5)	
	# 	simpleaxis(ax)
	# 	spine_shift(ax)
	# 	pl.legend(labeling_schema, bbox_to_anchor=(1.07, 1.1))
	# 	pl.title('Reward probability events (%s, %s)'%(use_domain, standard_deconvolution))
		
	# 	ax = f.add_subplot(212)
	# 	colours = ['g','r']
	# 	alphas = np.ones(len(corrected_new_event_timecourses)) * 1.0 
	# 	lws = np.ones(len(corrected_new_event_timecourses)) * 2.5 
	# 	for x in range(len(corrected_new_event_timecourses)):
	# 		pl.plot(time_points, corrected_new_event_timecourses[x], colours[x], alpha = alphas[x], linewidth = lws[x])
	# 	pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
	# 	pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
	# 	ax.set_xlim(xmin=interval[0], xmax=interval[1])
	# 	ax.set_ylim(ymin = -0.5, ymax = 0.5)
	# 	pl.legend(['PRPE', 'NRPE'])
	# 	pl.title('Positive and Negative Reward Prediction Error events (%s, %s) ' %(use_domain, standard_deconvolution))
	# 	simpleaxis(ax)
	# 	spine_shift(ax)
	# 	pl.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials + '%s_%s_deconv_reward_probabilities_prediction_baseline_corrected_%i_Hz.pdf'%(use_domain, standard_deconvolution, analysis_sample_rate)))

	# 	folder_name = 'deconvolve_reward_probabilities_%s_domain_%s_%s'%(use_domain, standard_deconvolution, str(microsaccades_added)) 
	# 	#store deconvolution 
	# 	with pd.get_store(self.ho.inputObject) as h5_file:
	# 		h5_file.put("/%s/%s"%(folder_name, 'residuals'), pd.Series(np.squeeze(np.array(do.residuals))))
	# 		h5_file.put("/%s/%s"%(folder_name, 'time_points'), pd.Series(time_points))
	# 		h5_file.put("/%s/%s"%(folder_name, 'dec_time_course'), pd.DataFrame(np.squeeze(do.deconvolvedTimeCoursesPerEventType).T))
	# 		h5_file.put("/%s/%s"%(folder_name, 'rsquared'), pd.Series([rsquared_full_input_signal, rsquared_on_residuals], index = ['rsquared_full_input_signal_%s'%standard_deconvolution, 'rsquared_on_residuals_%s'%standard_deconvolution]))
