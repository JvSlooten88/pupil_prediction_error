from __future__ import division

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
import matplotlib.pyplot as pl
import pandas as pd
import numpy.linalg as LA
import bottleneck as bn
import glob
import seaborn as sn
import statsmodels.stats as sm 
import statsmodels.api as sma
import matplotlib
import matplotlib.pyplot as pl

import rpy2.robjects as robjects
import rpy2.rlike.container as rlc

import mne 

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

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

from IPython import embed as shell

class ReversalLearningGroupLevel(object): 
	"""Instances of ReversalLearningGroupLevel can be used to do group analyses on ReversalLearningSession experiments"""

	def __init__ (self, sessions, data_folder, exp_name, loggingLevel = logging.DEBUG, downsample_rate = 25):
		self.experiment_name = exp_name 
		self.sessions = sessions 
		self.data_dir = data_folder     
		self.plot_dir = os.path.join(self.data_dir, self.experiment_name, 'group_level', 'figs')
		try: 
				os.mkdir(os.path.join(data_folder, self.experiment_name, 'group_level' ))
		except OSError: 
				pass 
		try: 
				os.mkdir(os.path.join(data_folder, self.experiment_name, 'group_level', 'data'))
		except OSError: 
				pass 
		try: 
				os.mkdir(os.path.join(data_folder, self.experiment_name, 'group_level', 'figs'))
		except OSError: 
				pass 
		try: 
				os.mkdir(os.path.join(data_folder, self.experiment_name, 'group_level', 'log'))
		except OSError: 
				pass 
		self.grouplvl_data_dir = os.path.join(data_folder, self.experiment_name, 'group_level' , 'data')
		self.grouplvl_plot_dir = os.path.join(data_folder, self.experiment_name, 'group_level', 'figs')
		self.grouplvl_log_dir = os.path.join(data_folder, self.experiment_name, 'group_level', 'log')
		self.hdf5_filename = os.path.join(self.grouplvl_data_dir, 'all_data.hdf5')

		# add logging for this session
		# sessions create their own logging file handler
		self.loggingLevel = loggingLevel
		self.logger = logging.getLogger( self.__class__.__name__ )
		self.logger.setLevel(self.loggingLevel)
		addLoggingHandler( logging.handlers.TimedRotatingFileHandler( os.path.join(self.grouplvl_log_dir, 'sessionLogFile.log'), when = 'H', delay = 2, backupCount = 10), loggingLevel = self.loggingLevel )
		loggingLevelSetup()
		for handler in logging_handlers:
			self.logger.addHandler(handler)
		self.logger.info('starting analysis in ' + self.grouplvl_log_dir)

		self.downsample_rate = downsample_rate
		self.new_sample_rate = 1000 / self.downsample_rate


	def gather_data_from_hdfs(self, group = 'deconvolve_colour_sound', data_type = 'time_points'):
		"""gather_data_from_hdfs takes arbitrary data from hdf5 files for all self.sessions.
		arguments:  group - the folder in the hdf5 file from which to take data
					data_type - the type of data to be read.
		returns a numpy array with all the data.
		"""
		gathered_data = [] 
		for s in self.sessions:
			with pd.get_store(s.hdf5_filename) as h5_file:					
				gathered_data.append(np.array(h5_file.get("/%s/%s"%(group, data_type))))			
		return np.array(gathered_data)
	  

	def gather_dataframes_from_hdfs(self, group = 'deconvolve_colour_sound', data_type = 'time_points'): 
		"""gather_data_from_hdfs takes group/datatype data from hdf5 files for all self.sessions.
		arguments:  group - the folder in the hdf5 file from which to take data
					data_type - the type of data to be read.
		returns a pd.dataFrame with a hierarchical index, meaning that you can access a specific self.session using 
		its key. Keys are converted from strings to numbers
		"""
				
		gathered_dataframes = [] 		
		for s in self.sessions: 
			print s.subject.initials
			with pd.get_store(s.hdf5_filename) as h5_file:					
				gathered_dataframes.append(pd.DataFrame(h5_file.get("/%s/%s"%(group, data_type))))	
		return gathered_dataframes


	def gather_data_from_pickles(self, data_types = ['aic', 'bic', 'rsquared'], which_kernel = 'event'): #group='processed'
		"""gather_data_from_pickles takes arbitrary data from pickle files for all self.sessions.""" 

		
		gathered_data = []      
		for s in self.sessions:
			with open('/home/shared/reward_pupil/3/data/reward_prediction_error/' + s.subject.initials + '/processed/' + s.subject.initials + '_single_trial_GLM_%s_gamma_kernel_demeaned_regressors_residuals.pickle'%which_kernel) as f: 
				this_session_GLM_results = pickle.load(f)
				exported_results = [eval('this_session_GLM_results.'+dt) for dt in data_types]
				gathered_data.append(exported_results)
		return np.array(gathered_data)

	def gather_data_from_npzs(self, data_type = 'ks_pvals_log'): 
		"""gather_data_from_npzs takes arbitrary data from npz files for all self.sessions.""" 
		
		path = os.path.join(self.data_dir, self.experiment_name) 
		gathered_data = []
		
		for s in self.sessions: 
			with open(os.path.join(path, s.subject.initials) + '/processed/' + '%s'%data_type) as f: 
				this_session_simulations = np.load(f)
				this_session_simulations = this_session_simulations['arr_0']
				gathered_data.append(this_session_simulations)
		return np.array(gathered_data)
		 

	def gather_blink_rates(self):

		names = [s.subject.initials for s in self.sessions]
		gathered_blink_rates=self.gather_data_from_hdfs(group = 'blink_rate', data_type = 'blink_rate')
		df = pd.DataFrame(dict(name=names, blinks=gathered_blink_rates))
		f = pl.figure(figsize=(20,10))
		ax = sn.violinplot(df.blinks, inner='box', names=df.name)
		pl.axhline(15, lw=1, alpha=0.5, color='k')
		simpleaxis(ax)
		pl.title('Average blink rate per minute')
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'averaged_blink_rates_bad_blocks_excluded.pdf')) 
		


	def basic_deconv_results_across_subjects(self, group = 'deconvolve_colour_sound', microsaccades_added=False):
		"""basic_deconv_results_across_subjects takes timepoints from deconvolve_colour_sound, and averages across subjects.
		"""
		folder_name = 'deconvolve_colour_sound_eye_jitter_microsaccades_%s' %str(microsaccades_added)

		dec_time_courses = self.gather_data_from_hdfs(group = folder_name, data_type = 'dec_time_course')
		time_points = self.gather_data_from_hdfs(group = folder_name, data_type = 'time_points')
		nuisance = self.gather_data_from_hdfs(group = folder_name, data_type = 'nuisance_betas').transpose(0,2,1)
		
		color= dict(Purple="purple")
		conds = pd.Series(['blink', 'color', 'sound'])
		conds2 = pd.Series(['eye jitter'])
		cis = np.linspace(95, 10, 4)

		sn.set(style="ticks")
		f = pl.figure(figsize = (12,6))
		ax = sn.tsplot(dec_time_courses[:,:,-4:], err_style="ci_band", condition=conds, time = time_points.mean(axis = 0)) 
		ax = sn.tsplot(nuisance[:,:,:], err_style="ci_band", condition = conds2, time=time_points.mean(axis=0), color="purple")
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.ylabel('pupil size')
		pl.xlabel('time (s)')
		sn.despine(offset=10, trim=True)
		simpleaxis(ax)
		pl.tight_layout()
		pl.savefig(os.path.join(self.grouplvl_plot_dir, '_group_%s.pdf'%folder_name))


	def deconv_standard_keypress_across_subjects(self, group = 'deconvolve_colour_sound', microsaccades_added=False):
		"""basic_deconv_results_across_subjects takes timepoints from deconvolve_colour_sound, and averages across subjects.
		"""
		folder_name = 'standard_deconvolve_keypress_%s'%str(microsaccades_added)
		
		dec_time_courses = self.gather_data_from_hdfs(group = folder_name, data_type = 'dec_time_course')
		time_points = self.gather_data_from_hdfs(group = folder_name, data_type = 'time_points')
		rsquared = np.mean(self.gather_data_from_hdfs(group = folder_name, data_type = 'rsquared'), axis=0) 
		analysis_sample_rate = len(time_points[0])/np.abs(time_points[0][0] - time_points[0][-1])
		
		conds = pd.Series(['blink', 'color', 'sound', 'keypress'])
		
		sn.set(style="ticks")
		f = pl.figure(figsize = (12,6))
		ax = sn.tsplot(dec_time_courses[:,:,:], err_style="ci_band", condition=conds, time = time_points.mean(axis = 0)) 
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.ylabel('pupil size')
		pl.xlabel('time (s)')
		pl.title('standard deconvolution (%i Hz)'%analysis_sample_rate)
		pl.text(0,0.05, 'rsquared: %.3f'%rsquared)
		sn.despine(offset=10, trim=True)
		simpleaxis(ax)
		pl.tight_layout()
		pl.savefig(os.path.join(self.grouplvl_plot_dir, '%s_group_%i_hz.pdf'%(folder_name, analysis_sample_rate)))


	def deconv_diff_colours_across_subjects(self, use_domain='full'):
		"""diff_colours_across_subjects takes timepoints from 'deconvolve_diff_colours' and averages across subjects.
		"""
		dec_time_courses = self.gather_data_from_hdfs(group = 'deconvolve_diff_colours_%s_domain'%use_domain, data_type = 'dec_time_course')
		time_points = self.gather_data_from_hdfs(group = 'deconvolve_diff_colours_%s_domain'%use_domain, data_type = 'time_points')
		baseline_times = time_points.mean(axis = 0) < 0  #take baseline from timepoints interval 0 (=-0.5)

		dec_time_courses_s = np.array([[myfuncs.smooth(dec_time_courses[i,:,j], window_len=20) - dec_time_courses[i,baseline_times,j].mean() for i in range(dec_time_courses.shape[0])] for j in range(dec_time_courses.shape[-1])]).transpose((1,2,0))
		dec_time_courses_diff = np.array([dec_time_courses_s[:,:,0] - dec_time_courses_s[:,:,1]]).transpose(1,2,0)  #timepoints, events, id
		
		conds = pd.Series(['Green', 'Purple'])
		conds_diff = pd.Series(['Green - Purple'])

		cis = np.linspace(95, 10, 4)

		sn.set(style="ticks")
		color_map = dict(Green="darkseagreen", Purple="purple")
		f = pl.figure()
		ax = f.add_subplot(211)
		ax = sn.tsplot(dec_time_courses_s, err_style="ci_band",  condition = conds, time = time_points.mean(axis = 0), color=color_map ) 
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10, trim=True)
		pl.title('Effect of cue color on pupil dilation (%s_domain)' %use_domain)

		ax= f.add_subplot(212)
		ax = sn.tsplot(dec_time_courses_diff, err_style="ci_band",  condition=conds_diff, time=time_points.mean(axis=0))
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10, trim=True)
		pl.title('Cue color difference score on pupil dilation (%s_domain)'%use_domain)
		pl.savefig(os.path.join(self.grouplvl_plot_dir, '%s_domain_deconv_cue_colours_across_subjects.pdf'%use_domain))


	def deconvolve_full_across_subjects(self, use_domain='full', baseline=True):
		"""deconvolve_full_across_subjects averages deconvolution "deconvolve_full"         
		regressor:               column:
		blink                    [0]
		keypress                 [1]
		saccade                  [2]
		colour                   [3]
		cue low, high            [4][5]
		sound                    [6]
		LPNR, LPHR, HPNR, HPHR   [7][8][9][10]
		""" 
		
		dec_time_courses = self.gather_data_from_hdfs(group = 'deconvolve_full_%s_domain'%use_domain, data_type = 'dec_time_course')
		time_points = self.gather_data_from_hdfs(group = 'deconvolve_full_%s_domain'%use_domain, data_type = 'time_points')
		baseline_times = time_points.mean(axis = 0) < 0  #take baseline from timepoints interval [-0.5 - 0] 

		if baseline: 
			dec_time_courses_s = np.array([[myfuncs.smooth(dec_time_courses[i,:,j], window_len=10) - dec_time_courses[i,baseline_times,j].mean() for i in range(dec_time_courses.shape[0])] for j in range(dec_time_courses.shape[-1])]).transpose((1,2,0))
		else: 
			dec_time_courses_s = np.array([[myfuncs.smooth(dec_time_courses[i,:,j], window_len=10) for i in range(dec_time_courses.shape[0])] for j in range(dec_time_courses.shape[-1])]).transpose((1,2,0))
	
		
		dec_time_courses_nuisance = np.array([dec_time_courses_s[:,:,0], dec_time_courses_s[:,:,1], dec_time_courses_s[:,:,2]]).transpose((1,2,0))
		dec_time_courses_colour_sound = np.array([dec_time_courses_s[:,:,3], dec_time_courses_s[:,:,6]]).transpose((1,2,0))
		dec_time_courses_cue = np.array([dec_time_courses_s[:,:,4], dec_time_courses_s[:,:,5]]).transpose((1,2,0))
		dec_time_courses_reward = np.array([dec_time_courses_s[:,:,7], dec_time_courses_s[:,:,8], dec_time_courses_s[:,:,9], dec_time_courses_s[:,:,10]]).transpose((1,2,0))
		dec_time_courses_rpe = np.array([dec_time_courses_s[:,:,8] - dec_time_courses_s[:,:,10], dec_time_courses_s[:,:,9] - dec_time_courses_s[:,:,7]]).transpose((1,2,0))
		dec_time_courses_pe = np.array([dec_time_courses_s[:,:,10] - dec_time_courses_s[:,:,7], dec_time_courses_s[:,:,8] - dec_time_courses_s[:,:,9]]).transpose((1,2,0))
			  
		 
		conds_nuisance = pd.Series(['blink', 'keypress', 'saccade'])
		conds_colour_sound = pd.Series(['colour', 'sound'])
		conds_cue = pd.Series(['LP cue', 'HP cue'])
		conds_reward = pd.Series(['LPNR', 'LPHR', 'HPNR', 'HPHR'])
		conds_rpe = pd.Series(['Positive RPE', 'Negative RPE'])
		conds_pe = pd.Series(['Prediction', 'Prediction error'])
		
		sn.set(style="ticks")
		f = pl.figure()
		ax = f.add_subplot(321)
		ax = sn.tsplot(dec_time_courses_nuisance, err_style="ci_band",  condition = conds_nuisance, time = time_points.mean(axis = 0)) 
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10, trim=True)
		pl.tight_layout()
		pl.title('Effect of blink, keypress and saccade (%s)'%use_domain)
		pl.ylabel('Z')
		ax = f.add_subplot(322)
		ax = sn.tsplot(dec_time_courses_colour_sound, err_style="ci_band",  condition=conds_colour_sound, time=time_points.mean(axis=0))
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		ax.set_ylim([-0.6, 0.6])
		sn.despine(offset=10, trim=True)
		pl.tight_layout()
		pl.title('Effect of colour and sound (%s)'%use_domain)
		pl.ylabel('Z')
		ax = f.add_subplot(323)
		ax = sn.tsplot(dec_time_courses_cue, err_style="ci_band",  condition=conds_cue, time=time_points.mean(axis=0))
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		ax.set_ylim([-0.15, 0.15])
		sn.despine(offset=10, trim=True)
		pl.tight_layout()
		pl.title('Effect of cue (%s)'%use_domain)
		pl.ylabel('Z')
		ax = f.add_subplot(324)
		ax = sn.tsplot(dec_time_courses_reward, err_style="ci_band",  condition=conds_reward, time=time_points.mean(axis=0))
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		ax.set_ylim([-0.2, 0.4])
		sn.despine(offset=10, trim=True)
		pl.tight_layout()
		pl.title('Effect of reward outcome (%s)'%use_domain)
		pl.ylabel('Z')
		ax = f.add_subplot(325)
		ax = sn.tsplot(dec_time_courses_rpe, err_style="ci_band",  condition=conds_rpe, time=time_points.mean(axis=0))
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		ax.set_ylim([-0.2, 0.4])
		sn.despine(offset=10, trim=True)
		pl.tight_layout()
		pl.title('Effect of reward prediction error (%s)'%use_domain)
		pl.ylabel('Z')
		ax = f.add_subplot(326)
		ax = sn.tsplot(dec_time_courses_pe, err_style="ci_band",  condition=conds_pe, time=time_points.mean(axis=0))
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10, trim=True)
		ax.set_ylim([-0.35, 0.2])
		pl.tight_layout()
		pl.title('Effect of prediction error (%s)'%use_domain)
		pl.ylabel('Z')
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'full_deconvolution_across_subjects_colour_sound_regressed_out_%s_domain_baseline_%s.pdf'%(use_domain, str(baseline))))
  
	def deconvolve_full_FIR_across_subjects(self, baseline=True, use_domain='full'): 
		"""FIR deconvolution ridge regression across subjects """

		folder_name = 'deconvolve_full_FIR_%s_domain'%use_domain
		time_points = self.gather_data_from_hdfs(group = folder_name, data_type = 'time_points')
		prediction_error_kernels = self.gather_data_from_hdfs(group = folder_name, data_type = 'prediction_error_kernel')
		baseline_times = time_points.mean(axis = 0) < 0 
	
		sj_deconvolved_timecourses_lst = self.gather_dataframes_from_hdfs(group = folder_name, data_type = 'deconvolved_pupil_timecourses')
		sj_data_lpnr_array = np.array([np.array(p['LPNR.gain'], dtype = float) for p in sj_deconvolved_timecourses_lst])
		sj_data_lphr_array = np.array([np.array(p['LPHR.gain'], dtype = float) for p in sj_deconvolved_timecourses_lst])
		sj_data_hpnr_array = np.array([np.array(p['HPNR.gain'], dtype = float) for p in sj_deconvolved_timecourses_lst])
		sj_data_hphr_array = np.array([np.array(p['HPHR.gain'], dtype = float) for p in sj_deconvolved_timecourses_lst])
		sj_data_sound_array = np.array([np.array(p['sound.gain'], dtype = float) for p in sj_deconvolved_timecourses_lst])

		if baseline: 
			prediction_error_kernel_s = np.array([myfuncs.smooth(prediction_error_kernels[i,:], window_len=10) - prediction_error_kernels[i,baseline_times].mean() for i in range(prediction_error_kernels.shape[0])])
		else: 
			prediction_error_kernel_s = np.array([myfuncs.smooth(prediction_error_kernels[i,:], window_len=10) for i in range(prediction_error_kernels.shape[0])])		
		
		#save group average prediction error kernel 
		np.save(os.path.join(self.grouplvl_data_dir, 'prediction_error_kernel.npy'), np.mean(prediction_error_kernel_s, axis=0))

		f = pl.figure() 
		sn.tsplot(prediction_error_kernel_s, time=time_points.mean(axis=0))
		pl.legend('prediction_error kernel')
		pl.title('unsigned prediction error kernel across subjects', fontsize=10)
		pl.ylabel('pupil size (Z)', fontsize=9)
		pl.xlabel('time (s)', fontsize=9)
		sn.despine(offset=10)
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.tight_layout()
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'unsigned_prediction_error_FIR_%s_domain_baseline_%s.pdf'%(use_domain, str(baseline))))



	def compare_first_last_domain_deconvolve_full(self, baseline=True, use_domain ='first_third_domain'): 
		""" compare_first_last_domain_deconvolve_full calculates the difference between deconvolved reward conditions from the first and last part of a reversal block 
		to inspect how pupil dilation changes as a function of learning the reward probabilities. 
		regressor:               column:
		blink                    [0]
		keypress                 [1]
		saccade                  [2]
		colour                   [3]
		cue low, high            [4][5]
		sound                    [6]
		LPNR, LPHR, HPNR, HPHR   [7][8][9][10]
		"""
		
		if use_domain == 'first_third_domain': 
			dec_time_courses_first_domain = self.gather_data_from_hdfs(group = 'deconvolve_full_first_3split_domain', data_type = 'dec_time_course')
			dec_time_courses_last_domain = self.gather_data_from_hdfs(group = 'deconvolve_full_third_3split_domain', data_type = 'dec_time_course')
		elif use_domain == 'first_second_domain': 
			dec_time_courses_first_domain = self.gather_data_from_hdfs(group = 'deconvolve_full_first_domain', data_type = 'dec_time_course')
			dec_time_courses_last_domain = self.gather_data_from_hdfs(group = 'deconvolve_full_second_domain', data_type = 'dec_time_course')

		time_points = self.gather_data_from_hdfs(group = 'deconvolve_full_first_3split_domain', data_type = 'time_points')
		baseline_times = time_points.mean(axis = 0) < 0  #take baseline from timepoints interval [-0.5 - 0] 

		if baseline: 
			dec_time_courses_first_domain_s = np.array([[myfuncs.smooth(dec_time_courses_first_domain[i,:,j], window_len=10) - dec_time_courses_first_domain[i,baseline_times,j].mean() for i in range(dec_time_courses_first_domain.shape[0])] for j in range(dec_time_courses_first_domain.shape[-1])]).transpose((1,2,0))
			dec_time_courses_last_domain_s = np.array([[myfuncs.smooth(dec_time_courses_last_domain[i,:,j], window_len=10) - dec_time_courses_last_domain[i,baseline_times,j].mean() for i in range(dec_time_courses_last_domain.shape[0])] for j in range(dec_time_courses_last_domain.shape[-1])]).transpose((1,2,0))
		else: 
			dec_time_courses_first_domain_s = np.array([[myfuncs.smooth(dec_time_courses_first_domain[i,:,j], window_len=10) for i in range(dec_time_courses_first_domain.shape[0])] for j in range(dec_time_courses_first_domain.shape[-1])]).transpose((1,2,0))
			dec_time_courses_last_domain_s = np.array([[myfuncs.smooth(dec_time_courses_last_domain[i,:,j], window_len=10) for i in range(dec_time_courses_last_domain.shape[0])] for j in range(dec_time_courses_last_domain.shape[-1])]).transpose((1,2,0))

		#reward timecourses 
		dec_time_courses_reward_first_s = np.array([dec_time_courses_first_domain_s[:,:,7], dec_time_courses_first_domain_s[:,:,8], dec_time_courses_first_domain_s[:,:,9], dec_time_courses_first_domain_s[:,:,10]]).transpose((1,2,0))
		dec_time_courses_reward_last_s = np.array([dec_time_courses_last_domain_s[:,:,7], dec_time_courses_last_domain_s[:,:,8], dec_time_courses_last_domain_s[:,:,9], dec_time_courses_last_domain_s[:,:,10]]).transpose((1,2,0))

		diff_deconvolved_reward_time_courses_first_last = np.array([dec_time_courses_reward_last_s[:,:,0] - dec_time_courses_reward_first_s[:,:,0],  dec_time_courses_reward_last_s[:,:,1] - dec_time_courses_reward_first_s[:,:,1], dec_time_courses_reward_last_s[:,:,2] - dec_time_courses_reward_first_s[:,:,2], dec_time_courses_reward_last_s[:,:,3] - dec_time_courses_reward_first_s[:,:,3]]).transpose((1,2,0))

		#positive and negative reward prediction error timecourses for first, third, and different domains
		prpe_npre_first = np.array([dec_time_courses_reward_first_s[:,:,1] - dec_time_courses_reward_first_s[:,:,3], dec_time_courses_reward_first_s[:,:,2] - dec_time_courses_reward_first_s[:,:,0]]).transpose((1,2,0))
		prpe_npre_last = np.array([dec_time_courses_reward_last_s[:,:,1] - dec_time_courses_reward_last_s[:,:,3], dec_time_courses_reward_last_s[:,:,2] - dec_time_courses_reward_last_s[:,:,0]]).transpose((1,2,0))		
		diff_prpe_npre = np.array([diff_deconvolved_reward_time_courses_first_last[:,:,1] - diff_deconvolved_reward_time_courses_first_last[:,:,3], diff_deconvolved_reward_time_courses_first_last[:,:,2] - diff_deconvolved_reward_time_courses_first_last[:,:,0]]).transpose((1,2,0))
		diff_of_diff_rpes = diff_prpe_npre[:,:,0] - diff_prpe_npre[:,:,1]
		
		shell() 
		#unsigned prediction error kernel 
		unsigned_prediction_error_first = np.mean(prpe_npre_first, axis=2)
		unsigned_prediction_error_last = np.mean(prpe_npre_last, axis=2)
		unsigned_prediction_error = np.mean(np.r_[[unsigned_prediction_error_first, unsigned_prediction_error_last]], axis=0)
		np.save(os.path.join(self.grouplvl_data_dir, 'prediction_error_kernel.npy'), unsigned_prediction_error)

		### effect of reward and loss 
		reward_first = np.array([dec_time_courses_reward_first_s[:,:,1] - dec_time_courses_reward_first_s[:,:,3], dec_time_courses_reward_first_s[:,:,2] - dec_time_courses_reward_first_s[:,:,0]]).transpose((1,2,0))

		#permutation testing of pupil responses (4 reward conditions)
		clusters_pval_diff, sig_timepoints_diff = self.permutation_testing_of_deconvolved_responses(diff_deconvolved_reward_time_courses_first_last, time_points = time_points.mean(axis=0))
		clusters_pval_first, sig_timepoints_first = self.permutation_testing_of_deconvolved_responses(dec_time_courses_reward_first_s, time_points = time_points.mean(axis=0))
		clusters_pval_last, sig_timepoints_last = self.permutation_testing_of_deconvolved_responses(dec_time_courses_reward_last_s, time_points = time_points.mean(axis=0))

		#permutation testing of PRPE NRPE conditions 
		clusters_pval_rpe_first, sig_timepoints_rpe_first = self.permutation_testing_of_deconvolved_responses(prpe_npre_first, time_points = time_points.mean(axis=0))
		clusters_pval_rpe_last, sig_timepoints_rpe_last = self.permutation_testing_of_deconvolved_responses(prpe_npre_last, time_points = time_points.mean(axis=0))
		clusters_pval_rpe_diff, sig_timepoints_rpe_diff = self.permutation_testing_of_deconvolved_responses(diff_prpe_npre, time_points = time_points.mean(axis=0))
		clusters_dif_of_dif =mne.stats.permutation_cluster_1samp_test(diff_of_diff_rpes[:,:])[1]
		clusters_dif_of_dif_pval = mne.stats.permutation_cluster_1samp_test(diff_of_diff_rpes[:,:])[2] 


		conds_reward = pd.Series(['Low pred - Loss', 'Low pred - Reward', 'High pred - Loss', 'High pred - Reward'])
		conds_rpe = pd.Series(['Positive RPE', 'Negative RPE'])
		hues = sn.color_palette()
		prpe_npre_cols = {hues[1], hues[2]}
	
		sn.set(font_scale=1, style="ticks")	
		fig = pl.figure(figsize=(10,6)) 
		s = fig.add_subplot(231)
		sn.tsplot(dec_time_courses_reward_first_s[:,:,:], err_style="ci_band",  condition=conds_reward, time=time_points.mean(axis=0))
		for x in range(len(sig_timepoints_first)): 
			s.plot(sig_timepoints_first[x], np.zeros(len(sig_timepoints_first[x]))-[0.4,0.42, 0.44][x], color=[hues[0],hues[1],hues[2]][x], ls='--', alpha = 0.8) #
		s.set_title('First part reversal block', fontsize=10)
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10)
		s.set_ylim([-0.5, 0.4])
		pl.tight_layout()
		s = fig.add_subplot(232)
		sn.tsplot(dec_time_courses_reward_last_s[:,:,:], err_style="ci_band",  condition=conds_reward, time=time_points.mean(axis=0))
		for x in range(len(sig_timepoints_last )): 
			s.plot(sig_timepoints_last [x], np.zeros(len(sig_timepoints_last [x]))-[0.4,0.42][x], color=[hues[2],hues[3]][x], ls='--', alpha = 0.8) #
		s.set_title('Last part reversal block', fontsize=10)
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10)
		s.set_ylim([-0.5, 0.4])
		pl.tight_layout()
		s = fig.add_subplot(233)
		sn.tsplot(diff_deconvolved_reward_time_courses_first_last[:,:,:], err_style="ci_band",  condition=conds_reward, time=time_points.mean(axis=0))
		for x in range(len(sig_timepoints_diff)): 
			s.plot(sig_timepoints_diff[x], np.zeros(len(sig_timepoints_diff[x]))-[0.4,0.42, 0.44][x], color=[hues[0], hues[1], hues[3]][x], ls='--', alpha = 0.8) #
		s.set_title('Difference first & last part reversal block', fontsize=10)
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		s.set_ylim([-0.5, 0.4])
		sn.despine(offset=10)
		pl.tight_layout()
		s = fig.add_subplot(234)
		sn.tsplot(prpe_npre_first[:,:,:], err_style="ci_band",  condition=conds_rpe, color =prpe_npre_cols, time=time_points.mean(axis=0))
		for x in range(len(sig_timepoints_rpe_first)): 
			s.plot(sig_timepoints_rpe_first[x], np.zeros(len(sig_timepoints_rpe_first[x]))-0.4, color=hues[1], ls='--', alpha = 0.8) #
		s.set_title('First part reversal block', fontsize=10)
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		s.set_ylim([-0.5, 0.4])
		sn.despine(offset=10)
		pl.tight_layout()
		s = fig.add_subplot(235)
		sn.tsplot(prpe_npre_last[:,:,:], err_style="ci_band",  condition=conds_rpe, color =prpe_npre_cols, time=time_points.mean(axis=0))		
		for x in range(len(sig_timepoints_rpe_last)): 
			s.plot(sig_timepoints_rpe_last[x], np.zeros(len(sig_timepoints_rpe_last[x]))-[0.4, 0.42][x], color=[hues[1], hues[2]][x], ls='--', alpha = 0.8) #
		s.set_title('Last part reversal block', fontsize=10)
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		s.set_ylim([-0.5, 0.4])
		sn.despine(offset=10)
		pl.tight_layout()
		s = fig.add_subplot(236)
		sn.tsplot(diff_prpe_npre[:,:,:], err_style="ci_band",  condition=conds_rpe, color =prpe_npre_cols, time=time_points.mean(axis=0))
		for x in range(len(sig_timepoints_rpe_diff)): 
			s.plot(sig_timepoints_rpe_diff[x], np.zeros(len(sig_timepoints_rpe_diff[x]))-[0.4, 0.42][x], color=[hues[1],hues[2]][x], ls='--', alpha = 0.8) #
		s.set_title('Difference first & last part reversal block', fontsize=10)
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		s.set_ylim([-0.5, 0.4])
		sn.despine(offset=10)
		pl.tight_layout()
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'compare_reward_conditions_%s_reversalblock.pdf'%use_domain))


	def permutation_testing_of_deconvolved_responses(self, deconvolved_responses, time_points): 
		"""permutation_testing_of_deconvolved_responses calculates the timepoints where the deconvolved_responses deviate significantly from 0. Deconvolved_responses should be put in in the following axis order: [participants, deconvolved timepoints, conditions]. Timepoints is an array containing [t_start, t_end, number of samples within the time interval]"""
		

		clusters=[mne.stats.permutation_cluster_1samp_test(deconvolved_responses[:,:,i])[1] for i in range(deconvolved_responses.shape[2])]
		clusters_pval = [mne.stats.permutation_cluster_1samp_test(deconvolved_responses[:,:,i])[2] for i in range(deconvolved_responses.shape[2])]
		sig_pval = np.concatenate([pval < 0.05 for pval in clusters_pval])
		cluster_timepoints = [time_points[clusters[i][j]] for i in range(len(clusters)) for j in range(len(clusters[i])) if len(clusters[i])>0] #all cluster timepoints
		sig_timepoints = [val for indx,val in enumerate(cluster_timepoints) if sig_pval[indx]] #only select significant timepoints 

		return clusters_pval, sig_timepoints


	def pupil_around_keypress_across_subjects(self, period_of_interest=60, analysis_sample_rate=20, padding=False, detrending = False):
		"""pupil signal around keypress across subjects gets the pupil timecourses around the period_of_interest, averages per participant
		and plots the average signal per participant.  """

		folder_name = 'pupil_around_keypress'
		if padding:
			folder_name = 'padded_' + folder_name
		if detrending:
			folder_name = 'detrended_' + folder_name

		deconv_pupil_timecourses = self.gather_data_from_hdfs(group = folder_name, data_type = 'deconvolved_pupil_timecourses')
		covariate_keys = np.concatenate(self.gather_data_from_hdfs(group = folder_name, data_type = 'covariate_keys')[0])
		rsquared = np.array(self.gather_data_from_hdfs(group = folder_name, data_type = 'rsquared'))

		names = [s.subject.initials for s in self.sessions]
		timepoints = np.linspace(-period_of_interest, period_of_interest, 2*(period_of_interest*analysis_sample_rate)) 
		shell() 
		#deconvolution results across participants 
		sn.set(font_scale=1)
		sn.set_style("ticks")		
		cond = pd.Series(['pupil around keypress', 'pupil around experimental reversal'])
		f = pl.figure(figsize = (5,6)) 
		s = f.add_subplot(111)
		sn.tsplot(deconv_pupil_timecourses[:,:,:], err_style="ci_band", condition = cond, time=timepoints)
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.xlabel('time (s)')
		pl.ylabel('Z')
		s.set_ylim([-0.4, 0.8])	
		sn.despine(offset=10)
		pl.tight_layout()
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'deconvolved_pupil_key_reversal_baseline_zscore_padding_%s_%s.pdf'%(str(padding), str(detrending))), rasterized=True)	

		#rsquared boxplot (not used now)
		# s = f.add_subplot(122)
		# df = pd.DataFrame(rsquared, index=names)
		# df.boxplot(grid=False)          
		# pl.scatter(np.tile(np.arange(df.shape[1])+1, df.shape[0]), df.values.ravel(), marker='o', alpha=0.3)
		# pl.ylabel('rsquared')
		# s.tick_params(bottom='off')
		# sn.despine(trim=True)
		# pl.tight_layout()
		# pl.savefig(os.path.join(self.grouplvl_plot_dir, 'deconvolved_pupil_key_reversal_baseline_zscore_padding_%s_%s.pdf'%(str(padding), str(detrending))))	
		
		# #keypress plot per participant
		fig = pl.figure(figsize=(12,12))
		for idx, name in enumerate(names):
			s = fig.add_subplot(5,6,idx+1)       #subplots
			sn.tsplot(deconv_pupil_timecourses[idx,:,0], err_style="ci_band", time = timepoints)           
			s.set_title(names[idx])
			pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
			pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')	
			sn.despine(offset=10)
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'pupil_around_keypress_baseline_zscore_padding_%s_%s.pdf'%(str(padding), str(detrending))))

		# #reversal point (sound) per participant
		fig = pl.figure(figsize=(12,12))
		for idx, name in enumerate(names):
			s = fig.add_subplot(5,6,idx+1)       #subplots 
			sn.tsplot(deconv_pupil_timecourses[idx,:,1], err_style="ci_band", time = timepoints)          
			s.set_title(names[idx])
			pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
			pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')	
			sn.despine(offset=10)
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'pupil_around_reversal_baseline_zscore_padding_%s.pdf'%(str(padding), str(detrending))))

	
	def pupil_around_keypress_tonic_baselines_across_subjects(self, analysis_sample_rate=5, period_of_interest=60, zscore=True): 
		"""pupil_around_keypress_tonic_baselines collects all deconvolution results from reversal and keypress events regressed on 
		tonic signals from self.filter_bank. It plots the deconvolution results for the pupil response around keypress and around true 
		reversal point. """

		#### TIME FREQUENCY DECONVOLUTION #####
		folder_name = 'deconvolve_pupil_around_keypress_20_tonic_baseline_zscore_%s'%str(zscore)
		#folder_name = 'deconvolve_pupil_around_keypress_6_tonic_baseline'
		#folder_name = 'deconvolve_pupil_around_keypress_6_tonic_baseline_zscore_False'
		keys = self.gather_data_from_hdfs(group = folder_name, data_type = 'keys')[0]
		keys = [k.rpartition('.')[-1] for k in keys]
		freqs = np.array([float('0.' + key) for key in keys])
		freq_order = np.argsort(freqs)	
		names = [s.subject.initials for s in self.sessions]
		time_points = self.gather_data_from_hdfs(group = folder_name, data_type = 'time_points')
		timepoints = np.linspace(-period_of_interest, period_of_interest, 2*(period_of_interest*analysis_sample_rate)) 
		
		###### POWER SPECTRA#######		
		av_pupil_power_spectum = np.mean(np.array(self.gather_data_from_hdfs(group = 'power_spectra_pupil_and_experiment', data_type = 'pupil_power_spectrum')), axis=0)
		experiment_power_spectrum = np.mean(np.array(self.gather_data_from_hdfs(group = 'power_spectra_pupil_and_experiment', data_type = 'experiment_power_spectrum')), axis=0)
		pupil_freqs = np.array(self.gather_data_from_hdfs(group = 'power_spectra_pupil_and_experiment', data_type = 'pupil_freqs'))[0]
		exp_freqs = np.array(self.gather_data_from_hdfs(group = 'power_spectra_pupil_and_experiment', data_type = 'exp_freqs'))[0]

		all_button_press_timecourses=[]
		all_reversal_point_timecourses=[]
		all_rsquared=[]	

		#loop over deconvolution results of all self.filter_bank deconvolutions	
		for idx in freq_order: 
			deconv_pupil_timecourses = self.gather_data_from_hdfs(group = folder_name, data_type = 'deconvolved_pupil_timecourses_%s_hz'%keys[idx])
			all_button_press_timecourses.append(deconv_pupil_timecourses[:,:,0])
			all_reversal_point_timecourses.append(deconv_pupil_timecourses[:,:,1])
			rsquared = self.gather_data_from_hdfs(group = folder_name, data_type = 'rsquared_%s_hz'%keys[idx])
			all_rsquared.append(rsquared)

		all_button_press_timecourses = np.array(all_button_press_timecourses).transpose(1,2,0)
		all_rsquared = np.squeeze(np.array(all_rsquared))
		cond = pd.Series([freqs[f] for f in freq_order],name='frequencies')
		rsquared_df = pd.DataFrame(all_rsquared.T, columns=cond)

		# greens = sn.cubehelix_palette(20, start=2, rot=0, dark=0, light=.85, reverse=True)	
		hues = sn.color_palette()		
		# shell() 
		# f = pl.figure(figsize=(6,10)) 
		# sn.set(font_scale=1)
		# sn.set_style("ticks")		
		# s = f.add_subplot(211)
		# sn.tsplot(all_button_press_timecourses, err_style='ci_band', time=timepoints, color=greens, condition=cond)
		# pl.xlabel('time (s)')
		# pl.ylabel('pupil size (Z)')
		# sn.despine(offset=10)
		# s = f.add_subplot(212)
		# rsquared_df.boxplot(grid=False, rot=45, fontsize=10) 
		# pl.xlabel('frequency band')
		# pl.ylabel('rsquared')
		# sn.despine(trim=True)
		# pl.tight_layout()
		# pl.savefig(os.path.join(self.grouplvl_plot_dir, 'pupil_around_keypress_20_tonic_baselines_zscore_%s.pdf'%str(zscore))) 		

			      
		f = pl.figure()
		s = f.add_subplot(111)
		sn.set(font_scale=1)
		sn.set_style("ticks")		
		exp_plottable_freqs = (exp_freqs < 0.05) & (exp_freqs > 0.002)
		pupil_plottable_freqs = (pupil_freqs < 0.05) & (pupil_freqs > 0.002)
		# pl.plot(pupil_freqs[1:], av_pupil_power_spectum[1:] / av_pupil_power_spectum[1:].max(), color=hues[0]) # plot single-sided power spectrum
		# pl.plot(exp_freqs[1:], experiment_power_spectrum[1:] / experiment_power_spectrum[1:].max(), color=hues[1]) # plot single-sided power spectrum
		pl.plot(pupil_freqs[pupil_plottable_freqs], np.log(av_pupil_power_spectum[pupil_plottable_freqs]) - np.mean(np.log(av_pupil_power_spectum[pupil_plottable_freqs])), color=hues[0]) # plot single-sided power spectrum
		pl.plot(exp_freqs[exp_plottable_freqs], np.log(experiment_power_spectrum[exp_plottable_freqs]) - np.mean(np.log(experiment_power_spectrum[exp_plottable_freqs])), color=hues[1]) # plot single-sided power spectrum
		s.set_xlim([0.002,0.05])
		s.set_xscale('log')
		pl.legend(['pupil power spectrum', 'experiment power spectrum'], fontsize=12)
		pl.xlabel('frequency')
		pl.ylabel('power')
		sn.despine(offset=5)
		pl.tight_layout()
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'pupil_experiment_powerspectrum_20_baselines_%s.pdf'%str(zscore))) 		


		f = pl.figure()
		s = f.add_subplot(111) 
		sn.set(font_scale=1)
		sn.set_style("ticks")		
		sn.despine(offset=5)
		rsquared_df.boxplot(grid=False, rot=45, fontsize=12) 
		pl.xlabel('frequency band', fontsize=12)
		pl.ylabel('rsquared', fontsize=12)
		pl.tight_layout()
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'pupil_experiment_powerspectrum_20_baselines_%s_barplot.pdf'%str(zscore))) 		



	def powerspectra_pupil_and_experiment_across_subjects(self): 
		"""takes all participant's powerspectra of the interpolated pupil and experimental design and averages across participants """

		folder_name = 'power_spectra_pupil_and_experiment' 		
		av_pupil_power_spectum = np.mean(np.array(self.gather_data_from_hdfs(group = folder_name, data_type = 'pupil_power_spectrum')), axis=0)
		experiment_power_spectrum = np.mean(np.array(self.gather_data_from_hdfs(group = folder_name, data_type = 'experiment_power_spectrum')), axis=0)
		pupil_freqs = np.array(self.gather_data_from_hdfs(group = folder_name, data_type = 'pupil_freqs'))[0]
		exp_freqs = np.array(self.gather_data_from_hdfs(group = folder_name, data_type = 'exp_freqs'))[0]


		#shell() 

		hues = sn.color_palette()		

		f = pl.figure() 
		sn.set(font_scale=1)
		sn.set_style("ticks")
		s = f.add_subplot(111)
		pl.plot(pupil_freqs, av_pupil_power_spectum, color=hues[0], alpha=0.5) # plot single-sided power spectrum
		pl.plot(exp_freqs, experiment_power_spectrum, color=hues[1], alpha=0.5) # plot single-sided power spectrum
		s.set_xlim([0.002,0.05])
		s.set_xscale('log')
		pl.legend(['pupil power spectrum', 'experiment power spectrum'])
		s.set_title('Pupil power spectrum', fontsize=10)
		f.text(0.5, 0.02, 'Freq(Hz)', ha='center',  va = 'center', fontsize=10)
		f.text(0.04, 0.5, 'Power', ha ='center', va='center', rotation='vertical', fontsize=10)
		sn.despine(offset=5)
		s = f.add_subplot(122)
		pl.plot(exp_freqs, experiment_power_spectrum, color=hues[0]) # plot single-sided power spectrum
		s.set_title('Experiment power spectrum', fontsize=10)
		s.set_xlim([0.002,0.05])
		s.set_xscale('log')
		sn.despine(offset=5)




	def calculate_distance_reversals_to_peak_and_keypresses_across_subjects(self, period_of_interest=60, analysis_sample_rate=20): 
		"""calculate_distance_reversals_to_peak_and_keypresses_across_subjects calculates the correlation between distance reversal point <-> peak 
		and reversal point <-> keypress.  """
		
		names = [s.subject.initials for s in self.sessions]
		folder_name = 'distance_reversal_keypress_behaviour' 
		rsquared = np.squeeze(np.array(self.gather_data_from_hdfs(group = 'detrended_padded_pupil_around_keypress', data_type = 'rsquared')))

		distance_reversal_peak = self.gather_data_from_hdfs(group = folder_name, data_type = 'distance_reversal_peak').flatten()	
		bottom_to_peak_amp = np.squeeze(self.gather_data_from_hdfs(group = folder_name, data_type = 'peak'))[:,0]
		distance_reversal_keypress = self.gather_data_from_hdfs(group = folder_name, data_type = 'distance_reversal_keypress').flatten()
		distance_peak_keypress = self.gather_data_from_hdfs(group = folder_name, data_type = 'distance_peak_keypress').flatten()
		distance_reversal_peak = self.gather_data_from_hdfs(group = folder_name, data_type = 'distance_reversal_peak').flatten()

		percentage_positive = self.gather_data_from_hdfs(group = folder_name, data_type = 'percentage_positive')
		positive_distance_times	= self.gather_data_from_hdfs(group = folder_name, data_type = 'percentage_positive')		

		data = {'rev_key': distance_reversal_keypress, 'rev_peak': distance_reversal_peak, 'peak_key': distance_peak_keypress, 'peak_amp': bottom_to_peak_amp}
		df = pd.DataFrame(data)
		
		#calculate weighted correlation 
		[weighted_cor_rev_key, t_val_rev_key, p_val_rev_key] = self.calculate_weighted_correlation(x_data = df.rev_key, y_data=df.rev_peak, weights=rsquared)
		[weighted_cor_peak_key, t_val_peak_key, p_val_peak_key] = self.calculate_weighted_correlation(x_data = df.peak_key, y_data=df.peak_amp, weights=rsquared)
		#calculate weighted regression line 
		[rev_sort_order, rev_sorted_x_data, rev_sorted_y_data, rev_sorted_weights] = self.calculate_weighted_regression_line(x_data = df.rev_key, y_data=df.rev_peak, weights=rsquared)
		[sort_order, sorted_x_data, sorted_y_data, sorted_weights] = self.calculate_weighted_regression_line(x_data = df.peak_key, y_data=df.peak_amp, weights=rsquared)
		shell() 
		#joint plot
		current_palette = sn.color_palette()
				#reversal point - baseline peak key press 
		fig = sn.jointplot(x="rev_key", y="rev_peak", kind="reg", data=df, ratio=6, stat_func=None)
		fig.plot_joint(pl.scatter, c=current_palette[0], s=40, edgecolors='white')
		pl.text(s='weighted spearmanr: %.2f, p: %.5f'%(weighted_cor_rev_key, p_val_rev_key), x=60, y=135)
		fig.set_axis_labels("Distance exp. reversal - keypress (seconds)", "Distance reversal - baseline peak (seconds) ")
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'correlation_reversal_peak_key.pdf'))
		#amplitude - baseline keypress distance 
		fig = sn.jointplot(x="peak_key", y="peak_amp", kind="reg", data=df, ratio=6, stat_func=None)
		fig.plot_joint(pl.scatter, c=current_palette[0], s=40, edgecolors='white')
		pl.text(s='weighted spearmanr: %.2f, p: %.5f'%(weighted_cor_peak_key, p_val_peak_key), x=10, y=2.8 )
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'correlation_peak_amp_key.pdf'))
		
		#scatter plot with alphas shaded by rsquared fit 
		sn.set(font_scale=1, style="ticks")	
		alpha = [rsquared[i]/np.max(rsquared) for i in range(len(rsquared))]
		fit = np.polyfit(rev_sorted_x_data, rev_sorted_y_data, 1, w = rev_sorted_weights)
		fit_fn = np.poly1d(fit)		
		fig = pl.figure(figsize=(9,5))
		fig.add_subplot(121)
		for i,val in enumerate(names):
			pl.plot(df.rev_key[i], df.rev_peak[i], c=current_palette[0], marker='o', ms=12, alpha=alpha[i], mec='gray', label='Distance reversal point and baseline peak')
		pl.plot(rev_sorted_x_data,fit_fn(rev_sorted_x_data), c=current_palette[0], lw=2)
		pl.xlabel('Interval reversal - keypress (s)', fontsize=10)
		pl.ylabel('Interval reversal - pupil peak (s)', fontsize=10)
		#pl.title('Correlation exp to keypress and baseline peak', fontsize=10)
		pl.text(s='spearmanr: %.4f, p: %.4f'%(weighted_cor_rev_key, p_val_rev_key), x=np.max(df.rev_key)/2, y=75, fontsize=10)
		sn.despine(offset=5)
		pl.tight_layout()

		fit = np.polyfit(sorted_x_data, sorted_y_data, 1, w = sorted_weights)
		fit_fn = np.poly1d(fit)		
		fig.add_subplot(122)
		for i,val in enumerate(names):
			pl.plot(df.peak_key[i], df.peak_amp[i], c=current_palette[0], marker='o', ms=12, alpha=alpha[i], mec='white')
		pl.plot(sorted_x_data,fit_fn(sorted_x_data), c=current_palette[0], lw=2)
		#pl.title('Correlation baseline peak amplitude and keypress distance', fontsize=10)
		pl.xlabel('Interval pupil peak  - keypress (s)', fontsize=10)
		pl.ylabel('Amplitude pupil peak', fontsize=10)
		pl.text(s='spearmanr: %.4f, p: %.4f'%(weighted_cor_peak_key, p_val_peak_key), x=-10, y=2.34, fontsize=10 )
		sn.despine(offset=5)
		pl.tight_layout()
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'weighted_correlation_rev_peak_amp_key.pdf'))	
	

	def calculate_weighted_correlation(self, x_data, y_data, weights): 
		"""calculate_weighted_correlation calculates the correlation between xdata and ydata taking into account a weighting factor, e.g. rsquared. """
		
		weighted_mean_x = np.average(x_data ,weights=weights)
		weighted_mean_y = np.average(y_data,weights=weights)
		weighted_cov_x_x = np.sum([w * (x - weighted_mean_x) * (y - weighted_mean_y) for x,y,w in zip(x_data,x_data,weights)]) / np.sum(weights)
		weighted_cov_x_y = np.sum([w * (x - weighted_mean_x) * (y - weighted_mean_y) for x,y,w in zip(x_data,y_data,weights)]) / np.sum(weights)
		weighted_cov_y_y = np.sum([w * (x - weighted_mean_x) * (y - weighted_mean_y) for x,y,w in zip(y_data,y_data,weights)]) / np.sum(weights)
		weighted_corr = weighted_cov_x_y / np.sqrt(weighted_cov_x_x*weighted_cov_y_y)
		t_val = weighted_corr / np.sqrt((1-weighted_corr**2)/len(x_data-2))
		p_val = sp.stats.t.sf(np.abs(t_val), len(x_data)-1) 

		return [weighted_corr, t_val, p_val]

	def calculate_weighted_regression_line(self, x_data, y_data, weights): 
		"""calculate_weighted_regression_line calculates the regression line that corresponds to the weighted correlation of 
		calculate_weighted_correlation """
		
		sort_order = x_data.argsort().argsort()
		sorted_x_data = x_data[sort_order]
		sorted_y_data = y_data[sort_order]
		sorted_weights = weights[sort_order]

		return [sort_order, sorted_x_data, sorted_y_data, sorted_weights]


	def clear_domain_group_results(self, use_domain = 'clear', standard_deconvolution='sound', smooth_width = 10, baseline=False ): 
		"""clear_domain_group_results averages deconvolution "deconvolve_clear using the following regressors"         
		regressor:               column:
		blink                    [0]
		keypress                 [1]
		saccade                  [2]
		colour                   [3]
		cue low, high            [4][5]
		sound                    [6]
		LPNR, LPHR, HPNR, HPHR   [7][8][9][10]
		""" 
	   
		folder_name = 'deconvolve_%s_domain_'%use_domain 
		dec_time_courses = self.gather_data_from_hdfs(group = 'deconvolve_%s_domain_'%use_domain , data_type = 'dec_time_course')
		time_points = self.gather_data_from_hdfs(group = 'deconvolve_%s_domain_'%use_domain, data_type = 'time_points')
		baseline_times = time_points.mean(axis = 0) < 0  #take baseline from timepoints interval [-0.5 - 0] 
		rsquared = np.mean(self.gather_data_from_hdfs(group = folder_name, data_type = 'rsquared'), axis=0) 
		analysis_sample_rate = len(time_points[0])/np.abs(time_points[0][0] - time_points[0][-1])

		if baseline: 
			dec_time_courses_s = np.array([[myfuncs.smooth(dec_time_courses[i,:,j], window_len=10) - dec_time_courses[i,baseline_times,j].mean() for i in range(dec_time_courses.shape[0])] for j in range(dec_time_courses.shape[-1])]).transpose((1,2,0))
		else: 
			dec_time_courses_s = np.array([[myfuncs.smooth(dec_time_courses[i,:,j], window_len=10) for i in range(dec_time_courses.shape[0])] for j in range(dec_time_courses.shape[-1])]).transpose((1,2,0))
	
		dec_time_courses_nuisance = np.array([dec_time_courses_s[:,:,0], dec_time_courses_s[:,:,1], dec_time_courses_s[:,:,2]]).transpose((1,2,0))
		dec_time_courses_colour_sound = np.array([dec_time_courses_s[:,:,3], dec_time_courses_s[:,:,6]]).transpose((1,2,0))
		dec_time_courses_cue = np.array([dec_time_courses_s[:,:,4], dec_time_courses_s[:,:,5]]).transpose((1,2,0))
		dec_time_courses_reward = np.array([dec_time_courses_s[:,:,7], dec_time_courses_s[:,:,8], dec_time_courses_s[:,:,9], dec_time_courses_s[:,:,10]]).transpose((1,2,0))
		dec_time_courses_rpe = np.array([dec_time_courses_s[:,:,8] - dec_time_courses_s[:,:,10], dec_time_courses_s[:,:,9] - dec_time_courses_s[:,:,7]]).transpose((1,2,0))
		dec_time_courses_pe = np.array([dec_time_courses_s[:,:,10] - dec_time_courses_s[:,:,7], dec_time_courses_s[:,:,8] - dec_time_courses_s[:,:,9]]).transpose((1,2,0))

		##permutation testing 
		clusters=[mne.stats.permutation_cluster_1samp_test(dec_time_courses_reward[:,:,i])[1] for i in range(dec_time_courses_reward.shape[2])]
		clusters_pval = [mne.stats.permutation_cluster_1samp_test(dec_time_courses_reward[:,:,i])[2] for i in range(dec_time_courses_reward.shape[2])]
		sig_pval = np.concatenate([pval < 0.05 for pval in clusters_pval])
		cluster_timepoints = [time_points.mean(axis=0)[clusters[i][j]] for i in range(len(clusters)) for j in range(len(clusters[i])) if len(clusters[i])>0] #all cluster timepoints
		sig_timepoints = [val for indx,val in enumerate(cluster_timepoints) if sig_pval[indx]] #only select significant timepoints 

		clusters_RPE = [mne.stats.permutation_cluster_1samp_test(dec_time_courses_rpe[:,:,i])[1] for i in range(dec_time_courses_rpe.shape[2])]
		clusters_pval_RPE = [mne.stats.permutation_cluster_1samp_test(dec_time_courses_rpe[:,:,i])[2] for i in range(dec_time_courses_rpe.shape[2])]
		sig_pval_RPE = np.concatenate([pval < 0.05 for pval in clusters_pval_RPE])
		cluster_timepoints_RPE = [time_points.mean(axis=0)[clusters_RPE[i][j]] for i in range(len(clusters_RPE)) for j in range(len(clusters_RPE[i])) if len(clusters_RPE[i])>0] #all cluster timepoints
		sig_timepoints_RPE = [val for indx,val in enumerate(cluster_timepoints_RPE) if sig_pval_RPE[indx]] #only select significant timepoints 

		clusters_pe = [mne.stats.permutation_cluster_1samp_test(dec_time_courses_pe[:,:,i])[1] for i in range(dec_time_courses_pe.shape[2])]
		clusters_pval_pe = [mne.stats.permutation_cluster_1samp_test(dec_time_courses_pe[:,:,i])[2] for i in range(dec_time_courses_pe.shape[2])]
		sig_pval_pe = np.concatenate([pval < 0.05 for pval in clusters_pval_pe])
		cluster_timepoints_pe = [time_points.mean(axis=0)[clusters_pe[i][j]] for i in range(len(clusters_pe)) for j in range(len(clusters_pe[i])) if len(clusters_pe[i])>0] #all cluster timepoints
		sig_timepoints_pe = [val for indx,val in enumerate(cluster_timepoints_pe) if sig_pval_pe[indx]] #only select significant timepoints 

		#condition labels
		conds_nuisance = pd.Series(['blink', 'keypress', 'saccade'])
		conds_colour_sound = pd.Series(['colour', 'sound'])
		conds_cue = pd.Series(['LP cue', 'HP cue'])
		conds_reward = pd.Series(['LPNR', 'LPHR', 'HPNR', 'HPHR'])
		conds_rpe = pd.Series(['Positive RPE', 'Negative RPE'])
		conds_pe = pd.Series(['Predicted', 'Unpredicted'])

		sn.set(style="ticks")
		f = pl.figure()
		ax = f.add_subplot(321)
		ax = sn.tsplot(dec_time_courses_nuisance, err_style="ci_band",  condition = conds_nuisance, time = time_points.mean(axis = 0)) 
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10, trim=True)
		pl.tight_layout()
		pl.title('Effect of blink, keypress and saccade (%s)'%use_domain)
		pl.ylabel('Z')
		ax = f.add_subplot(322)
		ax = sn.tsplot(dec_time_courses_colour_sound, err_style="ci_band",  condition=conds_colour_sound, time=time_points.mean(axis=0))
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		ax.set_ylim([-0.6, 0.6])
		sn.despine(offset=10, trim=True)
		pl.tight_layout()
		pl.title('Effect of colour and sound (%s)'%use_domain)
		pl.ylabel('Z')
		ax = f.add_subplot(323)
		ax = sn.tsplot(dec_time_courses_cue, err_style="ci_band",  condition=conds_cue, time=time_points.mean(axis=0))
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		ax.set_ylim([-0.15, 0.15])
		sn.despine(offset=10, trim=True)
		pl.tight_layout()
		pl.title('Effect of cue (%s)'%use_domain)
		pl.ylabel('Z')
		ax = f.add_subplot(324)
		if use_domain == 'clear': 
			for x in range(len(sig_timepoints)): 
				ax.plot(sig_timepoints[x], np.zeros(len(sig_timepoints[x]))-[0.33,0.35,0.37][x], color=['darkseagreen', 'indianred', '#8d5eb7'][x], ls='--', alpha = 0.8) #'royalblue', , 0.35, 0.37, 0.39][x]
		else: 
			for x in range(len(sig_timepoints)): 
				ax.plot(sig_timepoints[x], np.zeros(len(sig_timepoints[x]))-0.33, color='royalblue', ls='--', alpha = 0.8) 
		ax = sn.tsplot(dec_time_courses_reward, err_style="ci_band",  condition=conds_reward, time=time_points.mean(axis=0))
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		ax.set_ylim([-0.4, 0.4])
		sn.despine(offset=10, trim=True)
		pl.tight_layout()
		pl.title('Effect of reward outcome (%s)'%use_domain)
		pl.ylabel('Z')
		ax = f.add_subplot(325)
		if use_domain == 'clear': 
			for x in range(len(sig_timepoints_RPE)): 
				ax.plot(sig_timepoints_RPE[x], np.zeros(len(sig_timepoints_RPE[x]))-[0.2, 0.22][x], color =['royalblue', 'darkseagreen'][x], ls='--', alpha=0.8) 
		else: 
			for x in range(len(sig_timepoints_RPE)): 
				ax.plot(sig_timepoints_RPE[x], np.zeros(len(sig_timepoints_RPE[x]))-0.24, color ='darkseagreen', ls='--', alpha=0.8) 
		ax = sn.tsplot(dec_time_courses_rpe, err_style="ci_band",  condition=conds_rpe, time=time_points.mean(axis=0))
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		ax.set_ylim([-0.3, 0.4])
		sn.despine(offset=10, trim=True)
		pl.tight_layout()
		pl.title('Effect of reward prediction error (%s)'%use_domain)
		pl.ylabel('Z')
		ax = f.add_subplot(326)
		if use_domain == 'clear': 
			for x in range(len(sig_timepoints_pe)): 
				ax.plot(sig_timepoints_pe[x], np.zeros(len(sig_timepoints_pe[x]))-[0.30, 0.32][x], color = ['royalblue', 'darkseagreen'][x], ls='--', alpha=0.8) 
		else:             
			for x in range(len(sig_timepoints_pe)): 
				ax.plot(sig_timepoints_pe[x], np.zeros(len(sig_timepoints_pe[x]))-0.30, color = 'royalblue', ls='--', alpha=0.8) 
		ax = sn.tsplot(dec_time_courses_pe, err_style="ci_band",  condition=conds_pe, time=time_points.mean(axis=0))
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10, trim=True)
		ax.set_ylim([-0.35, 0.2])
		pl.tight_layout()
		pl.title('Effect of prediction error (%s)'%use_domain)
		pl.ylabel('Z')
		pl.savefig(os.path.join(self.grouplvl_plot_dir, '%s_domain_deconvolution_across_subjects_baseline_%s.pdf'%(use_domain, str(baseline))))


		
		# f = pl.figure()
		# ax = f.add_subplot(311)
		# for x in range(len(sig_timepoints)): 
		#     ax.plot(sig_timepoints[x], np.zeros(len(sig_timepoints[x]))-0.14, color='violet', ls='--', alpha = 0.7)
		# ax = sn.tsplot(dec_time_courses_s, err_style="ci_band", condition = conds, time = time_points.mean(axis = 0), color = color_map) # ci=cis, 
		# pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		# pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')      
		# sn.despine(offset=10, trim=True)
		# pl.tight_layout()   
		# pl.title('Effect Reward probability trials on pupil dilation (%s_domain, %s_resids, %i Hz)'%(use_domain, standard_deconvolution, analysis_sample_rate))
		# pl.text(0,0.05, 'rsquared: %.3f'%rsquared)
		# ax = f.add_subplot(312)
		# ax = sn.tsplot(dec_time_courses_diff, err_style="ci_band", condition = conds_diff, time = time_points.mean(axis = 0), color = color_map2) 
		# for x in range(len(sig_timepoints_RPE)): 
		#     ax.plot(sig_timepoints_RPE[x], np.zeros(len(sig_timepoints_RPE[x])) -0.14, color = 'gray', ls='--', alpha=0.8)
		# pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		# pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		# sn.despine(offset=10, trim=True)
		# pl.title('Effect of RPE on pupil dilation (%s_domain, %s_resids) '%(use_domain, standard_deconvolution))
		# pl.tight_layout()
		# ax = f.add_subplot(313) 
		# ax = sn.tsplot(dec_time_courses_reward, err_style="ci_band", condition = conds_reward, time = time_points.mean(axis=0), color = color_map3)
		# for x in range(len(sig_timepoints_rew)): 
		#     ax.plot(sig_timepoints_rew[x], np.zeros(len(sig_timepoints_rew[x])) -0.14, color = 'gray', ls='--', alpha=0.8)
		# pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		# pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		# sn.despine(offset=10, trim=True)
		# pl.title('Effect of reward on pupil dilation (%s_domain, %s_resids) '%(use_domain, standard_deconvolution))
		# pl.tight_layout()
		# pl.savefig(os.path.join(self.grouplvl_plot_dir, '%s_domain_deconv_%s_%s_across_subjects_%i_hz.pdf'%(use_domain, standard_deconvolution, str(baseline), analysis_sample_rate)))
		
		#old residual pupil analysis 
		# if standard_deconvolution=='sound':     
		#     folder_name = 'deconvolve_%s_domain_%s_%s'%(use_domain, str(microsaccades_added), standard_deconvolution) 
		# elif standard_deconvolution=='no_sound': 
		#     folder_name = 'deconvolve_%s_domain_%s_%s'%(use_domain, str(microsaccades_added), standard_deconvolution)         
		#folder_name = 'deconvolve_%s_domain_%s_%s'%(use_domain, str(microsaccades_added), standard_deconvolution)




	def ANOVA(self, standard_deconvolution='sound', baseline=True, use_domain='first_third_domain', microsaccades_added=False, smooth_width=10, analysis_sample_rate=10): 
		"""compute ANOVA for deconvolution conditions over time """

		#folder_name = 'deconvolve_%s_domain_%s_%s'%(use_domain, str(microsaccades_added), standard_deconvolution) 
		#dec_time_courses = self.gather_data_from_hdfs(group = folder_name , data_type= 'dec_time_course')

		
		dec_time_courses_first_domain = self.gather_data_from_hdfs(group = 'deconvolve_full_first_3split_domain', data_type = 'dec_time_course')
		dec_time_courses_last_domain = self.gather_data_from_hdfs(group = 'deconvolve_full_third_3split_domain', data_type = 'dec_time_course')
		dec_reward_timecourses_first_domain = dec_time_courses_first_domain[:,:,7:]
		dec_reward_timecourses_last_domain = dec_time_courses_last_domain[:,:,7:]
		diff_reward_timecourses = dec_reward_timecourses_last_domain - dec_reward_timecourses_first_domain

		time_points = self.gather_data_from_hdfs(group = 'deconvolve_full_first_3split_domain', data_type = 'time_points')[0]
		baseline_times = time_points.mean(axis=0) < 0
		 
		if baseline: 
			dec_time_courses_s = np.array([[myfuncs.smooth(diff_reward_timecourses[i,:,j], window_len=smooth_width) - diff_reward_timecourses[i,baseline_times,j].mean() for i in range(diff_reward_timecourses.shape[0])] for j in range(diff_reward_timecourses.shape[-1])]).transpose((0,1,2)) #condition, subject, time
		else: 
			dec_time_courses_s = np.array([[myfuncs.smooth(diff_reward_timecourses[i,:,j], window_len=smooth_width) for i in range(diff_reward_timecourses.shape[0])] for j in range(diff_reward_timecourses.shape[-1])]).transpose((0,1,2)) #condition, subject, time
		
		stat_results     = np.zeros((3, dec_time_courses_s.shape[2])) #matrix of stat_results x timepoints 
		stat_results_log = np.zeros((3, dec_time_courses_s.shape[2])) #matrix of stat_results x timepoints 

		#loop over timepoints: 
		for j in range(dec_time_courses_s.shape[2]): 
			data = np.array([dec_time_courses_s[i][s][j] for s in range(len(self.sessions)) for i in range(4)]).ravel() #data = data for each condition and each subject
			expectancy = np.array([np.array([0,0,1,1]) for s in range(len(self.sessions))]).ravel() #main effect expectancy. 0 = low expectation, 1 = high expectation
			outcome = np.array([np.array([0,1,0,1]) for s in range(len(self.sessions))]).ravel()    #main effect outcome. 0 = low outcome, 1 = high outcome    
			subject = np.array([np.repeat(s,4) for s in range(len(self.sessions))]).ravel()         #subject number       

			d = rlc.OrdDict([('expectancy', robjects.IntVector(list(expectancy.ravel()))), ('outcome', robjects.IntVector(list(outcome.ravel()))), ('subject', robjects.IntVector(list(subject.ravel()))), ('data', robjects.FloatVector(list(data.ravel())))])
			robjects.r.assign('dataf', robjects.DataFrame(d))
			robjects.r('attach(dataf)')
			statres = robjects.r('res = summary(aov(data ~ as.factor(expectancy)* as.factor(outcome) + Error(as.factor(subject)/(as.factor(expectancy)*as.factor(outcome))), dataf))')            
			#[column.rclass[0] for column in statres] #different columns in statres summary.aov
			stat_results[0,j] = statres.rx2(2)[0][-1][0] #expectancy
			stat_results[1,j] = statres.rx2(3)[0][-1][0] #outcome 
			stat_results[2,j] = statres.rx2(4)[0][-1][0] #expectancy * outcome, variance due to subject is parcelled out             
			
			#log-transform and min p-values of factors and their interaction
			stat_results_log[0,j] = -np.log10(statres.rx2(2)[0][-1][0])  #expectancy
			stat_results_log[1,j] = -np.log10(statres.rx2(3)[0][-1][0])  #outcome 
			stat_results_log[2,j] = -np.log10(statres.rx2(4)[0][-1][0])  #expectancy * outcome, subject variance is parcelled out 
		

		FDR_corrected_expectancy = mne.stats.fdr_correction(stat_results[0,:], alpha=0.05, method='indep') 
		FDR_corrected_outcome = mne.stats.fdr_correction(stat_results[1,:], alpha=0.05, method='indep') 
		FDR_corrected_interaction = mne.stats.fdr_correction(stat_results[2,:], alpha=0.05, method='indep') 

		FDR_corrected_expectancy_log = -np.log10(FDR_corrected_expectancy[1])
		FDR_corrected_outcome_log = -np.log10(FDR_corrected_outcome[1])
		FDR_corrected_interaction_log = -np.log10(FDR_corrected_interaction[1])

		cond = pd.Series(['Low prediction - Loss', 'Low prediction - Reward','High prediction - Loss', 'High prediction - Reward'])

		sn.set(font_scale=1)
		sn.set(style='ticks')
		fig = pl.figure(figsize = (6,5))
		ax = pl.subplot2grid((2,2), (0,0), colspan=2)
		ax.set_title('Anova reward-based learning over time')
		sn.tsplot(dec_time_courses_s.transpose(1,2,0), time=time_points, condition=cond)
		ax = fig.add_subplot(223)
		x = np.linspace(time_points[0], time_points[-1], time_points.shape[0])   
		ax.set_title('log scale results')     
		# pl.plot(x, stat_results_log[0,:], 'r', lw=1, alpha=0.5, label='Main: expectancy')
		# pl.plot(x, stat_results_log[1,:], 'b', lw=1, alpha = 0.5, label='Main: outcome')
		# pl.plot(x, stat_results_log[2,:], 'g', lw=2, label='Interaction')
		pl.plot(x, FDR_corrected_expectancy_log, 'r', lw=1, alpha=0.5, label='Main: expectancy')
		pl.plot(x, FDR_corrected_outcome_log, 'b', lw=1, alpha = 0.5, label='Main: outcome')
		pl.plot(x, FDR_corrected_interaction_log, 'g', lw=2, label='Interaction')
		pl.axhline(y=1.3, ls='--', lw=1)
		pl.ylabel('(-log10(p-values))', size=8)
		pl.xlabel('time (s)', size=8)
		pl.legend(loc="best", fontsize=8)
		pl.gca().spines['bottom'].set_linewidth(0.5)
		pl.gca().spines['left'].set_linewidth(0.5)
		sn.despine(offset=5, trim=True)
		pl.tight_layout()
		ax = fig.add_subplot(224)
		# pl.plot(x, stat_results[0,:], 'r', lw=1, alpha=0.5, label='Main: expectancy')
		# pl.plot(x, stat_results[1,:], 'b', lw=1, alpha = 0.5, label='Main: outcome')
		# pl.plot(x, stat_results[2,:], 'g', lw=2, label='Interaction') 
		pl.plot(x, FDR_corrected_expectancy[1], 'r', lw=1, alpha=0.5, label='Main: expectancy')
		pl.plot(x, FDR_corrected_outcome[1], 'b', lw=1, alpha = 0.5, label='Main: outcome')
		pl.plot(x, FDR_corrected_interaction[1], 'g', lw=2, label='Interaction')
		ax.set_title('p-value scale results')     
		pl.axhline(y=0.05, ls='--', lw=1)
		pl.ylabel('p-values', size=8)
		pl.xlabel('time (s)', size=8)
		pl.legend(loc="upper right", fontsize=8)
		pl.gca().spines['bottom'].set_linewidth(0.5)
		pl.gca().spines['left'].set_linewidth(0.5)
		sn.despine(offset=5, trim=True)
		pl.tight_layout()
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'ANOVA_%s_domain_deconv_%s_baseline_%s_across_subjects.pdf'%(use_domain, standard_deconvolution, str(baseline))))

	def linear_mixed_model(self, standard_deconvolution='sound', baseline=True, use_domain='clear', microsaccades_added=False, smooth_width=10, analysis_sample_rate=25):
		"""linear_mixed_model performs a linear mixed model analysis using statsmodels MixedLM on pupil deconvolution results. In contrary to to the (repeated measures) ANOVA, it's able to 
		handle continuous IV's such as time next to categorical IV's """
	   
		folder_name = 'deconvolve_%s_domain_%s_%s'%(use_domain, str(microsaccades_added), standard_deconvolution) 

		dec_time_courses = self.gather_data_from_hdfs(group = folder_name , data_type= 'dec_time_course')
		time_points = self.gather_data_from_hdfs(group= folder_name , data_type = 'time_points')
		baseline_times = time_points.mean(axis=0) < 0

		if baseline:
			#dimensions: subject, condition, time 
			dec_time_courses_s = np.array([[myfuncs.smooth(dec_time_courses[i,:,j], window_len=smooth_width) - dec_time_courses[i,baseline_times,j].mean() for i in range(dec_time_courses.shape[0])] for j in range(dec_time_courses.shape[-1])]).transpose((1,0,2)) #subject, condition, time
		else: 
			dec_time_courses_s = np.array([[myfuncs.smooth(dec_time_courses[i,:,j], window_len=smooth_width) for i in range(dec_time_courses.shape[0])] for j in range(dec_time_courses.shape[-1])]).transpose((1,0,2)) #subject, condition, time

		#make IV's to put in pandas dataframe in CSV like format (columns = variables, rows = observations)
		subject = np.array([np.repeat(s,(dec_time_courses_s.shape[1]*dec_time_courses_s.shape[2])) for s in range(len(self.sessions))]).ravel() #subjects in ascending order
		expectancy = np.tile(np.repeat([0,0,1,1], dec_time_courses_s.shape[2]), dec_time_courses_s.shape[0])    #main effect expectancy. 0 = low, 1 = high 
		outcome = np.tile(np.repeat([0,1,0,1], dec_time_courses_s.shape[2]), dec_time_courses_s.shape[0])       #main effect outcome. 0 = low outcome, 1 = high outcome
		time = np.tile(np.tile(np.arange(dec_time_courses_s.shape[2]), dec_time_courses_s.shape[1]), dec_time_courses_s.shape[0]) #time in samples
		#put pupil data in CSV like format 
		pupil_data = dec_time_courses_s.ravel(order='C') #last axis is changing fastest

		#put everything in pandas dataframe 
		data = {'subject': subject, 'expectancy': expectancy, 'outcome': outcome, 'time': time, 'pupil_data': pupil_data}
		df = pd.DataFrame(data)        
		#linear mixed model  
		#model = sma.MixedLM.from_formula("pupil_data ~  time + expectancy + outcome ", df, groups=df["subject"]) #welke formule hanteren? 
		#model = sma.MixedLM.from_formula("pupil_data ~ expectancy + outcome ", df, groups=df["subject"]) #laagste REML     
		result = model.fit()
		print result.summary()

 
	def TD_states_across_subjects(self, do_zoom='zoom'):
		"""TD_states_across_subjects calculates the average simulation results from the TD model over all participants """

		#array containing alpha,gamma,lambda,summed distances, summed reversals for each parameter combination 
		TD_simulations = np.array(self.gather_data_from_npzs(data_type = 'TD_model_behav_distance_%s.npz'%do_zoom))
		av_summed_distances = np.mean(TD_simulations[:,:,:,:,-1], axis=0)	
		
		#parameter combinations 
		best_TD_fit_values = np.squeeze(self.gather_data_from_hdfs(group = 'TD' , data_type= 'best_TD_fit_values_%s'%do_zoom))
		best_param_combi = np.squeeze(self.gather_data_from_hdfs(group = 'TD' , data_type= 'best_parameter_combination_%s'%do_zoom))
		av_best_param_combi = np.mean(best_param_combi, axis=0)
		#save best average TD parameters in .npy file
		np.save(os.path.join(self.grouplvl_data_dir, 'av_best_param_combi.npy'), av_best_param_combi)

		names = [s.subject.initials for s in self.sessions]		
		
		#plot boxplot of parameter values 
		alpha, gamma, lamb = best_param_combi[:,0], best_param_combi[:,1], best_param_combi[:,2]
		data_to_plot = [alpha, gamma, lamb]
		f = pl.figure(figsize=(4,4))
		ax = f.add_subplot(111)
		bp = ax.boxplot(data_to_plot, patch_artist=True)
		for box in bp['boxes']: 
			box.set(color='indianred', lw=2)
			box.set(facecolor='lightcoral')
		for wisker in bp['whiskers']: 
			wisker.set(color='k', lw=2)
		for cap in bp['caps']: 
			cap.set(color='k', lw=2)
		ax.set_xticklabels(['alpha', 'gamma', 'lambda'])
		ax.set_ylim(0,1.1)
		df = pd.DataFrame(best_param_combi, index=names, columns=['alpha', 'gamma', 'lambda'])
		df.boxplot(grid=False)          
		pl.scatter(np.tile(np.arange(df.shape[1])+1, df.shape[0]), df.values.ravel(), marker='o', alpha=0.3)
		pl.xlabel('parameters')
		pl.ylabel('parameter values')
		sn.despine(trim=True)
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'best_parameter_boxplot_%s.pdf'%do_zoom))

		#violin plot of the optimal parameter values 
		f=pl.figure(figsize=(4,4))
		sn.violinplot(df, inner='box', scale='width')		
		pl.xlabel('parameters')
		pl.ylabel('parameter value')
		sn.despine(trim=True)
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'best_parameter_violinplot_%s.pdf'%do_zoom))

		#clip the simulation to the minimum and maximum distance values  
		flattened_max_dist, flattened_min_dist = np.nanmax(av_summed_distances.flatten()), np.min(av_summed_distances.flatten()[np.nonzero(av_summed_distances.flatten())])
		simulation_to_plot = np.linspace(0,29,30)
		specific_lambda = np.linspace(0.7,1.0,30)

		#imshow group plot 
		fig, axes = pl.subplots(nrows=5, ncols=6, sharey=True, sharex=True, figsize = (10,8))       
		for i, ax in enumerate(axes.flat):
			im = ax.imshow(av_summed_distances[:,:,simulation_to_plot[i]], cmap='YlGnBu')
			ax.set_title('lambda:%.2f'%specific_lambda[i],fontsize=7) 
			#the x and y axis for alpha and gamma are named by their slice number, not by their real value (between 0.0-0.3 and 0.9-1.0 respectively.. how to solve?)		             
		im.set_clim(flattened_min_dist, flattened_max_dist)
		im.set_interpolation('bicubic')     
		fig.text(0.5, 0.02, 'alpha', ha='center', va='center', fontsize='medium')
		fig.text(0.08, 0.5, 'gamma', ha='center', va='center', rotation='vertical', fontsize='medium')
		fig.text(0.5, 0.98, 'Summed distance between model and participant behaviour \n of all TD simulation parameter combinations', ha='center', va='center', fontsize='large')
		#color bar 
		fig.subplots_adjust(right=0.8)
		cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
		fig.text(0.88, 0.88, 'summed distance', ha='center', va='center', fontsize='medium')
		fig.colorbar(im, cax=cbar_ax)
		pl.savefig(os.path.join(self.grouplvl_plot_dir,'summed_distance_imshow_%s.pdf'%do_zoom))
		
		##estimate probability density function using scipy's kernel density estimation 
		my_pdf = [stats.kde.gaussian_kde(keypress_timecourses[i]) for i in range(len(keypress_timecourses))]
		x = np.linspace(-50,30,100)
		#plot individual response patterns in a panel plot 
		f = pl.figure(figsize=(12,12))
		for i in range(len(keypress_timecourses)): 
		  for name in names: 
			  ax = f.add_subplot(5,6,i+1)
			  my_pdf = stats.kde.gaussian_kde(keypress_timecourses[i], bw_method='scott')
			  pl.plot(x, my_pdf(x),'royalblue', lw=2.0) # distribution function
			  pl.hist(keypress_timecourses[i],normed=1,alpha=.3, color='indianred') # histogram
			  ax.set_title(names[i])              
			  ax.set_xlim([-50,30])
			  ax.yaxis.label.set_size(7)
			  ax.xaxis.label.set_size(7)
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'kde_histogram_keypresstimes_scott_method.pdf'))






	def regress_TD_prediction_error_across_subjects(self, t_before=0.5, t_after=3.5, analysis_sample_rate=25, smooth_width=10, standard_deconvolution='no_sound', baseline=False, do_zscore='z_scored'): 

		regressed_timecourses_s = self.gather_data_from_hdfs(group = 'TD_regression' , data_type= 'regres_coeff_s_%s_%s'%(standard_deconvolution, do_zscore))
		rsquared_s = self.gather_data_from_hdfs(group = 'TD_regression' , data_type= 'rsquared_s_%s_%s'%(standard_deconvolution, do_zscore))
		regressed_timecourses_us = self.gather_data_from_hdfs(group = 'TD_regression' , data_type= 'regres_coeff_us_%s_%s'%(standard_deconvolution, do_zscore))
		
		timepoints = np.linspace(-0.5, (t_before+t_after), int(t_before + t_after)*analysis_sample_rate)
		#baseline_times = timepoints < 0 #[interval -0.5, 0]
		baseline_times = timepoints == 0
		
 
		if baseline: 
			regressed_timecourses_ss = np.array([[myfuncs.smooth(regressed_timecourses_s[i,:,j], window_len=smooth_width) - regressed_timecourses_s[i,baseline_times,j].mean() for i in range(regressed_timecourses_s.shape[0])] for j in range(regressed_timecourses_s.shape[-1])]).transpose((1,2,0))
			regressed_timecourses_uss = np.array([[myfuncs.smooth(regressed_timecourses_us[i,:,j], window_len=smooth_width) - regressed_timecourses_us[i,baseline_times,j].mean() for i in range(regressed_timecourses_us.shape[0])] for j in range(regressed_timecourses_us.shape[-1])]).transpose((1,2,0))
		else: 
			regressed_timecourses_ss = np.array([[myfuncs.smooth(regressed_timecourses_s[i,:,j], window_len=smooth_width) for i in range(regressed_timecourses_s.shape[0])] for j in range(regressed_timecourses_s.shape[-1])]).transpose((1,2,0))
			regressed_timecourses_uss = np.array([[myfuncs.smooth(regressed_timecourses_us[i,:,j], window_len=smooth_width) for i in range(regressed_timecourses_us.shape[0])] for j in range(regressed_timecourses_us.shape[-1])]).transpose((1,2,0))


		##permutation cluster test signed rpe
		clusters= [mne.stats.permutation_cluster_1samp_test(regressed_timecourses_ss[:,:,i])[1] for i in range(regressed_timecourses_ss.shape[2])]
		clusters_pval = [mne.stats.permutation_cluster_1samp_test(regressed_timecourses_ss[:,:,i])[2] for i in range(regressed_timecourses_ss.shape[2])]
		sig_pval = np.concatenate([pval < 0.05 for pval in clusters_pval])
		cluster_timepoints = [timepoints[clusters[i][j]] for i in range(len(clusters)) for j in range(len(clusters[i])) if len(clusters[i])>0] #all cluster timepoints
		sig_timepoints = [val for indx,val in enumerate(cluster_timepoints) if sig_pval[indx]] #only select significant timepoints 
		
		conds = pd.Series(['constant', 'signed_prediction_error', 'pupil baseline'])
		conds_uss = pd.Series(['constant', 'unsigned_prediction_error', 'pupil baseline'])
	   
		sn.set(style='ticks')
		f = pl.figure(figsize=(20,10))
		ax = f.add_subplot(121)
		ax = sn.tsplot(regressed_timecourses_ss[:,:,:], err_style='ci_band', condition=conds, time=timepoints)
		for x in range(len(sig_timepoints)): 
			ax.plot(sig_timepoints[x], np.zeros(len(sig_timepoints[x]))-[0.09, 0.08][x], color=['royalblue','indianred'][x], ms = 1, ls='-.', alpha=0.8) #-[0.09, 0.09, 0.08][x], color=['royalblue','royalblue', 'indigo'][x] #
		ax.text(0, -0.06, 'pvalues \n constant: %s \n signed: %s \n pupil baseline: %s'%(clusters_pval[0], clusters_pval[1], clusters_pval[2]))
		pl.axvline(0, lw=0.25, alpha=0.5, color='k')
		pl.axhline(0, lw=0.25, alpha=0.5, color='k')
		pl.ylabel('beta value')
		pl.xlabel('time (s)')
		sn.despine(offset=10, trim=True)
		simpleaxis(ax)   

		#clustertest unsigned rpe
		clusters= [mne.stats.permutation_cluster_1samp_test(regressed_timecourses_uss[:,:,i])[1] for i in range(regressed_timecourses_uss.shape[2])]
		clusters_pval = [mne.stats.permutation_cluster_1samp_test(regressed_timecourses_uss[:,:,i])[2] for i in range(regressed_timecourses_uss.shape[2])]
		sig_pval = np.concatenate([pval < 0.05 for pval in clusters_pval])
		cluster_timepoints = [timepoints[clusters[i][j]] for i in range(len(clusters)) for j in range(len(clusters[i])) if len(clusters[i])>0] #all cluster timepoints
		sig_timepoints = [val for indx,val in enumerate(cluster_timepoints) if sig_pval[indx]] #only select significant timepoints 

		ax = f.add_subplot(122)
		ax = sn.tsplot(regressed_timecourses_uss[:,:,:], err_style='ci_band', condition=conds_uss, time=timepoints)
		for x in range(len(sig_timepoints)): 
			ax.plot(sig_timepoints[x], np.zeros(len(sig_timepoints[x]))-0.09, color=['royalblue', 'indianred'][x], ms = 1, ls='-.', alpha=0.8) #-[0.09, 0.09, 0.08][x], color=['royalblue','royalblue', 'indigo'][x] #
		ax.text(0, -0.6, 'pvalues \n constant: %s \n unsigned: %s \n pupil baseline: %s'%(clusters_pval[0], clusters_pval[1], clusters_pval[2]))
		pl.axvline(0, lw=0.25, alpha=0.5, color='k')
		pl.axhline(0, lw=0.25, alpha=0.5, color='k')
		pl.ylabel('beta value')
		pl.xlabel('time (s)')
		sn.despine(offset=10, trim=True)
		simpleaxis(ax)
		pl.tight_layout()
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'TD_regression_sep_signed_%s_zscore_fix_trialcorrection_t0.pdf'%standard_deconvolution))



	def deconvolve_ridge_covariates_across_subjects(self, best_sim_params='average_TD_params', baseline=True, analysis_sample_rate=20, smooth_width=10, which_covariates='pupil_baseline_df'): 
		""" deconvolve_ridge_covariates_across_subjects averages deconvolved pupil responses across participants. Ridge_type = 'standard' (only
			standard regressors) or 'full' (standard regressors and value and prediction error) """
		
		if which_covariates == 'pupil_baseline': 
			folder_name = 'deconvolve_full_%iHz_ridge_%s'%(analysis_sample_rate, best_sim_params) 
		elif which_covariates == 'pupil_baseline_df':
			folder_name = 'deconvolve_full_%iHz_ridge_%s_baseline_derivative'%(analysis_sample_rate, best_sim_params) 
		elif which_covariates == 'pupil_df': 
			folder_name = 'deconvolve_full_%iHz_ridge_%s_derivative'%(analysis_sample_rate, best_sim_params) 
		elif which_covariates == 'pupil_baseline_filtered_sound_df': 
			folder_name = 'deconvolve_padded_ridge_regression_%s'%analysis_sample_rate
		elif which_covariates == 'pupil_baseline_filtered_sound_av': 
			folder_name = 'deconvolve_padded_ridge_regression_%s_av_pupil_baseline'%analysis_sample_rate
		elif which_covariates == 'tonic_pupil_baseline_zscore': 
			folder_name = 'deconvolve_ridge_regression_%s_tonic_pupil_zscore'%analysis_sample_rate
		elif which_covariates == 'detrended_tonic_pupil_baseline_zscore': 
			folder_name = 'deconvolve_ridge_regression_%s_detrended_tonic_pupil_zscore'%analysis_sample_rate

		
		ridge_timecourses = np.squeeze(self.gather_data_from_hdfs(group = folder_name , data_type= 'dec_time_course'))		
		time_points = self.gather_data_from_hdfs(group = folder_name , data_type= 'time_points')
		rsquared = self.gather_data_from_hdfs(group = folder_name , data_type= 'rsquared')
		cov_keys = np.concatenate(self.gather_data_from_hdfs(group = folder_name , data_type= 'covariate_keys')[0])
		best_alpha = self.gather_data_from_hdfs(group = folder_name , data_type= 'alpha_value')
		names = [s.subject.initials for s in self.sessions]	
		baseline_times = time_points.mean(axis=0) < 0
		
		#baseline_correction 
		if baseline: 
			ridge_timecourses_s = np.array([[myfuncs.smooth(ridge_timecourses[i,:,j], window_len=smooth_width) - ridge_timecourses[i,baseline_times,j].mean() for i in range(ridge_timecourses.shape[0])] for j in range(ridge_timecourses.shape[-1])]).transpose(0,2,1)
			ridge_timecourses_r = dict(zip(cov_keys, ridge_timecourses_s))	#zip covariate names to corresponding betas 
		else: 
			ridge_timecourses_ns = np.array([[myfuncs.smooth(ridge_timecourses[i,:,j], window_len=smooth_width) for i in range(ridge_timecourses.shape[0])] for j in range(ridge_timecourses.shape[-1])]).transpose(0,2,1)
			ridge_timecourses_r = dict(zip(cov_keys, ridge_timecourses_ns))	
		
		ridge_cue_timecourses = ridge_timecourses_r['colour_times.detrend_tonic_baseline']
		ridge_sound_timecourses = ridge_timecourses_r['sound_times.detrend_tonic_baseline']
		ridge_cue_sound_timecourses = np.array([ridge_cue_timecourses, ridge_sound_timecourses])

		clusters, sig_timepoints = self.permutation_testing_of_deconvolved_responses(ridge_cue_sound_timecourses.transpose(2,1,0), time_points=time_points.mean(axis=0))
		
		#zip ridge_timecourses with correct covariate_keys 
		con_nuis = pd.Series(['blink', 'keypress', 'saccade'])
		con_nuis_b = pd.Series(['blink*detrend_tonic_baseline', 'keypress*detrend_tonic_baseline', 'saccade*detrend_tonic_baseline'])
		# con_nuis_df = pd.Series(['blink*df_base', 'keypress*df_base', 'saccade*df_base'])
		# con_nuis_fil = pd.Series(['blink*filt_baseline', 'keypress*filt_baseline', 'saccade*filt_baseline'])		
		con_cue = pd.Series(['green cue', 'purple cue'])
		con_sound = pd.Series(['loss sound', 'win sound'])
		con_cue_b = pd.Series(['green*detrend_tonic_baseline', 'purple*detrend_tonic_baseline'])
		con_sound_b = pd.Series(['loss*detrend_tonic_baseline', 'win*detrend_tonic_baseline'])
		# con_st_df = pd.Series(['green*df_base', 'purple*df_base', 'loss*df_base', 'win*df_base'])
		# con_st_f = pd.Series(['green*filt_baseline', 'purple*filt_baseline', 'loss*filt_baseline', 'win*filt_baseline'])
		con_td = pd.Series(['TD Expected Value', 'TD Reward Prediction Error'])	
		con_cs = pd.Series(['Cue*detrend_tonic_baseline', 'Outcome*detrend_tonic_baseline'])	
				
		#make empty lists to order betas meaningfully
		nuis, nuis_bl, nuis_df = [],[],[]
		cue, cue_bl, sound, sound_bl = [],[],[],[]
		td_reg, c_s = [],[] 

		#group average of all regressors 	
		hues = sn.color_palette()#select green and purple
		green_purple = {hues[1],hues[3]} 
		red_green = {hues[2], hues[1]}
		
		f = pl.figure()
		f.text(0.5, 0.99, 'Ridge deconvolution detrended tonic pupil baseline', ha='center', va='center')		
		s = f.add_subplot(421)
		s.set_title('IRF')
		for cov in ['blink.gain', 'keypress.gain', 'saccade.gain']: 			
			nuis.append(ridge_timecourses_r[cov])
		nuisance = np.array(nuis).transpose(2,1,0)
		sn.tsplot(nuisance[:,:,:], condition=con_nuis, time=time_points.mean(axis = 0))
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')	
		sn.despine(offset=10)	
		s = f.add_subplot(422)
		s.set_title('IRF * detrend_tonic_pupil_zscore')
		for cov in ['blink.detrend_tonic_baseline', 'keypress.detrend_tonic_baseline', 'saccade.detrend_tonic_baseline']: 			
			nuis_bl.append(ridge_timecourses_r[cov])
		nuisance_bl = np.array(nuis_bl).transpose(2,1,0)
		sn.tsplot(nuisance_bl[:,:,:], condition=con_nuis_b, time=time_points.mean(axis = 0), linestyle='--')
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')	
		sn.despine(offset=10)	
		s = f.add_subplot(423)
		for cov in ['cue_green.gain', 'cue_purple.gain']: 
			cue.append(ridge_timecourses_r[cov])
		cues = np.array(cue).transpose(2,1,0)
		sn.tsplot(cues[:,:,:], condition=con_cue, time=time_points.mean(axis = 0), color=green_purple)
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')	
		sn.despine(offset=10)	
		s = f.add_subplot(424)
		for cov in ['cue_green.detrend_tonic_baseline', 'cue_purple.detrend_tonic_baseline']: 
			cue_bl.append(ridge_timecourses_r[cov])
		cues_bl = np.array(cue_bl).transpose(2,1,0)
		sn.tsplot(cues_bl[:,:,:], condition=con_cue_b, time=time_points.mean(axis = 0), linestyle='--', color=green_purple)
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')	
		sn.despine(offset=10)	
		s = f.add_subplot(425)
		for cov in ['sound_loss.gain', 'sound_win.gain']:
			sound.append(ridge_timecourses_r[cov])
		sounds = np.array(sound).transpose(2,1,0)
		sn.tsplot(sounds[:,:,:], condition=con_sound, time=time_points.mean(axis = 0), color=red_green)
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')	
		sn.despine(offset=10)
		s = f.add_subplot(426)
		for cov in ['sound_loss.detrend_tonic_baseline', 'sound_win.detrend_tonic_baseline']: 
			sound_bl.append(ridge_timecourses_r[cov])
		sounds_bl = np.array(sound_bl).transpose(2,1,0)
		sn.tsplot(sounds_bl[:,:,:], condition=con_sound_b, time=time_points.mean(axis = 0), color=red_green, linestyle='--')
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')	
		sn.despine(offset=10)
		s = f.add_subplot(427)
		for cov in ['colour_times.detrend_tonic_baseline', 'sound_times.detrend_tonic_baseline']:
			c_s.append(ridge_timecourses_r[cov])
		colour_sound = np.array(c_s).transpose(2,1,0)
		sn.tsplot(colour_sound[:,:,0], condition='Cue*detrend_tonic_baseline', time=time_points.mean(axis=0), linestyle='--')
		for x in range(len(sig_timepoints)): 
			s.plot(sig_timepoints[x], np.zeros(len(sig_timepoints[x]))-0.04, ls='--', lw=4, alpha = 0.9) #
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')	
		sn.despine(offset=10)
		s = f.add_subplot(428)
		sn.tsplot(colour_sound[:,:,1], condition='Outcome*detrend_tonic_baseline', time=time_points.mean(axis=0), linestyle='--')	
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')	
		sn.despine(offset=10)
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'ridge_regression_detrend_tonic_pupil_baseline_sound_interactions_seperate.pdf'))

		#per participant plot of TD_value and TD_RPE regressor betas 				
		fig = pl.figure(figsize=(12,12))
		for idx, name in enumerate(names):
			s = fig.add_subplot(5,6,idx+1)       
			sn.tsplot(TD_regressors[idx,:,0], err_style="ci_band", time = time_points.mean(axis=0), color='royalblue', condition='TD_value')
			sn.tsplot(TD_regressors[idx,:,1], err_style="ci_band", time = time_points.mean(axis=0), color='darkseagreen', condition='TD_RPE')             
			s.set_title(names[idx])
			pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
			pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')				
			sn.despine(offset=10)
		pl.tight_layout()
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'TD_RPE_per_participant_df.pdf'))

	
	def ridge_phasic_pupil_tonic_baselines_across_subjects(self, baseline=True, smooth_width = 10): 
		shell()
		folder_name = 'ridge_phasic_6_tonic_baselines' 
		#list of pd.DataFrames
		df_ridge_timecourses = self.gather_dataframes_from_hdfs(group = folder_name , data_type= 'deconvolved_pupil_timecourses')
		#numpy arrays 
		ridge_timecourses = self.gather_data_from_hdfs(group = folder_name , data_type= 'deconvolved_pupil_timecourses')
		time_points = self.gather_data_from_hdfs(group = folder_name , data_type= 'time_points')
		best_alpha = self.gather_data_from_hdfs(group = folder_name , data_type= 'alpha_value')
		names = [s.subject.initials for s in self.sessions]	
		baseline_times = time_points.mean(axis=0) < 0
		keys = df_ridge_timecourses[0].keys()

		if baseline: 
			ridge_timecourses_s = np.array([[myfuncs.smooth(ridge_timecourses[i,:,j], window_len=smooth_width) - ridge_timecourses[i,baseline_times,j].mean() for i in range(ridge_timecourses.shape[0])] for j in range(ridge_timecourses.shape[-1])]).transpose(0,2,1)
			ridge_timecourses_s = dict(zip(keys, ridge_timecourses_s))
		else: 
			ridge_timecourses_s = np.array([[myfuncs.smooth(ridge_timecourses[i,:,j], window_len=smooth_width) for i in range(ridge_timecourses.shape[0])] for j in range(ridge_timecourses.shape[-1])]).transpose(0,2,1)
			ridge_timecourses_s = dict(zip(keys, ridge_timecourses_s))

		#colour dicts
		purples = sn.color_palette("Oranges")
		blues = sn.color_palette("Blues")
		reds = sn.color_palette("Reds")
		greens = sn.color_palette("Greens")
		basic_hues = sn.color_palette()
		cue_hues =  {basic_hues[1], basic_hues[3], basic_hues[0], basic_hues[2]} #green, purple, red, blue
		sound_hues = {basic_hues[0], basic_hues[2]} #red, blue 

		#conditions
		con_nuis = pd.Series(['blink', 'keypress', 'saccade'])
		con_cue = pd.Series(['green cue', 'purple cue', 'cue_low', 'cue_high'])
		con_sound = pd.Series(['loss sound', 'win sound'])
		con_hz = pd.Series(['0.1Hz', '0.04Hz', '0.02Hz', '0.009Hz', '0.004Hz', '0.002Hz'])

		nuis, cue, sound, kp, cue_l, cue_h, sound_l, sound_h, ct, st =[],[],[],[],[],[],[],[],[],[]
		#line plot 
		f = pl.figure(figsize = (10,7)) 
		s = f.add_subplot(431)			
		for cov in ['blink.gain', 'keypress.gain', 'saccade.gain']: 	
			nuis.append(ridge_timecourses_s[cov])
		nuisance = np.array(nuis).transpose(2,1,0)
		sn.tsplot(nuisance[:,:,:], condition=con_nuis, time=time_points.mean(axis = 0), color=basic_hues)
		s.set_title('Nuisance', fontsize=10)
		sn.despine(offset=5)
		s = f.add_subplot(432)
		for cov in ['cue_green.gain', 'cue_purple.gain', 'cue_low.gain', 'cue_high.gain']: 
			cue.append(ridge_timecourses_s[cov])
		cues = np.array(cue).transpose(2,1,0)
		sn.tsplot(cues[:,:,:], condition=con_cue, time=time_points.mean(axis = 0), color=cue_hues)
		s.set_title('Prediction', fontsize=10)
		sn.despine(offset=5)
		s = f.add_subplot(433)
		for cov in ['sound_loss.gain', 'sound_win.gain']: 
			sound.append(ridge_timecourses_s[cov])
		sounds = np.array(sound).transpose(2,1,0)
		sn.tsplot(sounds[:,:,:], condition=con_sound, time=time_points.mean(axis = 0), color=sound_hues)
		s.set_title('Outcome', fontsize=10)
		sn.despine(offset=5)
		s = f.add_subplot(434)
		for cov in['keypress.10000', 'keypress.04573', 'keypress.02091', 'keypress.00956', 'keypress.00437', 'keypress.00200']: 
			kp.append(ridge_timecourses_s[cov])
		keypress = np.array(kp).transpose(2,1,0)
		sn.tsplot(keypress[:,:,:], condition=con_hz, time=time_points.mean(axis = 0), color=purples, linestyle='--')
		s.set_title('keypress*bl')
		sn.despine(offset=5)
		s = f.add_subplot(438)
		for cov in ['cue_low.10000', 'cue_low.04573', 'cue_low.02091', 'cue_low.00956', 'cue_low.00437', 'cue_low.00200']: 
			cue_l.append(ridge_timecourses_s[cov])
		cues_l = np.array(cue_l).transpose(2,1,0)
		sn.tsplot(cues_l[:,:,:], condition=con_hz, time=time_points.mean(axis = 0), color = reds, linestyle='--' )
		s.set_title('low prediction * bl')
		sn.despine(offset=5)
		s = f.add_subplot(4,3,11)
		for cov in ['cue_high.10000', 'cue_high.04573', 'cue_high.02091', 'cue_high.00956', 'cue_high.00437', 'cue_high.00200']: 
			cue_h.append(ridge_timecourses_s[cov])
		cues_h = np.array(cue_h).transpose(2,1,0)
		sn.tsplot(cues_h[:,:,:], condition=con_hz, time=time_points.mean(axis = 0), color = blues, linestyle='--')
		s.set_title('high prediction * bl')
		sn.despine(offset=5)
		s = f.add_subplot(439)
		for cov in ['sound_loss.10000', 'sound_loss.04573', 'sound_loss.02091', 'sound_loss.00956', 'sound_loss.00437', 'sound_loss.00200']: 
			sound_l.append(ridge_timecourses_s[cov])
		sounds_l = np.array(sound_l).transpose(2,1,0)
		sn.tsplot(sounds_l[:,:,], condition=con_hz, time=time_points.mean(axis = 0), color= reds, linestyle='--')
		s.set_title('loss * bl')
		sn.despine(offset=5)
		s = f.add_subplot(4,3,12)
		for cov in ['sound_win.10000', 'sound_win.04573', 'sound_win.02091', 'sound_win.00956', 'sound_win.00437', 'sound_win.00200']: 
			sound_h.append(ridge_timecourses_s[cov])
		sounds_h = np.array(sound_h).transpose(2,1,0)
		sn.tsplot(sounds_h[:,:,:], condition=con_hz, time=time_points.mean(axis = 0), color=blues, linestyle='--')
		s.set_title('win * bl')
		sn.despine(offset=5)
		s = f.add_subplot(4,3,5)
		for cov in ['colour_times.10000', 'colour_times.04573', 'colour_times.02091', 'colour_times.00956', 'colour_times.00437', 'colour_times.00200']:
			ct.append(ridge_timecourses_s[cov]) 
		colour_times = np.array(ct).transpose(2,1,0)
		sn.tsplot(colour_times[:,:,:], condition=con_hz, time=time_points.mean(axis = 0), color=purples, linestyle='--')
		s.set_title('all predictions * bl')
		sn.despine(offset=5)
		s = f.add_subplot(4,3,6)
		for cov in ['sound_times.10000', 'sound_times.04573', 'sound_times.02091', 'sound_times.00956', 'sound_times.00437', 'sound_times.00200']:
			st.append(ridge_timecourses_s[cov])
		sound_times = np.array(st).transpose(2,1,0)
		sn.tsplot(sound_times[:,:,:], condition=con_hz, time=time_points.mean(axis = 0), color = purples, linestyle='--')
		s.set_title('all outcomes * bl')
		sn.despine(offset=5)
		pl.tight_layout()
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'ridge_phasic_6_tonic_baselines_bl_%s.pdf'%str(baseline)))



		hoi = []
		#imshow plot 
		fig = pl.figure()
		s = f.add_subplot(331)
		for cov in ['keypress.10000', 'keypress.04573', 'keypress.02091', 'keypress.00956', 'keypress.00437', 'keypress.00200']:
			hoi.append(ridge_timecourses_s[cov]) 
		doei = np.array(np.mean(hoi, axis=2))
		s.imshow(doei)
		s = f.add_subplot(332)
		for i,cov in enumerate(['cue_low.10000', 'cue_low.04573', 'cue_low.02091', 'cue_low.00956', 'cue_low.00437', 'cue_low.00200']): 
			im = ax.imshow(timepoints, ridge_timecourses_s[cov], cmap='YlGnBu')
			im.set_title('cue_low')
		s = f.add_subplot(333)
		for i,cov in enumerate(['cue_high.10000', 'cue_high.04573', 'cue_high.02091', 'cue_high.00956', 'cue_high.00437', 'cue_high.00200']): 
			im = ax.imshow(timepoints, ridge_timecourses_s[cov], cmap='YlGnBu')
			im.set_title('cue_high')
		s = f.add_subplot(334)
		for i,cov in enumerate(['sound_low.10000', 'sound_low.04573', 'sound_low.02091', 'sound_low.00956', 'sound_low.00437', 'sound_low.00200']): 
			im = ax.imshow(timepoints, ridge_timecourses_s[cov], cmap='YlGnBu')
			im.set_title('sound_low')
		s = f.add_subplot(335)
		for i,cov in enumerate(['sound_high.10000', 'sound_high.04573', 'sound_high.02091', 'sound_high.00956', 'sound_high.00437', 'sound_high.00200']): 
			im = ax.imshow(timepoints, ridge_timecourses_s[cov], cmap='YlGnBu')
			im.set_title('sound_high')
		s = f.add_subplot(336)
		for i,cov in enumerate(['colour_times.10000', 'colour_times.04573', 'colour_times.02091', 'colour_times.00956', 'colour_times.00437', 'colour_times.00200']): 
			im = ax.imshow(timepoints, ridge_timecourses_s[cov], cmap='YlGnBu')
			im.set_title('colour_times')
		s = f.add_subplot(337)
		for i,cov in enumerate(['sound_times.10000', 'sound_times.04573', 'sound_times.02091', 'sound_times.00956', 'sound_times.00437', 'sound_times.00200']): 
			im = ax.imshow(timepoints, ridge_timecourses_s[cov], cmap='YlGnBu')
			im.set_title('sound_times')
			im.set_interpolation('bicubic')   
			f.text(0.5, 0.02, 'time (s)', ha='center', va='center', fontsize='medium')
		f.text(0.08, 0.5, 'frequency', ha='center', va='center', rotation='vertical', fontsize='medium')
		f.text(0.5, 0.98, 'Time-frequency correlation of phasic pupil responses and tonic baselines', ha='center', va='center', fontsize='large')
		f.subplots_adjust(right=0.8)
		cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
		f.text(0.88, 0.88, 'r', ha='center', va='center', fontsize='medium')
		f.colorbar(im, cax=cbar_ax)
	




	def correlate_tonic_phasic_pupil_around_keypress_across_subjects(self): 
		""" correlate_tonic_phasic_pupil_around_keypress_across_subjects calculates the correlation of tonic and phasic pupil signals around key press.  Correlations are calculated for phasic and tonic [tonic, detrended_tonic, filtered_detrended_tonic] pupil signals. """
		
		folder_name = 'tonic_phasic_pupil_keypress' 		
		#pupil change interval [-1,1] around keypress 
		tonic = self.gather_data_from_hdfs(group = folder_name , data_type= 'tonic_pupil')	
		tonic_detrended_padded = self.gather_data_from_hdfs(group = folder_name , data_type= 'tonic_pupil_detrended_padded')	
		tonic_filter_detrend = self.gather_data_from_hdfs(group = folder_name , data_type= 'tonic_pupil_filter_detrend')	
		pupil_change_phasic = self.gather_data_from_hdfs(group = folder_name , data_type= 'phasic_pupil_change')	
		pupil_change_phasic_padded = self.gather_data_from_hdfs(group = folder_name , data_type= 'phasic_pupil_change_padded')
		phasic_pupil_1s_after_keypress = self.gather_data_from_hdfs(group = folder_name , data_type= 'phasic_pupil_1s_after_keypress')
		padded_phasic_pupil_1s_after_keypress = self.gather_data_from_hdfs(group = folder_name , data_type= 'padded_phasic_pupil_1s_after_keypress')

			
		data = {'tonic': tonic, 'tonic_filter_detrend': tonic_filter_detrend, 'tonic_detrend_padded': tonic_detrended_padded, 'phasic': pupil_change_phasic, 'phasic_padded': pupil_change_phasic_padded, 'phasic_pupil_1s_after_keypress': phasic_pupil_1s_after_keypress, 'padded_phasic_pupil_1s_after_keypress': padded_phasic_pupil_1s_after_keypress}		
		
		names = [s.subject.initials for s in self.sessions]	

		cor_tp, cor_tp_detrend, cor_tp_filter_detrend = [], [], []
		for signals in zip(data['tonic'], data['phasic']):
			cor_tp.append(sp.stats.spearmanr(signals[0], signals[1])[0])
		for signals in zip(data['tonic_detrend_padded'], data['phasic_padded']):
			cor_tp_detrend.append(sp.stats.spearmanr(signals[0], signals[1])[0])
		for signals in zip(data['tonic_filter_detrend'], data['phasic']):
			cor_tp_filter_detrend.append(sp.stats.spearmanr(signals[0], signals[1])[0])
		grand_mean_correlations = np.mean([np.array(cor_tp), np.array(cor_tp_detrend), np.array(cor_tp_filter_detrend)], axis=0)

		demeaned_correlations=[]
		for correlations in [cor_tp, cor_tp_detrend, cor_tp_filter_detrend]: 
			demean = correlations - grand_mean_correlations
			demeaned_correlations.append(demean)

		demeaned_correlation_dict = {'cor_tonic_phasic': demeaned_correlations[0], 'cor_tonic_phasic_detrend': demeaned_correlations[1], 'cor_tonic_phasic_filter_detrend': demeaned_correlations[2]}
		#shell() 
		#compare correlation changes phasic & tonic pupil for tonic, tonic_detrend, tonic_filter_detrend 
		hues = sn.color_palette()
		fig = pl.figure(figsize = (5,5))	
		sn.regplot(x=demeaned_correlation_dict['cor_tonic_phasic'], y=demeaned_correlation_dict['cor_tonic_phasic_detrend'], scatter=True, color=hues[1])
		sn.regplot(x=demeaned_correlation_dict['cor_tonic_phasic'], y=demeaned_correlation_dict['cor_tonic_phasic_filter_detrend'], scatter=True, color=hues[2])
		sn.regplot(x=demeaned_correlation_dict['cor_tonic_phasic_detrend'], y=demeaned_correlation_dict['cor_tonic_phasic_filter_detrend'], scatter=True, color=hues[3])
		pl.legend(['tonic_phasic/tonic_detrend_phasic', 'tonic_phasic/tonic_filter_detrend_phasic', 'tonic_detrend_phasic/tonic_filter_detrend_phasic'])
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'correlation_comparison_tonic_phasic_pupil_signals.pdf'))

		#tonic phasic correlation pupil change
		fig = pl.figure(figsize=(10,10))
		fig.text(0.5, 0.01, 'average tonic pupil around keypress', ha='center', va='center')
		fig.text(0.01, 0.5, 'phasic pupil change around keypress', ha='center', va='center', rotation='vertical')
		fig.text(0.5, 0.97, 'Correlation phasic/tonic', fontsize=12)
		for idx, name in enumerate(names):
			s = fig.add_subplot(5,6,idx+1)    
			sn.regplot(x=data['tonic'][idx], y=data['phasic'][idx], scatter=True)   
			s.set_title(names[idx])		
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'corr_tonic_phasic_pupil_change2.pdf'))

		fig = pl.figure(figsize=(10,10))
		fig.text(0.5, 0.01, 'average tonic pupil around keypress', ha='center', va='center')
		fig.text(0.01, 0.5, 'phasic pupil 1s. after keypress', ha='center', va='center', rotation='vertical')
		fig.text(0.5, 0.97, 'Correlation phasic/tonic', fontsize=12)
		for idx, name in enumerate(names):
			s = fig.add_subplot(5,6,idx+1)    
			sn.regplot(x=data['tonic'][idx], y=data['phasic_pupil_1s_after_keypress'][idx], scatter=True)   
			s.set_title(names[idx])		
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'corr_tonic_phasic_1s_after_keypress.pdf'))
	
		#detrended tonic phasic correlation pupil change
		fig = pl.figure(figsize=(10,10))
		fig.text(0.5, 0.01, 'detrended tonic pupil change', ha='center', va='center')
		fig.text(0.01, 0.5, 'phasic pupil change', ha='center', va='center', rotation='vertical')
		fig.text(0.5, 0.97, 'Correlation phasic/tonic_detrend', fontsize=12)
		for idx, name in enumerate(names):
			s = fig.add_subplot(5,6,idx+1)    
			sn.regplot(x=data['tonic_detrend_padded'][idx], y=data['phasic_padded'][idx], scatter=True)   
			s.set_title(names[idx])		
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'corr_detrended_tonic_phasic_pupil_change2.pdf'))

		#detrended filtered tonic phasic correlation pupil change 
		fig = pl.figure(figsize=(10,10))
		fig.text(0.5, 0.01, 'filtered detrended tonic pupil change', ha='center', va='center')
		fig.text(0.01, 0.5, 'phasic pupil change', ha='center', va='center', rotation='vertical')
		fig.text(0.5, 0.97, 'Correlation phasic/tonic_filter_detrend', fontsize=12)
		for idx, name in enumerate(names):
			s = fig.add_subplot(5,6,idx+1)    
			sn.regplot(x=data['tonic_filter_detrend'][idx], y=data['phasic'][idx], scatter=True)   
			s.set_title(names[idx])		
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'corr_filtered_detrended_tonic_phasic_pupil_change2.pdf'))



	def deconvolve_sound_using_baseline_slope_across_subjects(self, analysis_sample_rate=20, smooth_width=10, baseline=True, t_start=-1, t_end=6): 
		""" """
			
		folder_name ='positive_negative_slope_sound_deconvolution' 		
		deconvolved_timecourses = np.squeeze(self.gather_data_from_hdfs(group = folder_name , data_type= 'deconvolved_pupil_timecourses'))		
		rsquared = self.gather_data_from_hdfs(group = folder_name , data_type= 'rsquared')
		cov_keys = np.concatenate(self.gather_data_from_hdfs(group = folder_name , data_type= 'covariate_keys')[0])
		names = [s.subject.initials for s in self.sessions]	
		time_points = np.linspace(t_start, t_end, deconvolved_timecourses.shape[1])
		baseline_times = time_points < 0

		#baseline_correction 
		if baseline: 
			deconvolved_timecourses_s = np.array([[myfuncs.smooth(deconvolved_timecourses[i,:,j], window_len=smooth_width) - deconvolved_timecourses[i,baseline_times,j].mean() for i in range(deconvolved_timecourses.shape[0])] for j in range(deconvolved_timecourses.shape[-1])]).transpose(0,2,1)
			deconvolved_timecourses_r = dict(zip(cov_keys, deconvolved_timecourses_s))	#zip covariate names to corresponding betas 
		else: 
			deconvolved_timecourses_ns = np.array([[myfuncs.smooth(deconvolved_timecourses[i,:,j], window_len=smooth_width) for i in range(deconvolved_timecourses.shape[0])] for j in range(deconvolved_timecourses.shape[-1])]).transpose(0,2,1)
			deconvolved_timecourses_r = dict(zip(cov_keys, deconvolved_timecourses_ns))	

		con_win = pd.Series(['win_negative_slope', 'win_positive_slope'])
		con_loss = pd.Series(['loss_negative_slope', 'loss_positive_slope'])
		#make empty lists to order betas meaningfully
		loss, win = [],[]		
		#group average of all regressors 	
		f = pl.figure()		
		s = f.add_subplot(121)
		for cov in ['win_negative_slope.gain', 'win_positive_slope.gain']: 			
			win.append(deconvolved_timecourses_r[cov])
		win_sound = np.array(win).transpose(2,1,0)
		sn.tsplot(win_sound[:,:,:], condition=con_win, time=time_points)
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')	
		sn.despine(offset=10)	
		s = f.add_subplot(122)
		for cov in ['loss_negative_slope.gain', 'loss_positive_slope.gain']: 
			loss.append(deconvolved_timecourses_r[cov])
		loss_sound = np.array(loss).transpose(2,1,0)
		sn.tsplot(loss_sound[:,:,:], condition=con_loss, time=time_points)
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')	
		sn.despine(offset=10)	
	

		
		


	def BISBAS(self, baseline=True, analysis_sample_rate=20, best_sim_params='average_TD_params', smooth_width=10):
		"""correlate BIS BAS scores with deconvolved pupil signals from the ridge regression in time period time_period_of_interest """ 
		
		#load bis bas scores
		bis_bas_scores = pd.read_csv(os.path.join(self.grouplvl_data_dir,'BIS_BAS.csv'), sep=';', header=False)
		bis_bas_participants = list(bis_bas_scores['ppn name'])

		#only select data from participants with bis bas scores 
		self.all_sessions = np.copy(self.sessions)
		self.sessions = [s for s in self.sessions if s.subject.initials in bis_bas_participants] 

		#all BIS BAS variables
		BAS_Drive = bis_bas_scores.BAS_Drive.as_matrix()
		BAS_Fun = bis_bas_scores.BAS_Fun_Seeking.as_matrix()
		BAS_Rew = bis_bas_scores.BAS_Reward_Responsiveness.as_matrix()
		BAS = np.sum(np.array([BAS_Drive, BAS_Fun, BAS_Rew]), axis=0)
		BIS = bis_bas_scores.BIS.as_matrix()

		z_scored_bisbas =[]
		for variable in [BAS_Drive, BAS_Fun, BAS_Rew,BAS, BIS]: #BAS_Drive, BAS_Fun, BAS_Rew, 
			normalised = (variable - np.mean(variable))/variable.std()
			z_scored_bisbas.append(normalised)


		folder_name = 'deconvolve_full_%iHz_ridge_%s'%(analysis_sample_rate, best_sim_params) 
		ridge_timecourses = np.squeeze(self.gather_data_from_hdfs(group = folder_name , data_type= 'dec_time_course'))		
		time_points = self.gather_data_from_hdfs(group = folder_name , data_type= 'time_points')
		cov_keys = np.concatenate(self.gather_data_from_hdfs(group = folder_name , data_type= 'covariate_keys')[0])
		baseline_times = time_points.mean(axis=0) < 0
		
		#baseline_correction 
		if baseline: 
			ridge_timecourses_s = np.array([[myfuncs.smooth(ridge_timecourses[i,:,j], window_len=smooth_width) - ridge_timecourses[i,baseline_times,j].mean() for i in range(ridge_timecourses.shape[0])] for j in range(ridge_timecourses.shape[-1])]).transpose(0,2,1)
			ridge_timecourses_r = dict(zip(cov_keys, ridge_timecourses_s))	#zip covariate names to corresponding betas 
		else: 
			ridge_timecourses_ns = np.array([[myfuncs.smooth(ridge_timecourses[i,:,j], window_len=smooth_width) for i in range(ridge_timecourses.shape[0])] for j in range(ridge_timecourses.shape[-1])]).transpose(0,2,1)
			ridge_timecourses_r = dict(zip(cov_keys, ridge_timecourses_ns))	

		#get TD_value and TD_rpe regressor
		td_reg=[]
		for cov in ['colour_times.TD_value', 'sound_times.signed_RPE']: 
			td_reg.append(ridge_timecourses_r[cov])
		TD_regressors = np.array(td_reg).transpose(2,1,0)
		TD_value = TD_regressors[:,:,0]; TD_rpe = TD_regressors[:,:,1]

		#correlation result matrix
		TD_value_cor_results = np.zeros((120,len(z_scored_bisbas),3)) #timepoints x  bis bas scale x rho, pvalue, fisher rho 
		TD_rpe_cor_results = np.zeros((120,len(z_scored_bisbas),3))
	
		#loop over timepoints: 
		for j in range(TD_value.shape[1]): 
			data_value = np.array([TD_value[s][j] for s in range(len(self.sessions))]).ravel() 
			data_rpe = np.array([TD_rpe[s][j] for s in range(len(self.sessions))]).ravel()
			#loop over bis bas scales 
			for bb in range(len(z_scored_bisbas)): 
				rho, pvalue = sp.stats.spearmanr(TD_value[:,j], z_scored_bisbas[bb])
				fisher_rho = np.arctanh(rho)				
				TD_value_cor_results[j,bb] = fisher_rho, rho, pvalue
				rho, pvalue = sp.stats.spearmanr(TD_rpe[:,j], z_scored_bisbas[bb])
				fisher_rho = np.arctanh(rho)
				TD_rpe_cor_results[j,bb] = fisher_rho, rho, pvalue

		
		
		sig_pvals_val = [np.arange(time_points.shape[1])[TD_value_cor_results[:,i,2]< 0.05] for i in range(TD_value_cor_results.shape[1])] #BAS Reward responsiveness
		sig_pvals_rpe = [np.arange(time_points.shape[1])[TD_rpe_cor_results[:,i,2]< 0.05] for i in range(TD_rpe_cor_results.shape[1])]
		

		f = pl.figure(figsize=(8,5)) 
		s = f.add_subplot(121)
		for i in range(TD_value_cor_results.shape[1]): 
			pl.plot(np.mean(time_points, axis=0), TD_value_cor_results[:,i,0])
			#pl.plot(sig_pvals_val[i], np.zeros(len(sig_pvals_val[i]))-0.4, color = ['royalblue', 'indianred', 'yellow'][i], alpha=0.8)
			s.set_title('TD_value BIS/BAS correlation')
			pl.ylabel('fisher rho')
			pl.xlabel('time')
		pl.legend(['BAS_Drive', 'BAS_Fun', 'BAS_Rew','BAS', 'BIS'])
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')	
		sn.despine(offset=10)
		pl.tight_layout()	
		s = f.add_subplot(122)
		for i in range(TD_rpe_cor_results.shape[1]): 
			pl.plot(np.mean(time_points, axis=0), TD_rpe_cor_results[:,i,2])
			s.set_title('TD_rpe BIS/BAS correlation')
			pl.ylabel('fisher rho')
			pl.xlabel('time')
		pl.legend(['BAS_Drive', 'BAS_Fun', 'BAS_Rew', 'BAS', 'BIS'])
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')	
		sn.despine(offset=10)
		pl.tight_layout()	
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'BIS_BAS_ridge_fisher_rho_p_val_baseline_%s.pdf'%str(baseline)))

	

	def pupil_baseline_amplitude_overview(self, signal_for_baseline='tonic_at_fixation'):
		"""make overview plots of the correlation between baseline pupil size and pupil response amplitude on sound and colour """

		#load variables 
		pupil_at_fixation = self.gather_data_from_hdfs(group = 'pupil_baseline_amplitude_%s'%signal_for_baseline, data_type= 'fix_pupil')
		#remove last trial of phasic_pupil_at_fixation because sound amplitude calculation missed 1 trial due to too little samples 
		pupil_at_fixation_sound_amp = [pupil_at_fixation[subj][:-1] for subj in range(len(pupil_at_fixation))]
		sound_amp = self.gather_data_from_hdfs(group = 'pupil_baseline_amplitude_%s'%signal_for_baseline, data_type= 'sound_amp')
		colour_amp = self.gather_data_from_hdfs(group = 'pupil_baseline_amplitude_%s'%signal_for_baseline, data_type= 'colour_amp')
		corr_baseline_sound_amp = self.gather_data_from_hdfs(group = 'pupil_baseline_amplitude_%s'%signal_for_baseline, data_type= 'corr_pupil_fix_sound_amp')
		corr_baseline_colour_amp = self.gather_data_from_hdfs(group = 'pupil_baseline_amplitude_%s'%signal_for_baseline, data_type= 'corr_pupil_fix_colour_amp')
		names = [s.subject.initials for s in self.sessions]
		
		#panel plot correlation pupil baseline response amplitude [sound response & cue response]
		fig = pl.figure(figsize=(15,15))
		for i, name in enumerate(names):
			s = fig.add_subplot(5,6,i+1)       #subplots 
			sn.regplot(pupil_at_fixation[i], colour_amp[i], scatter_kws={"alpha": .3}, line_kws={"alpha": .8})  
			sn.regplot(pupil_at_fixation_sound_amp[i], sound_amp[i], scatter_kws={"alpha": .3}, line_kws={"alpha": .8})  
			pl.legend(['cue: r=%.2f'%(corr_baseline_colour_amp[i][0]), 'sound: r=%.2f' %corr_baseline_sound_amp[i][0], ], loc='best', fontsize=8)   #           
			s.set_title(names[i])
			s.yaxis.label.set_size(12)
			s.xaxis.label.set_size(12)
			fig.text(0.5, 0.02, '%s pupil at fixation'%signal_for_baseline, ha='center', va='center', fontsize=11)
			fig.text(0.02, 0.5, 'pupil response amplitude ', ha='center', va='center', rotation='vertical', fontsize=11)
		pl.savefig(os.path.join(self.grouplvl_plot_dir, '_correlation_pupil_at_fixation_sound_cue_amplitude_%s_hp_01Hz.pdf'%signal_for_baseline)) 
	

	def baseline_phasic_pupil_correlations(self):
		"""inspect baseline amplitude correlations of all frequency_bank frequencies and phasic ampitudes. Amplitude is calculated in two 
		different ways. 1) simple: event baseline is subtracted from sound window and the peak within the sound window is used. 
		2) projection: per trial sound interval is demeaned using the event baseline and dot product is taken of the deconvolved prediction error kernel and each trial's sound response """


		for data in ['pupil_baseline_correlations_simple']:#, , 'pupil_baseline_diff_correlations_simple''pupil_baseline_correlations_av_projection', 'pupil_baseline_correlations_av_projection_diff', 'pupil_baseline_correlations_projection', 'pupil_baseline_correlations_projection_diff']: 
			sj_data_lst = self.gather_dataframes_from_hdfs(group = 'pupil_all_baselines_phasic', data_type = data)
			sj_data_c_array = np.array([np.array(p['correlations'], dtype = float) for p in sj_data_lst])
			f_array = np.array([np.array(p['frequencies'], dtype = float) for p in sj_data_lst])[0]	

			f = pl.figure() 
			pl.plot(f_array, np.mean(sj_data_c_array, axis=0))
			pl.title('baseline amplitude correlations of data %s'%data, fontsize=10)
			pl.xlabel('frequency', fontsize=10)
			pl.ylabel('correlation', fontsize=10)
			shell()
			#pl.savefig(os.path.join(self.grouplvl_plot_dir, 'baseline_phasic_corr_%s_.pdf'%data)) 
		sj_data_low_freq_array = sj_data_c_array[:,:-2]
		low_freqs = f_array[:-2]

		keys = self.gather_data_from_hdfs(group = 'deconvolve_pupil_around_keypress_20_tonic_baseline_zscore_True', data_type = 'keys')[0]
		keys = [k.rpartition('.')[-1] for k in keys]
		freqs = np.array([float('0.' + key) for key in keys])
		freq_order = np.argsort(freqs)	

		all_rsquared=[]	
		#loop over rsquared results offilter_bank deconvolutions	
		for idx in freq_order: 
			rsquared = self.gather_data_from_hdfs(group = 'deconvolve_pupil_around_keypress_20_tonic_baseline_zscore_True', data_type = 'rsquared_%s_hz'%keys[idx])
			all_rsquared.append(np.squeeze(rsquared))
		all_rsquared = np.array(all_rsquared)
		av_all_rsquared=np.mean(np.array(all_rsquared), axis=1)
	
		corr_rsquared_amp_bl=[]
		fisher_z = []
		f = pl.figure(figsize=(10,10))
		sn.set(font_scale=0.6, style="ticks")	
		for i, name in enumerate(names):  
			s = f.add_subplot(5,6,i+1)
			s.set_xscale('log')
			pl.plot(low_freqs, all_rsquared[:,i], color='#9d0216')
			pl.plot(low_freqs, sj_data_low_freq_array[i,:], color='#03719c')
			pl.axhline(0, color = 'k', linewidth=0.25) 
			pl.legend(['R2', 'b/a cor'])
			corr_rsquared_amp_bl.append(sp.stats.spearmanr(all_rsquared[:,i], sj_data_low_freq_array[i,:]))
			fisher_z.append(np.arctanh(corr_rsquared_amp_bl[i][0]))				
			s.text(0.02, sj_data_low_freq_array[i,:].mean(), 'r=%.2f \np=%.2f'%(corr_rsquared_amp_bl[i][0],corr_rsquared_amp_bl[i][1]), fontsize=6)		
			s.set_title(name)
		pl.axhline(0, color = 'k', linewidth=0.25) 		
		f.text(0.5, 0.01, 'frequency', fontsize=8)
		f.text(0.47, 0.98, 'per subject R2 b/a correlations', fontsize=8)
		pl.tight_layout()
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'baseline_amplitude_r2_correlations.pdf')) 

		corr_rsquared_amp_bl = np.array(corr_rsquared_amp_bl)
		corr_baseline_amp_rsquared = sp.stats.spearmanr(np.mean(sj_data_c_array, axis=0)[:-2], av_all_rsquared)
		ttest_correlation_pattern = sp.stats.ttest_1samp(fisher_z, 0)
		
		f = pl.figure() 
		s = f.add_subplot(111)
		sn.regplot(np.mean(sj_data_low_freq_array, axis=0),av_all_rsquared)
		pl.text(0.03,0.15, 'r=%.3f \np=%.3f'%(corr_baseline_amp_rsquared[0], corr_baseline_amp_rsquared[1]))
		pl.xlabel('average baseline amplitude correlation')
		pl.ylabel('R2')
		
		#shared y-axis plot R2 and baseline-amplitude correlation
		bl_amp_err = sj_data_low_freq_array.std(axis=0)/np.sqrt(len(sj_data_low_freq_array))
		R2_err = all_rsquared.std(axis=1)/np.sqrt(all_rsquared.shape[1])
		
		fig, ax1 = pl.subplots()
		ax1.set_xscale('log')
		#ax1.errorbar(low_freqs, av_all_rsquared, yerr=R2_err, color='b', fmt='-o', ecolor='k', lw=0.5, capthick=0.5)
		ax1.plot(low_freqs, av_all_rsquared, '#6e1005')
		ax1.fill_between(low_freqs, y1=av_all_rsquared-R2_err, 
			y2 =av_all_rsquared+R2_err, 
			alpha=0.5, edgecolor='#840000', 
			facecolor='#9d0216')
		ax1.set_xlabel('frequency', fontsize=10)
		ax1.set_ylabel('average R2', color='#9d0216', fontsize=10)
		for tl in ax1.get_yticklabels():
			tl.set_color('#9d0216')
			tl.set_size('small')
		ax2 = ax1.twinx()
		ax2.plot(low_freqs, np.mean(sj_data_low_freq_array,axis=0), '#647d8e')
		ax2.fill_between(low_freqs, y1=np.mean(sj_data_low_freq_array,axis=0)-bl_amp_err, 
			y2 =np.mean(sj_data_low_freq_array, axis=0)+bl_amp_err, 
			alpha=0.5, edgecolor='#516572', 
			facecolor='#03719c')
		# ax2.errorbar(low_freqs, np.mean(sj_data_low_freq_array, axis=0), yerr=bl_amp_err, color='r', fmt='-o', ecolor='k', lw=0.5, capthick=0.5)
		ax2.set_ylabel('average amplitude baseline correlation', color='#03719c', fontsize=10)
		for tl in ax2.get_yticklabels():
				tl.set_color('#03719c')
				tl.set_size('small')
		pl.title('relation between R2 and baseline amplitude correlation', fontsize=11)
		pl.axhline(0, color = 'k', linewidth=0.25) 
		pl.tight_layout()
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'group_average_ba_r2_correlations.pdf')) 

		#histogram of R2 values and baseline/amplitude values
		f = pl.figure() 
		s1=f.add_subplot(121)
		pl.hist(all_rsquared.T)
		pl.xlabel('R2 values')
		pl.ylabel('count')
		s1.set_title('R2 of pupil response around keypress')
		s2= f.add_subplot(122)
		pl.hist(sj_data_low_freq_array)
		pl.xlabel('Correlation values')
		pl.ylabel('count')
		s2.set_title('Baseline amplitude correlations')
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'pooled_histogram_R2_ba_correlations.pdf')) 


		fig, axes = pl.subplots(nrows=4, ncols=5, sharey=True, sharex=True)
		for i, ax in enumerate(axes.flat):
			ax.hist(sj_data_low_freq_array[:,i])
			ax.set_title(str(low_freqs[i]))
			ax.vlines(0,ymin=0, ymax=9, colors=u'r', linestyles=u'dashed', lw=0.7)
			ax.set_xlabel('correlation value')
			ax.set_ylabel('count')
		pl.tight_layout()
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'histogram_per_freq_ba_correlations.pdf')) 

		# #save correlations for each iteration of variable 
		# spearman = []
		# for name, variable in zip(['max_slope', 'av_slope', 'max_acceleration', 'av_acceleration'],[max_slope, av_slope, max_accel, av_accel]): #av_slope, max_accel, av_accel
		# 	f = pl.figure()
		# 	pl.title(name, fontsize=10)
		# 	spearman.append([])
		# 	for i,freq in enumerate(f_array): 
		# 		s = f.add_subplot(4,6,i+1)
		# 		frequency_corr = sj_data_c_array[:,i]
		# 		sn.regplot(x=frequency_corr, y=variable)
		# 		spearman_cor = sp.stats.spearmanr(frequency_corr, variable)
		# 		spearman[-1].append(spearman_cor)
		# 		s.text(0.05, 0.0025, 'r: %.2f \n p: %.2f '%(spearman_cor[0], spearman_cor[1]), fontsize=7)
		# 		s.set_title(str(freq))
		# 		pl.tight_layout()

		# #calculate correlation between filter_bank rsquared and average acceleration 
		# av_acceleration_cor = np.array(spearman[-1])[:-2,0]
		# av_slope_cor = np.array(spearman[1])[:-2,0]
		# #rsquared and average acceleration / baseline amplitude corr are inversely correlated 
		# corr_r2_av_acceleration = sp.stats.spearmanr(av_all_rsquared, av_acceleration_cor)
		
		# f = pl.figure() 
		# pl.plot(f_array[:-2], av_acceleration_cor, 'b') 
		# pl.plot(f_array[:-2], av_all_rsquared, 'g') 
		# pl.plot(f_array[:-2], av_slope_cor, 'y')
		# pl.xlabel('Frequency band')
		# pl.ylabel('R')
		# pl.legend(['cor of average acceleration with baseline/amplitude cor', 'average R2 of pupil response around keypress', 'cor of average slope with baseline/amplitude cor'])
		# pl.savefig(os.path.join(self.grouplvl_plot_dir, 'av_acceleration_baseline_amplitude_correlation.pdf')) 
		

	def baseline_phasic_pupil_correlations_reversal_performance_median_split(self, split_on='high_performers'): 
		"""Looks at baseline/amplitude correlations in relation to R2 of deconvolved pupil response around keypress  
		for two groups: participants that performed high during reversal detection and participants that performed less 
		high """	
		self.all_sessions = np.copy(self.sessions)
		names = np.array([s.subject.initials for s in self.all_sessions])
		
		folder_name = 'slope_acceleration_pupil_around_keypress'
		av_accel = np.array(np.squeeze(self.gather_dataframes_from_hdfs(group = folder_name, data_type = 'av_accel')))
		#median split on performance 	
		percentage_correct_reversals = np.array(np.squeeze(self.gather_dataframes_from_hdfs(group = 'distance_reversal_keypress_behaviour', data_type = 'percentage_positive')))
		percentage_correct_reversals_2 = np.array(np.squeeze(self.gather_dataframes_from_hdfs(group = 'keypresses', data_type = 'press_correct')))

		high_performers = names[percentage_correct_reversals_2>=np.median(percentage_correct_reversals_2)]
		steep_peak_ppns = names[av_accel>np.median(av_accel)]

		#median split
		if split_on == 'high_performers':
			self.sessions = [s for s in self.sessions if s.subject.initials in high_performers] 
			var_to_use = high_performers
		elif split_on == 'steep_peak':
			self.sessions = [s for s in self.sessions if s.subject.initials in steep_peak_ppns] 
			var_to_use = steep_peak_ppns
		else: 
			print ('incorrect split_on response')

		sj_data_lst = self.gather_dataframes_from_hdfs(group = 'pupil_all_baselines_phasic', data_type = 'pupil_baseline_correlations_simple')
		sj_data_low_freq_array = np.array([np.array(p['correlations'], dtype = float) for p in sj_data_lst])[:,:-2]
		f_array = np.array([np.array(p['frequencies'], dtype = float) for p in sj_data_lst])[0][:-2]	

		keys = self.gather_data_from_hdfs(group = 'deconvolve_pupil_around_keypress_20_tonic_baseline_zscore_True', data_type = 'keys')[0]
		keys = [k.rpartition('.')[-1] for k in keys]
		freqs = np.array([float('0.' + key) for key in keys])
		freq_order = np.argsort(freqs)	

		all_rsquared=[]	
		#loop over rsquared results of filter_bank deconvolutions	
		for idx in freq_order: 
			rsquared = self.gather_data_from_hdfs(group = 'deconvolve_pupil_around_keypress_20_tonic_baseline_zscore_True', data_type = 'rsquared_%s_hz'%keys[idx])
			all_rsquared.append(np.squeeze(rsquared))
		all_rsquared = np.array(all_rsquared)
		av_all_rsquared=np.mean(np.array(all_rsquared), axis=1)

		corr_rsquared_amp_bl=[]
		fisher_z = []
		for i, name in enumerate(var_to_use):  
			corr_rsquared_amp_bl.append(sp.stats.spearmanr(all_rsquared[:,i], sj_data_low_freq_array[i,:]))
			fisher_z.append(np.arctanh(corr_rsquared_amp_bl[i][0]))

		corr_rsquared_amp_bl = np.array(corr_rsquared_amp_bl)
		corr_baseline_amp_rsquared = sp.stats.spearmanr(np.mean(sj_data_low_freq_array, axis=0), av_all_rsquared)
		ttest_correlation_pattern = sp.stats.ttest_1samp(fisher_z, 0)
		#select ppns based on reversal learning performance: r = -0.78. p<0.01, ttest non significant due to power issue
		#select ppns based on steepness of pupil response aroudn keypress: r = -0.87, p<0.01, fisher ttest, z=-2.26, p=0.03
				#shared y-axis plot R2 and baseline-amplitude correlation
		
		bl_amp_err = sj_data_low_freq_array.std(axis=0)/np.sqrt(len(sj_data_low_freq_array))
		R2_err = all_rsquared.std(axis=1)/np.sqrt(all_rsquared.shape[1])

		fig, ax1 = pl.subplots()
		ax1.set_xscale('log')
		ax1.plot(f_array, av_all_rsquared, '#6e1005')
		ax1.fill_between(f_array, y1=av_all_rsquared-R2_err, 
			y2 =av_all_rsquared+R2_err, 
			alpha=0.5, edgecolor='#840000', 
			facecolor='#9d0216')
		ax1.set_xlabel('frequency', fontsize=10)
		ax1.set_ylabel('average R2', color='#9d0216', fontsize=10)
		for tl in ax1.get_yticklabels():
			tl.set_color('#9d0216')
			tl.set_size('small')
		ax2 = ax1.twinx()
		ax2.plot(f_array, np.mean(sj_data_low_freq_array,axis=0), '#647d8e')
		ax2.fill_between(f_array, y1=np.mean(sj_data_low_freq_array,axis=0)-bl_amp_err, 
			y2 =np.mean(sj_data_low_freq_array, axis=0)+bl_amp_err, 
			alpha=0.5, edgecolor='#516572', 
			facecolor='#03719c')
		# ax2.errorbar(low_freqs, np.mean(sj_data_low_freq_array, axis=0), yerr=bl_amp_err, color='r', fmt='-o', ecolor='k', lw=0.5, capthick=0.5)
		ax2.set_ylabel('average amplitude baseline correlation', color='#03719c', fontsize=10)
		for tl in ax2.get_yticklabels():
				tl.set_color('#03719c')
				tl.set_size('small')
		pl.text(0.02, 0.03, 'r = %.3f \n p = %.3f'%(corr_baseline_amp_rsquared[0], corr_baseline_amp_rsquared[1]), fontsize=10)
		pl.title('relation between R2 and baseline amplitude correlation (%s)'%split_on, fontsize=11)
		pl.axhline(0, color = 'k', linewidth=0.25) 
		pl.tight_layout()
		pl.savefig(os.path.join(self.grouplvl_plot_dir, '%s_ppns_ba_r2_correlations.pdf'%split_on)) 


	def correlate_blinkrate_with_prediction_errors(self, use_domain='full', baseline=True, analysis_sample_rate=10): 
		"""Deconvolved positive and negative prediction errors are correlated with blink rate across subjects"""
		
		#folder_name = 'deconvolve_full_FIR_%s_domain'%use_domain
		folder_name = 'deconvolve_full_FIR_%s_domain_long_interval'%use_domain
		sj_deconvolved_timecourses_lst = self.gather_dataframes_from_hdfs(group = folder_name, data_type = 'deconvolved_pupil_timecourses')
		alpha_values = self.gather_data_from_hdfs(group = folder_name, data_type = 'alpha')

		sj_data_lpnr_array = np.array([np.array(p['LPNR.gain'], dtype = float) for p in sj_deconvolved_timecourses_lst])
		sj_data_lphr_array = np.array([np.array(p['LPHR.gain'], dtype = float) for p in sj_deconvolved_timecourses_lst])
		sj_data_hpnr_array = np.array([np.array(p['HPNR.gain'], dtype = float) for p in sj_deconvolved_timecourses_lst])
		sj_data_hphr_array = np.array([np.array(p['HPHR.gain'], dtype = float) for p in sj_deconvolved_timecourses_lst])
			
		time = self.gather_data_from_hdfs(group = folder_name, data_type = 'time_points')
		baseline_times = time.mean(axis = 0) < 0 


		reward_matrix = np.array(np.r_[[sj_data_lpnr_array, sj_data_lphr_array, sj_data_hpnr_array, sj_data_hphr_array]]).transpose(1,2,0)
		prediction_error_matrix = np.r_[[reward_matrix[:,:,1] - reward_matrix[:,:,-1], reward_matrix[:,:,2] - reward_matrix[:,:,0]]].transpose(1,2,0)
		
		# folder_name = 'deconvolve_full_FIR_third_3split_domain'
		# time = self.gather_data_from_hdfs(group = folder_name, data_type = 'time_points')
		# baseline_times = time.mean(axis = 0) < 0 	
		# t_sj_deconvolved_timecourses_lst = self.gather_dataframes_from_hdfs(group = folder_name, data_type = 'deconvolved_pupil_timecourses')
		
		# sj_data_lpnr_array = np.array([np.array(p['LPNR.gain'], dtype = float) for p in t_sj_deconvolved_timecourses_lst])
		# sj_data_lphr_array = np.array([np.array(p['LPHR.gain'], dtype = float) for p in t_sj_deconvolved_timecourses_lst])
		# sj_data_hpnr_array = np.array([np.array(p['HPNR.gain'], dtype = float) for p in t_sj_deconvolved_timecourses_lst])
		# sj_data_hphr_array = np.array([np.array(p['HPHR.gain'], dtype = float) for p in t_sj_deconvolved_timecourses_lst])

		# t_reward_matrix = np.array(np.r_[[sj_data_lpnr_array, sj_data_lphr_array, sj_data_hpnr_array, sj_data_hphr_array]]).transpose(1,2,0)
		# #prediction_error_matrix = np.r_[[reward_matrix[:,:,1] - reward_matrix[:,:,-1], reward_matrix[:,:,2] - reward_matrix[:,:,0]]].transpose(1,2,0)

		# diff_reward_matrix = t_reward_matrix - f_reward_matrix
		# diff_prediction_error_matrix = np.r_[[diff_reward_matrix[:,:,1] - diff_reward_matrix[:,:,-1], diff_reward_matrix[:,:,2] - diff_reward_matrix[:,:,0]]].transpose(1,2,0)


		if baseline: 
			prediction_error_matrix_s = np.array([[myfuncs.smooth(prediction_error_matrix[i,:,j], window_len=10) - prediction_error_matrix[i,baseline_times,j].mean() for i in range(prediction_error_matrix.shape[0])] for j in range(prediction_error_matrix.shape[-1])]).transpose(1,2,0)
		else: 
			prediction_error_matrix_s = np.array([[myfuncs.smooth(prediction_error_matrix[i,:,j], window_len=10) for i in range(prediction_error_matrix.shape[0])] for j in range(prediction_error_matrix.shape[-1])]).transpose(1,2,0)	
		

		blink_pe_correlation_matrix = self.correlate_blinkrate_with_prediction_errors(input_object=prediction_error_matrix_s)

		blink_prpe_correlation_t = blink_pe_correlation_matrix[:,0]
		blink_nrpe_correlation_t = blink_pe_correlation_matrix[:,1]

		significant_prpe_cor = [np.where(blink_prpe_correlation_t[:,-1]<0.05)[0]]
		significant_nrpe_cor = [np.where(blink_nrpe_correlation_t[:,-1]<0.05)[0]]

		pe_cond = pd.Series(['PRPE', 'NRPE'])
		cond = pd.Series(['LP_NR', 'LP_HR', 'HP_NR', 'HP_HR'])
		
		f = pl.figure()
		f.text(0.45, 0.97, 'deconvolution of %s domain'%use_domain) 
		#f.text(0.45, 0.97, 'deconvolution of difference first and third domain') 
		f.add_subplot(221)
		sn.tsplot(reward_matrix, time=time.mean(axis=0), condition = cond)
		pl.axhline(0, color = 'k', linewidth=0.25) 
		f.add_subplot(222)
		sn.tsplot(prediction_error_matrix_s, time=time.mean(axis=0), condition=pe_cond)
		pl.axhline(0, color = 'k', linewidth=0.25) 
		f.add_subplot(223)
		pl.plot(time.mean(axis=0), blink_prpe_correlation_t[:,:-1])
		#pl.plot(time.mean(axis=0)[significant_prpe_cor], np.zeros((time.mean(axis=0).shape[0]))[significant_prpe_cor], ls='--')
		pl.axhline(0, color = 'k', linewidth=0.25) 
		pl.legend(['fisher_z', 'r'])
		pl.title('prpe/blink correlation over time')
		f.add_subplot(224)
		pl.plot(time.mean(axis=0), blink_nrpe_correlation_t[:,:-1])
		#pl.plot(time.mean(axis=0)[significant_nrpe_cor], np.zeros((time.mean(axis=0).shape[0]))[significant_nrpe_cor], ls='--')
		pl.axhline(0, color = 'k', linewidth=0.25) 
		pl.legend(['fisher_z', 'r'])
		pl.title('nrpe/blink correlation over time')
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'full_FIR_deconvolution_%s_domain_long_interval_cue_duration_baseline_%s.pdf'%(use_domain, str(baseline))))

	
	def correlate_blinkrate_with_prediction_errors(self, input_object): 
		"""correlate blinkrate with prediciton errors across subjects takes input_object of shape subjects, timepoints, conditions and 
		correlates this with the mean blink rate per participant. It returns a correlation matrix of timepoints x conditions x values (fisher z, rho, pvalue) """

		#blink rate info 
		blink_rate = self.gather_data_from_hdfs(group = 'blink_rate', data_type = 'blink_rate')
		mean_blink_rate = np.array([np.mean(b, axis=0) for b in blink_rate])
		se_blink_rate = np.array([np.std(b, axis=0)/np.sqrt(len(b)) for b in blink_rate])

		#correlate prediction errors with blink rate 
		blink_pe_correlation_matrix = np.zeros((input_object.shape[1],input_object.shape[-1],3)) #timepoints x conditions x  fisher z, rho, pvalue
		
		#loop over timepoints
		for j in range(input_object.shape[1]): 
			#timepoints per condition 
			time_points = np.array([input_object[:,j,cond] for cond in range(input_object.shape[-1])])
			#correlate each condition's timepoints with blink rate 
			for i in range(time_points.shape[0]):
				rho, pvalue = sp.stats.spearmanr(time_points[i], mean_blink_rate)
				fisher_z = np.arctanh(rho)
				#save correlations in matrix
				blink_pe_correlation_matrix[j,i] = fisher_z, rho, pvalue
		return blink_pe_correlation_matrix


	def event_related_average_across_subjects(self, use_domain='full', analysis_sample_rate=10): 
		"""event related average results across subjects and correlation with blinks  """

		#blink rate info 
		blink_rate = self.gather_data_from_hdfs(group = 'blink_rate', data_type = 'blink_rate')
		mean_blink_rate = np.array([np.mean(b, axis=0) for b in blink_rate])
		se_blink_rate = np.array([np.std(b, axis=0)/np.sqrt(len(b)) for b in blink_rate])

		#ERA info 
		folder_name = 'epoched_data_%s_domain_%s_Hz'%(use_domain, analysis_sample_rate)
		labels = self.gather_data_from_hdfs(group = folder_name, data_type = 'labels')[0]
		time_points = self.gather_data_from_hdfs(group = folder_name, data_type = 'time_points')[0]
		event_related_average_sjs = self.gather_data_from_hdfs(group = folder_name, data_type = 'mean_pupil_event_responses')

		#standard events 
		colour, sound = 	event_related_average_sjs[:,0,:], event_related_average_sjs[:,1,:]
		cues_absolute = 	np.r_[[event_related_average_sjs[:,2,:], event_related_average_sjs[:,3,:]]].transpose(1,2,0) #low, high
		cues_relative =  	np.r_[[event_related_average_sjs[:,2,:] - colour, event_related_average_sjs[:,3,:] - colour]].transpose(1,2,0) #low, high
		rewards_absolute = 	np.r_[[event_related_average_sjs[:,4,:], event_related_average_sjs[:,5,:]]].transpose(1,2,0)  #loss, win 
		rewards_relative = 	np.r_[[event_related_average_sjs[:,4,:] - sound , event_related_average_sjs[:,5,:] - sound]].transpose(1,2,0)  #loss, win 
		
		#reward events: LPNR, LPHR, HPNR, HPHR
		reward_events_absolute = np.r_[[event_related_average_sjs[:,6,:] , event_related_average_sjs[:,7,:],  event_related_average_sjs[:,8,:], event_related_average_sjs[:,9,:]]].transpose(1,2,0)
		reward_events_relative = np.r_[[event_related_average_sjs[:,6,:] - sound , event_related_average_sjs[:,7,:] - sound,  event_related_average_sjs[:,8,:] - sound, event_related_average_sjs[:,9,:]-sound]].transpose(1,2,0)
		
		#NRPE, PRPE
		prediction_error_absolute = np.r_[[reward_events_absolute[:,:,2] - reward_events_absolute[:,:,0], reward_events_absolute[:,:,1] - reward_events_absolute[:,:,-1]]].transpose(1,2,0)
		prediction_error_relative = np.r_[[reward_events_relative[:,:,2] - reward_events_relative[:,:,0], reward_events_relative[:,:,1] - reward_events_relative[:,:,-1]]].transpose(1,2,0)

		blink_pe_correlation_matrix = self.correlate_blinkrate_with_prediction_errors(input_object=prediction_error_absolute[:,30:,:])
		mask_neg = np.ma.masked_where(blink_pe_correlation_matrix[:,0,-1] > 0.05, blink_pe_correlation_matrix[:,0,-1])
		mask_pos = np.ma.masked_where(blink_pe_correlation_matrix[:,1,-1] > 0.05, blink_pe_correlation_matrix[:,1,-1])

		#select color palettes
		reward_colors = ['#fc5a50', '#15b01a', '#ff000d','#789b73']
		reward_palette = sn.color_palette(reward_colors)
		low_high_colors = ['#fc5a50','#789b73']
		low_high_pallette = sn.color_palette(low_high_colors)
		pe = ['#ff000d','#15b01a']
		pe_pallet = sn.color_palette(pe)

		cues= pd.Series(['cue low', 'cue high'])
		rewards = pd.Series(['loss','reward'])
		reward_events = pd.Series(['LPNR','LPHR', 'HPNR' ,'HPHR'])
		p_e = pd.Series(['NRPE', 'PRPE'])
		
		#absolute responses 
		f = pl.figure(figsize=(8,5))
		f.text(0.4, 0.97, 'Absolute event related average %s domain'%use_domain)
		s = f.add_subplot(231)
		sn.tsplot(cues_absolute, time=time_points, condition = cues)
		pl.axhline(0, color='k', lw=0.25)
		pl.axvline(0, color='k', lw=0.25)
		s.set_title('absolute cue response')	
		s = f.add_subplot(232)
		sn.tsplot(rewards_absolute, time=time_points, condition = rewards, color=low_high_colors)
		pl.axhline(0, color='k', lw=0.25)
		pl.axvline(0, color='k', lw=0.25)
		s.set_title('absolute reward response')	
		s = f.add_subplot(233)
		sn.tsplot(reward_events_absolute, time=time_points, condition = reward_events, color = reward_palette )
		pl.axhline(0, color='k', lw=0.25)
		pl.axvline(0, color='k', lw=0.25)
		s.set_ylim([-0.5, 0.5])
		s.set_title('absolute reward event responses')	
		s = f.add_subplot(234)
		sn.tsplot(prediction_error_absolute[:,30:,:], time=time_points[30:], condition = p_e, color=pe_pallet)
		pl.axhline(0, color='k', lw=0.25)
		pl.axvline(0, color='k', lw=0.25)
		s.set_title('prediction error responses')
		s.set_ylim([-0.5, 0.5])
		s = f.add_subplot(235)
		s.set_title('correlation blink rate & prediction errors ')
		pl.plot(time_points[30:], blink_pe_correlation_matrix[:,0,0], color='#ff000d')
		pl.plot(time_points[30:], blink_pe_correlation_matrix[:,1,0], color='#15b01a')
		pl.plot(time_points[30:], mask_neg*0, 'r--', lw=3)
		pl.legend(['cor NRPE', 'cor PRPE'])
		pl.axhline(0, color='k', lw=0.25)
		pl.tight_layout()
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'absolute_era_responses_%s_domain.pdf'%use_domain))
	
		#relative responses
		f = pl.figure(figsize=(8,5))
		f.text(0.4, 0.97, 'Relative event related average %s domain'%use_domain)
		s = f.add_subplot(231)
		sn.tsplot(cues_relative, time=time_points, condition = cues)
		pl.axhline(0, color='k', lw=0.25)
		pl.axvline(0, color='k', lw=0.25)
		s.set_title('relative cue response')
		s = f.add_subplot(232)
		sn.tsplot(rewards_relative, time=time_points, condition = rewards, color=low_high_colors)
		pl.axhline(0, color='k', lw=0.25)
		pl.axvline(0, color='k', lw=0.25)
		s.set_title('relative reward response')
		s = f.add_subplot(233)
		sn.tsplot(reward_events_relative, time=time_points, condition = reward_events, color=reward_palette)
		pl.axhline(0, color='k', lw=0.25)
		pl.axvline(0, color='k', lw=0.25)
		s.set_ylim([-0.5, 0.5])
		s.set_title('relative reward event responses')
		s = f.add_subplot(234)
		sn.tsplot(prediction_error_relative[:,30:,:], time=time_points[30:], condition = p_e, color=pe_pallet)
		pl.axhline(0, color='k', lw=0.25)
		pl.axvline(0, color='k', lw=0.25)
		s.set_title('prediction error responses')
		s.set_ylim([-0.5, 0.5])
		s = f.add_subplot(235)
		s.set_title('correlation blink rate & prediction error ')
		pl.plot(time_points[30:], blink_pe_correlation_matrix[:,0,0], color='#ff000d')
		pl.plot(time_points[30:], blink_pe_correlation_matrix[:,1,0], color='#15b01a')
		pl.plot(time_points[30:], mask_neg*0, 'r--', lw=3)
		pl.legend(['cor NRPE', 'cor PRPE'])
		pl.axhline(0, color='k', lw=0.25)
		pl.tight_layout()
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'relative_era_responses_%s_domain.pdf'%use_domain))

	def single_trial_GLM_one_kernel_group_results(self, which_kernel='event'): 
		"""accumulates beta values of single_trial_GLM_one_gamma_kernel and averages over subjects and conditions """   
		#load pickle data       
		event_kernel_models = self.gather_data_from_pickles(data_types = ['aic', 'bic', 'rsquared'], which_kernel='event')
		reward_kernel_models = self.gather_data_from_pickles(data_types = ['aic', 'bic', 'rsquared'], which_kernel = 'one')
		event_reward_models = self.gather_data_from_pickles(data_types = ['aic', 'bic', 'rsquared'], which_kernel = 'dual')
		#kdeplot 
		fig = pl.figure(figsize=(10,4))
		for i, names in enumerate(['AIC', 'BIC', 'Rsquared']): 
			ax = fig.add_subplot(1,3,i+1)
			ax.set_title(names)
			sn.kdeplot(event_kernel_models[:,i], label = 'event kernel', shade=True)
			sn.kdeplot(reward_kernel_models[:,i], label='reward kernel', shade=True) 
			sn.kdeplot(event_reward_models[:,i], label = 'reward & event', shade=True)
			pl.legend() 
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'model distributions.pdf'))

		fig = pl.figure(figsize=(10,4))
		for i, names in enumerate(['AIC', 'BIC', 'Rsquared']):  
			ax = fig.add_subplot(1,3,i+1)
			ax.set_title(names)
			pl.bar([0.1], [event_kernel_models[:,i].mean(axis=0)], yerr= event_kernel_models[:,i].std(axis=0)/np.sqrt(len(event_kernel_models)), ecolor='k', width=0.1)
			pl.bar([0.25], [reward_kernel_models[:,i].mean(axis=0)], yerr= reward_kernel_models[:,i].std(axis=0)/np.sqrt(len(event_kernel_models)) ,ecolor='k', width=0.1)
			pl.bar([0.4], [event_reward_models[:,i].mean(axis=0)], yerr= event_reward_models[:,i].std(axis=0)/np.sqrt(len(event_kernel_models)),ecolor='k', width=0.1)
			pl.ylabel(names, fontsize=8)
			ax.set_xticks([0.15, 0.3, 0.45])
			ax.tick_params(axis='both', which='major', labelsize=8)
			ax.set_xticklabels(['event', 'reward', 'event \n & reward'], rotation=45, fontsize=10)      
			simpleaxis(ax)
			fig.tight_layout()

			#one-way ANOVA
			f_value, p_value = stats.f_oneway(event_kernel_models[:,i], reward_kernel_models[:,i], event_reward_models[:,i])
			print(names + ' f_value: %.3f' %f_value)
			print(names + ' p_value: %.3f' %p_value)

			#Tukey's HSD 
			x1 = pd.DataFrame(event_kernel_models[:,i], columns=['observation'])
			x1['grouplabel'] = 'event_kernel'
			x2 = pd.DataFrame(reward_kernel_models[:,i], columns=['observation'])
			x2['grouplabel'] = 'reward_kernel'
			x3 = pd.DataFrame(event_reward_models[:,i], columns=['observation'])
			x3['grouplabel'] = 'event_reward_kernel'

			data = x1.append(x2).append(x3)
			data.head() 

			result = sm.multicomp.pairwise_tukeyhsd(data.observation, data.grouplabel)
			print(result.summary())
			print(result.groupsunique)

		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'model_comparisons.pdf'))
		
	
		#load HDF5 data
		for name in (['diff_betas_%s_kernel'%which_kernel, 'zscored_diff_betas_%s_kernel'%which_kernel,'colour_%s_kernel'%which_kernel,'sound_%s_kernel'%which_kernel, 'zscored_beta_sound_%s'%which_kernel, 'zscored_beta_colour_%s'%which_kernel]):
			prpe_median_start_end_data = self.gather_data_from_hdfs(group = 'betas_%s_gamma_kernel'%which_kernel, data_type='prpe_median_start_end_data_'+name) # participant x 2 [begin, end]  
			nrpe_median_start_end_data = self.gather_data_from_hdfs(group = 'betas_%s_gamma_kernel'%which_kernel, data_type='nrpe_median_start_end_data_'+name)
			hphr_median_start_end_data = self.gather_data_from_hdfs(group = 'betas_%s_gamma_kernel'%which_kernel, data_type='hphr_median_start_end_data_'+name)
			lplr_median_start_end_data = self.gather_data_from_hdfs(group = 'betas_%s_gamma_kernel'%which_kernel, data_type='lplr_median_start_end_data_'+name)

			#calculate overall median beta values [begin, end] for different trial types
			mean_prpe_median_start_end_data = np.array([np.array(map(np.median, prpe_median_start_end_data[:,i])).mean(axis=0)for i in range(prpe_median_start_end_data.shape[1])])
			mean_nrpe_median_start_end_data = np.array([np.array(map(np.median, nrpe_median_start_end_data[:,i])).mean(axis=0)for i in range(nrpe_median_start_end_data.shape[1])])
			mean_hphr_median_start_end_data = np.array([np.array(map(np.median, hphr_median_start_end_data[:,i])).mean(axis=0)for i in range(hphr_median_start_end_data.shape[1])])
			mean_lplr_median_start_end_data = np.array([np.array(map(np.median, lplr_median_start_end_data[:,i])).mean(axis=0)for i in range(lplr_median_start_end_data.shape[1])])
			#calculate overal SEM for different trial types 
			se_prpe_median_start_end_data = np.array([np.array(map(np.median, prpe_median_start_end_data[:,i])).std(axis=0) for i in range(prpe_median_start_end_data.shape[1])])/np.sqrt(len(prpe_median_start_end_data))
			se_nrpe_median_start_end_data = np.array([np.array(map(np.median, nrpe_median_start_end_data[:,i])).std(axis=0) for i in range(nrpe_median_start_end_data.shape[1])])/np.sqrt(len(nrpe_median_start_end_data))
			se_hphr_median_start_end_data = np.array([np.array(map(np.median, hphr_median_start_end_data[:,i])).std(axis=0) for i in range(hphr_median_start_end_data.shape[1])])/np.sqrt(len(hphr_median_start_end_data))
			se_lplr_median_start_end_data = np.array([np.array(map(np.median, lplr_median_start_end_data[:,i])).std(axis=0) for i in range(lplr_median_start_end_data.shape[1])])/np.sqrt(len(lplr_median_start_end_data))

			#One-way ANOVA 
			# f_value, p_value = stats.f_oneway(mean_prpe_median_start_end_data[:,i], mean_nrpe_median_start_end_data[:,i])
			# print(name + '  f_value: %.3f' %f_value)
			# print(name + '  p_value: %.3f' %p_value)

			# #Tukey's HSD 
			# x1 = pd.DataFrame(prpe_median_start_end_data[i,i], columns=['observation'])
			# x1['grouplabel'] = 'prpe'
			# x2 = pd.DataFrame(nrpe_median_start_end_data[i,i], columns=['observation'])
			# x2['grouplabel'] = 'nrpe'
			# x3 = pd.DataFrame(hphr_median_start_end_data[i,i], columns=['observation'])
			# x3['grouplabel'] = 'hphr'
			# x4 = pd.DataFrame(lplr_median_start_end_data[i,i], columns=['observation'])
			# x4['grouplabel'] = 'lplr'


			# data = x1.append(x2).append(x3).append(x4)
			# data.head() 

			# result = sm.multicomp.pairwise_tukeyhsd(data.observation, data.grouplabel)
			# print(result.summary())
			# print(result.groupsunique)




			#plot group barplot 
			fig= pl.figure(figsize=(6,4))
			ax1= fig.add_subplot(121)
			ax1.set_title(name + '\n group results (N=19) \n prediction error trials', fontsize=8)
			pl.bar([0,1], mean_prpe_median_start_end_data, yerr =se_prpe_median_start_end_data, color = 'r', width=0.2, ecolor='k') 
			pl.bar([0.3,1.3], mean_nrpe_median_start_end_data, yerr =se_nrpe_median_start_end_data, color = 'b', width=0.2, ecolor='k')
			pl.ylabel('beta values')
			pl.xlabel('time')   
			pl.legend(['prpe', 'nrpe'], loc = 'best')
			simpleaxis(ax1)
			spine_shift(ax1)
			pl.axhline(0, color = 'k', linewidth=0.25)  
			ax1.set_xticks([0.25, 1.25])
			ax1.set_xticklabels(['early', 'late'])
			fig.text(0.02, 0.6, 'beta value', ha='center', va='center', rotation='vertical', fontsize=12)

			ax2= fig.add_subplot(122, sharey=ax1)
			ax2.set_title(name + '\n no prediction error trials', fontsize=8)
			pl.ylabel('beta values')
			pl.xlabel('time')   
			pl.bar([0,1], mean_hphr_median_start_end_data, yerr =se_hphr_median_start_end_data, color = 'r', width=0.2, ecolor='k', alpha=0.5)  
			pl.bar([0.3,1.3], mean_lplr_median_start_end_data, yerr =se_lplr_median_start_end_data, color = 'b', width=0.2, ecolor='k', alpha=0.5)
			pl.legend(['hphr', 'lplr'], loc = 'best')
			simpleaxis(ax2)
			spine_shift(ax2)
			pl.axhline(0, color = 'k', linewidth=0.25)  
			ax2.set_xticks([0.25, 1.25])
			ax2.set_xticklabels(['early', 'late'])
			pl.tight_layout()
			pl.savefig(os.path.join(self.grouplvl_plot_dir, 'bar_plot_beta_val_%s_%s_gamma_kernel.pdf'%(name, which_kernel)))



	def single_trial_GLM_dual_kernel_group_results(self):
		"""single_trial_GLM_dual_kernel_group_results accumulates beta values of single trial GLM and averages over subjects and conditions  """

		# # ### EXPECTANCY BETAS ###
		# for name in (['colour_event_kernel','colour_reward_kernel']):
		#   prpe_median_start_end_data = self.gather_data_from_hdfs(group = 'betas', data_type='prpe_median_start_end_data_'+name) # participant x 2 [begin, end]   
		#   nrpe_median_start_end_data = self.gather_data_from_hdfs(group = 'betas', data_type='nrpe_median_start_end_data_'+name)
		#   hphr_median_start_end_data = self.gather_data_from_hdfs(group = 'betas', data_type='hphr_median_start_end_data_'+name)
		#   lplr_median_start_end_data = self.gather_data_from_hdfs(group = 'betas', data_type='lplr_median_start_end_data_'+name)
		#   #calculate overall median beta values [begin, end] for different trial types
		#   mean_prpe_median_start_end_data = np.array([np.array(map(np.median, prpe_median_start_end_data[:,i])).mean(axis=0)for i in range(prpe_median_start_end_data.shape[1])])
		#   mean_nrpe_median_start_end_data = np.array([np.array(map(np.median, nrpe_median_start_end_data[:,i])).mean(axis=0)for i in range(nrpe_median_start_end_data.shape[1])])
		#   mean_hphr_median_start_end_data = np.array([np.array(map(np.median, hphr_median_start_end_data[:,i])).mean(axis=0)for i in range(hphr_median_start_end_data.shape[1])])
		#   mean_lplr_median_start_end_data = np.array([np.array(map(np.median, lplr_median_start_end_data[:,i])).mean(axis=0)for i in range(lplr_median_start_end_data.shape[1])])
		#   #calculate overal SEM for different trial types 
		#   se_prpe_median_start_end_data = np.array([np.array(map(np.median, prpe_median_start_end_data[:,i])).std(axis=0) for i in range(prpe_median_start_end_data.shape[1])])/np.sqrt(len(prpe_median_start_end_data))
		#   se_nrpe_median_start_end_data = np.array([np.array(map(np.median, nrpe_median_start_end_data[:,i])).std(axis=0) for i in range(nrpe_median_start_end_data.shape[1])])/np.sqrt(len(nrpe_median_start_end_data))
		#   se_hphr_median_start_end_data = np.array([np.array(map(np.median, hphr_median_start_end_data[:,i])).std(axis=0) for i in range(hphr_median_start_end_data.shape[1])])/np.sqrt(len(hphr_median_start_end_data))
		#   se_lplr_median_start_end_data = np.array([np.array(map(np.median, lplr_median_start_end_data[:,i])).std(axis=0) for i in range(lplr_median_start_end_data.shape[1])])/np.sqrt(len(lplr_median_start_end_data))

		#   #plot group barplot for expectancy (only high expectation and low expectation)
		#   fig = pl.figure(figsize=(6,4))
		#   ax1= fig.add_subplot(121)
		#   ax1.set_title(name + '\n group results (N=19) \n reward expectancy (high vs low)', fontsize=8)
		#   pl.bar([0,1], np.mean((mean_prpe_median_start_end_data, mean_hphr_median_start_end_data),axis=0), yerr =np.mean((se_prpe_median_start_end_data, se_hphr_median_start_end_data),axis=0), color = 'r', width=0.2, ecolor='k') 
		#   pl.bar([0.3,1.3], np.mean((mean_nrpe_median_start_end_data, mean_lplr_median_start_end_data),axis=0), yerr =np.mean((se_nrpe_median_start_end_data, se_lplr_median_start_end_data),axis=0), color = 'b', width=0.2, ecolor='k')
		#   pl.ylabel('beta values')
		#   pl.xlabel('time')
		#   pl.legend(['high prediction', 'low prediction'], loc='best')
		#   simpleaxis(ax1)
		#   spine_shift(ax1)
		#   pl.axhline(0, color = 'k', linewidth=0.25)  
		#   ax1.set_xticks([0.25, 1.25])
		#   ax1.set_xticklabels(['early', 'late'])
		#   fig.text(0.02, 0.6, 'beta value', ha='center', va='center', rotation='vertical', fontsize=12)
		#   pl.tight_layout()
		#   pl.savefig(os.path.join(self.grouplvl_plot_dir, 'bar_plot_expectancy_betas_%s.pdf'%name))

		###OUTCOME BETAS### 
		for name in (['sound_event_kernel','sound_reward_kernel', 'zscored_beta_sound_reward']): #'diff_betas_event_kernel', 'diff_betas_reward_kernel', 'zscored_diff_betas_reward_kernel',, 'zscored_beta_colour_reward'
			prpe_median_start_end_data = self.gather_data_from_hdfs(group = 'betas_correct_runs', data_type='prpe_median_start_end_data_'+name) # participant x 2 [begin, end]  
			nrpe_median_start_end_data = self.gather_data_from_hdfs(group = 'betas_correct_runs', data_type='nrpe_median_start_end_data_'+name)
			hphr_median_start_end_data = self.gather_data_from_hdfs(group = 'betas_correct_runs', data_type='hphr_median_start_end_data_'+name)
			lplr_median_start_end_data = self.gather_data_from_hdfs(group = 'betas_correct_runs', data_type='lplr_median_start_end_data_'+name)
			#calculate overall median beta values [begin, end] for different trial types
			mean_prpe_median_start_end_data = np.array([np.array(map(np.median, prpe_median_start_end_data[:,i])).mean(axis=0)for i in range(prpe_median_start_end_data.shape[1])])
			mean_nrpe_median_start_end_data = np.array([np.array(map(np.median, nrpe_median_start_end_data[:,i])).mean(axis=0)for i in range(nrpe_median_start_end_data.shape[1])])
			mean_hphr_median_start_end_data = np.array([np.array(map(np.median, hphr_median_start_end_data[:,i])).mean(axis=0)for i in range(hphr_median_start_end_data.shape[1])])
			mean_lplr_median_start_end_data = np.array([np.array(map(np.median, lplr_median_start_end_data[:,i])).mean(axis=0)for i in range(lplr_median_start_end_data.shape[1])])
			#calculate overal SEM for different trial types 
			se_prpe_median_start_end_data = np.array([np.array(map(np.median, prpe_median_start_end_data[:,i])).std(axis=0) for i in range(prpe_median_start_end_data.shape[1])])/np.sqrt(len(prpe_median_start_end_data))
			se_nrpe_median_start_end_data = np.array([np.array(map(np.median, nrpe_median_start_end_data[:,i])).std(axis=0) for i in range(nrpe_median_start_end_data.shape[1])])/np.sqrt(len(nrpe_median_start_end_data))
			se_hphr_median_start_end_data = np.array([np.array(map(np.median, hphr_median_start_end_data[:,i])).std(axis=0) for i in range(hphr_median_start_end_data.shape[1])])/np.sqrt(len(hphr_median_start_end_data))
			se_lplr_median_start_end_data = np.array([np.array(map(np.median, lplr_median_start_end_data[:,i])).std(axis=0) for i in range(lplr_median_start_end_data.shape[1])])/np.sqrt(len(lplr_median_start_end_data))

			#plot group barplot for outcome (all 4 separate reward conditions)
			fig= pl.figure(figsize=(6,4))
			ax1= fig.add_subplot(121)
			ax1.set_title(name + '\n group results (N=19) \n prediction error trials', fontsize=8)
			pl.bar([0,1], mean_prpe_median_start_end_data, yerr =se_prpe_median_start_end_data, color = 'r', width=0.2, ecolor='k') 
			pl.bar([0.3,1.3], mean_nrpe_median_start_end_data, yerr =se_nrpe_median_start_end_data, color = 'b', width=0.2, ecolor='k')
			pl.ylabel('beta values')
			pl.xlabel('time')   
			pl.legend(['prpe', 'nrpe'], loc = 'best')
			simpleaxis(ax1)
			spine_shift(ax1)
			pl.axhline(0, color = 'k', linewidth=0.25)  
			ax1.set_xticks([0.25, 1.25])
			ax1.set_xticklabels(['early', 'late'])
			fig.text(0.02, 0.6, 'beta value', ha='center', va='center', rotation='vertical', fontsize=12)

			ax2= fig.add_subplot(122, sharey=ax1)
			ax2.set_title(name + '\n no prediction error trials', fontsize=8)
			pl.ylabel('beta values')
			pl.xlabel('time')   
			pl.bar([0,1], mean_hphr_median_start_end_data, yerr =se_hphr_median_start_end_data, color = 'r', width=0.2, ecolor='k', alpha=0.5)  
			pl.bar([0.3,1.3], mean_lplr_median_start_end_data, yerr =se_lplr_median_start_end_data, color = 'b', width=0.2, ecolor='k', alpha=0.5)
			pl.legend(['hphr', 'lplr'], loc = 'best')
			simpleaxis(ax2)
			spine_shift(ax2)
			pl.axhline(0, color = 'k', linewidth=0.25)  
			ax2.set_xticks([0.25, 1.25])
			ax2.set_xticklabels(['early', 'late'])
			pl.tight_layout()
			pl.savefig(os.path.join(self.grouplvl_plot_dir, 'bar_plot_beta_outcome_betas_%s_correct_runs.pdf'%name))




















		
# 		# COMPUTE SCALAR VALUES PUPIL:
# 		self.scalar_ind = [np.zeros(dec_time_courses_diff[i,:,:].size, dtype=bool) for i in range(2)] 
# 		self.scalar_ind = np.zeros(dec_time_courses_diff[0,:,:].size, dtype=bool)
# 		self.scalar_ind[time_of_interest[0]*analysis_sample_rate:time_of_interest[1]*analysis_sample_rate] = True
# 		self.scalars = []
# 		for i in range(4):
# 			self.scalars.append(np.array([np.mean(dec_time_courses_diff[i][subj][self.scalar_ind]) for subj in range(len(self.sessions()[0]))]))
# #
#       self.scalars2 = []
#       for i in range(1,4):
#           # self.scalars2.append(self.scalars[i] - np.mean(self.scalars[0]))
#           self.scalars2.append(self.scalars[i] - self.scalars[0])
#           # self.scalars2.append(self.scalars[i])
#
#       MEANS = [np.mean(scalars) for scalars in self.scalars2]
#       SEMS = [sp.stats.sem(scalars) for scalars in self.scalars2]
#
#       p_values = [myfuncs.permutationTest(scalars, np.zeros(len(scalars)), nrand=10000)[1] for scalars in self.scalars2]
#
#       my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
#
#       N = len(MEANS)
#       ind = np.linspace(0,2,N)  # the x locations for the groups
#       bar_width = 0.50       # the width of the bars
#       spacing = [0, 0, 0]
#
#       # FIGURE 1
#       fig = plt.figure(figsize=(4,3))
#       ax = fig.add_subplot(111)
#       for i in range(N):
#           ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr = SEMS[i], color = ['r','b','r'][i], alpha = [1,.5,.5][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
#       myfuncs.simpleaxis(ax)
#       myfuncs.spine_shift(ax)
#       ax.set_ylabel('difference high vs. low reward')
#       ax.axes.get_xaxis().set_visible(False)
#       fig.tight_layout()
#       plt.legend(['flip p = 0.4, temp uncertainty = 0', 'flip p = 1, temp uncertainty = 1', 'flip p = 0.4, temp uncertainty = 1']) #.format(self.cond['condition'][j], self.cond['flip_p'][j], self.cond['temp_uncertainty'][j]), size=12 )
#       plt.axhline(0, ls='--', alpha=0.5)
#       for i in range(N):
#           plt.text(ind[i]+spacing[i], 0.01, 'p = {}'.format(round(p_values[i],3)), horizontalalignment='center')
#       fig.savefig(os.path.join('/home/shared/reward_pupil/combined/figures', 'exp2', 'barplot.pdf'))

		
#   def BISBAS(self):
#
#       scalars = np.mean(np.vstack((self.scalars[1], self.scalars[3])), axis=0)
#
#       for j in range(5):
#           score = np.array([self.BISBAS_scores.values()[i][j] for i in range(len(self.BISBAS_scores.values()))])
#           fig  = myfuncs.correlation_plot(scalars, score )
#           fig.savefig('/home/shared/PUPIL_REWARD_EXP1_3/data/across_subj_figs/BISBAS/BISBAS' + str(j) + '.pdf')   

	# def deconv_reward_probabilities_across_subjects(self, use_domain = 'full', standard_deconvolution = 'basic', microsaccades_added=False): 
	#     """deconv_diff_reward_probabilities_across_subjects takes timepoints from 'deconvolve_reward_probabilities' 
	#      and averages across subjects"""
	#     folder_name = 'deconvolve_reward_probabilities_%s_domain_%s_%s'%(use_domain, standard_deconvolution, str(microsaccades_added)) 
	#     dec_time_courses = self.gather_data_from_hdfs(group = folder_name , data_type= 'dec_time_course')
	#     time_points = self.gather_data_from_hdfs(group= folder_name , data_type = 'time_points')
	#     baseline_times = time_points.mean(axis=0) < 0

	#     dec_time_courses_s = np.array([[myfuncs.smooth(dec_time_courses[i,:,j], window_len=20) - dec_time_courses[i,baseline_times,j].mean() for i in range(dec_time_courses.shape[0])] for j in range(dec_time_courses.shape[-1])]).transpose((1,2,0))
	#     # dec_time_courses_s = np.array([[myfuncs.smooth(dec_time_courses[i,:,j], window_len=10) for i in range(dec_time_courses.shape[0])] for j in range(dec_time_courses.shape[-1])]).transpose((1,2,0))
	#     dec_time_courses_diff = np.array([dec_time_courses_s[:,:,1] - dec_time_courses_s[:,:,3], dec_time_courses_s[:,:,2] - dec_time_courses_s[:,:,0]]).transpose(1,2,0) 

	#     conds = pd.Series(['LP_NR', 'LP_HR', 'HP_NR', 'HP_HR'])
	#     conds_diff = pd.Series(['PRPE', 'NRPE'])

	#     # cis = np.linspace(95, 10, 4)

	#     sn.set(style="ticks")
	#     f = pl.figure()
	#     ax = f.add_subplot(211)
	#     ax = sn.tsplot(dec_time_courses_s, err_style="ci_band", condition = conds, time = time_points.mean(axis = 0)) # ci=cis, 
	#     pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
	#     pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
	#     ax.set_ylim([-0.55,0.55])
	#     sn.despine(offset=10, trim=True)
	#     pl.tight_layout()   
	#     pl.title('Effect reward probability events on pupil dilation (%s_domain, %s_deconvolution)'%(use_domain, standard_deconvolution))

	#     ax = f.add_subplot(212)
	#     ax = sn.tsplot(dec_time_courses_diff, err_style="ci_band", condition = conds_diff, time = time_points.mean(axis = 0)) # ci=cis,
	#     pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
	#     pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
	#     ax.set_ylim([-0.15,0.55])
	#     sn.despine(offset=10, trim=True)
	#     pl.title('Effect of RPE on pupil dilation (%s_domain, %s_deconvolution) '%(use_domain, standard_deconvolution))
	#     pl.tight_layout()   
	#     pl.savefig(os.path.join(self.grouplvl_plot_dir, '%s_domain_%s_deconv_reward_probabilities_across_subjects.pdf'%(use_domain, standard_deconvolution)))
