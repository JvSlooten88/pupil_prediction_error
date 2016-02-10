#!/usr/bin/env python
# encoding: utf-8

import os, sys, datetime, pickle
import subprocess, logging, time
import pp
import scipy as sp
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pylab as pl
import pandas as pd
from IPython import embed as shell

sys.path.append( os.environ['ANALYSIS_HOME'] )
import Tools.Operators.HDFEyeOperator
from Tools.Subjects.Subject import Subject
import ReversalLearningSession
import ReversalLearningSessionB 
from group_level import ReversalLearningGroupLevel as group_level
do_preps = False            
	
sjs_all = [
				 ###N=30
					[Subject('ac', '?', None, None, None), ['ac_1_2015-04-08_17.08.20.edf', 'ac_2_2015-04-08_17.20.56.edf', 'ac_3_2015-04-08_17.34.26.edf', 'ac_4_2015-04-08_17.45.55.edf', 'ac_5_2015-04-22_09.05.16.edf', 'ac_6_2015-04-22_09.16.52.edf', 'ac_7_2015-04-22_09.28.06.edf', 'ac_8_2015-04-22_09.46.05.edf', 'ac_9_2015-04-22_10.01.44.edf', 'ac_10_2015-04-22_10.11.40.edf'], 'PE'],
				    [Subject('bl', '?', None, None, None), ['bl_2_2015-04-08_12.15.14.edf', 'bl_3_2015-04-08_12.27.34.edf', 'bl_4_2015-04-08_12.39.23.edf', 'bl_5_2015-04-17_10.25.32.edf','bl_6_2015-04-17_10.36.56.edf','bl_7_2015-04-17_10.49.29.edf', 'bl_8_2015-04-17_11.01.41.edf', 'bl_9_2015-04-17_11.13.47.edf', 'bl_10_2015-04-17_11.24.27.edf'], 'PE'], #'bl_1_2015-04-08_11.59.32.edf',
					[Subject('kv', '?', None, None, None), ['kv_1_2015-04-08_13.17.55.edf', 'kv_2_2015-04-08_13.31.38.edf', 'kv_4_2015-04-22_10.25.42.edf', 'kv_5_2015-04-22_10.38.08.edf', 'kv_6_2015-04-22_10.51.01.edf', 'kv_7_2015-04-22_11.04.00.edf', 'kv_8_2015-04-22_11.17.22.edf', 'kv_9_2015-04-22_11.28.47.edf'], 'PE'], #'kv_3_2015-04-08_13.46.01.edf',
					[Subject('ln', '?', None, None, None), ['ln_1_2015-04-09_16.57.08.edf', 'ln_2_2015-04-09_17.09.36.edf', 'ln_3_2015-04-09_17.23.20.edf', 'ln_4_2015-04-09_17.36.42.edf', 'ln_5_2015-04-22_15.30.49.edf', 'ln_6_2015-04-22_15.43.18.edf', 'ln_7_2015-04-22_15.56.15.edf', 'ln_8_2015-04-22_16.14.06.edf', 'ln_9_2015-04-22_16.25.30.edf', 'ln_10_2015-04-22_16.38.05.edf'], 'PE'],
					[Subject('ms', '?', None, None, None), ['ms_2_2015-04-09_14.49.03.edf', 'ms_3_2015-04-09_15.02.39.edf', 'ms_4_2015-04-23_10.42.37.edf', 'ms_5_2015-04-23_10.55.55.edf', 'ms_6_2015-04-23_11.11.08.edf', 'ms_7_2015-04-23_11.23.33.edf', 'ms_9_2015-04-23_11.47.46.edf', 'ms_10_2015-04-23_11.57.37.edf'], 'PE'], #'ms_1_2015-04-09_14.25.40.edf',, 'ms_8_2015-04-23_11.35.39.edf'
 				    [Subject('mt', '?', None, None, None), ['mt_1_2015-04-08_15.52.50.edf', 'mt_2_2015-04-08_16.05.51.edf', 'mt_3_2015-04-08_16.18.47.edf', 'mt_4_2015-04-08_16.30.13.edf', 'mt_5_2015-05-01_09.01.55.edf', 'mt_6_2015-05-01_09.14.34.edf', 'mt_7_2015-05-01_09.28.07.edf', 'mt_8_2015-05-01_09.41.41.edf', 'mt_9_2015-05-01_09.56.00.edf', 'mt_10_2015-05-01_10.09.34.edf'], 'PE'],
					[Subject('pm', '?', None, None, None), ['pm_1_2015-04-08_09.33.22.edf', 'pm_2_2015-04-08_09.49.05.edf', 'pm_3_2015-04-08_09.59.58.edf', 'pm_4_2015-04-17_09.08.31.edf', 'pm_5_2015-04-17_09.20.47.edf', 'pm_6_2015-04-17_09.33.36.edf', 'pm_7_2015-04-17_09.49.09.edf', 'pm_8_2015-04-17_10.01.35.edf', 'pm_9_2015-04-17_10.12.30.edf'], 'PE'],
					[Subject('ta', '?', None, None, None), ['ta_1_2015-04-08_14.43.37.edf', 'ta_2_2015-04-08_14.54.47.edf', 'ta_3_2015-04-08_15.07.48.edf', 'ta_4_2015-04-08_15.18.52.edf', 'ta_5_2015-04-17_14.13.44.edf', 'ta_6_2015-04-17_14.25.07.edf', 'ta_7_2015-04-17_14.37.21.edf', 'ta_8_2015-04-17_14.50.02.edf', 'ta_9_2015-04-17_15.09.44.edf'], 'PE'], #, 'ta_10_2015-04-17_15.20.27.edf'
					[Subject('bcm', '?', None, None, None), ['bcm_2_2015-04-16_18.36.42.edf', 'bcm_3_2015-04-16_18.49.13.edf', 'bcm_4_2015-04-17_13.00.00.edf', 'bcm_5_2015-04-17_13.11.51.edf', 'bcm_6_2015-04-17_13.24.52.edf', 'bcm_7_2015-04-17_13.36.18.edf', 'bcm_8_2015-04-17_13.47.18.edf', 'bcm_9_2015-04-17_13.59.27.edf'], 'PE'],
					[Subject('jss', '?', None, None, None), ['jss_1_2015-04-16_12.12.09.edf', 'jss_2_2015-04-16_12.24.25.edf', 'jss_3_2015-04-16_12.36.49.edf', 'jss_4_2015-04-16_12.52.16.edf', 'jss_5_2015-04-23_12.08.51.edf', 'jss_6_2015-04-23_12.23.00.edf', 'jss_7_2015-04-23_12.35.53.edf', 'jss_8_2015-04-23_12.49.40.edf', 'jss_9_2015-04-23_12.59.09.edf', 'jss_10_2015-04-23_13.16.13.edf'], 'PE'], 
					[Subject('lha', '?', None, None, None), ['lha_1_2015-04-16_09.29.40.edf', 'lha_2_2015-04-16_09.41.58.edf', 'lha_3_2015-04-16_09.53.52.edf', 'lha_4_2015-04-16_10.05.00.edf', 'lha_5_2015-04-22_17.45.29.edf', 'lha_6_2015-04-22_17.56.04.edf', 'lha_7_2015-04-22_18.07.37.edf', 'lha_8_2015-04-22_18.21.47.edf', 'lha_9_2015-04-22_18.33.54.edf', 'lha_10_2015-04-22_18.46.45.edf'], 'PE'],
					[Subject('mtt', '?', None, None, None), ['mtt_2_2015-04-16_11.02.18.edf', 'mtt_3_2015-04-16_11.14.30.edf', 'mtt_4_2015-04-16_11.26.59.edf', 'mtt_5_2015-04-22_12.57.54.edf', 'mtt_6_2015-04-22_13.10.44.edf', 'mtt_7_2015-04-22_13.20.44.edf', 'mtt_8_2015-04-22_13.33.42.edf', 'mtt_9_2015-04-22_13.44.44.edf', 'mtt_10_2015-04-22_13.55.01.edf'], 'PE'], #'mtt_1_2015-04-16_10.47.57.edf'
					[Subject('sa', '?', None, None, None), ['sa_2_2015-04-16_14.44.53.edf', 'sa_3_2015-04-16_14.56.55.edf', 'sa_4_2015-04-16_15.08.27.edf', 'sa_5_2015-04-22_11.42.11.edf', 'sa_6_2015-04-22_11.54.07.edf', 'sa_7_2015-04-22_12.05.55.edf', 'sa_8_2015-04-22_12.19.08.edf', 'sa_9_2015-04-22_12.33.37.edf', 'sa_10_2015-04-22_12.44.42.edf'], 'PE'], #'sa_1_2015-04-16_14.25.19.edf'
					[Subject('di', '?', None, None, None), ['di_1_2015-05-13_09.42.07.edf', 'di_2_2015-05-13_09.54.33.edf', 'di_3_2015-05-13_10.05.29.edf', 'di_4_2015-05-13_10.17.05.edf', 'di_5_2015-05-15_10.25.29.edf', 'di_6_2015-05-15_10.36.28.edf', 'di_7_2015-05-15_10.48.51.edf', 'di_8_2015-05-15_10.59.52.edf', 'di_9_2015-05-15_11.10.38.edf', 'di_10_2015-05-15_11.22.14.edf' ], 'PE'],
					[Subject('gi', '?', None, None, None), ['gi_3_2015-05-07_09.50.35.edf', 'gi_4_2015-05-07_10.02.51.edf', 'gi_5_2015-05-18_11.33.12.edf','gi_6_2015-05-18_11.46.35.edf','gi_7_2015-05-18_12.00.11.edf', 'gi_8_2015-05-18_12.13.20.edf', 'gi_9_2015-05-18_12.31.58.edf', 'gi_10_2015-05-18_12.46.06.edf'], 'PE'],  #'gi_1_2015-05-07_09.23.15.edf', 'gi_2_2015-05-07_09.36.39.edf',
					[Subject('mat', '?', None, None, None), ['mat_1_2015-05-07_12.15.09.edf', 'mat_3_2015-05-07_12.39.16.edf', 'mat_4_2015-05-15_09.14.14.edf', 'mat_5_2015-05-15_09.24.56.edf', 'mat_6_2015-05-15_09.36.02.edf', 'mat_7_2015-05-15_09.47.25.edf', 'mat_8_2015-05-15_09.57.48.edf', 'mat_9_2015-05-15_10.08.04.edf'], 'PE'], #'mat_2_2015-05-07_12.27.56.edf',
					[Subject('mk', '?', None, None, None), ['mk_3_2015-05-07_14.58.39.edf', 'mk_4_2015-05-07_15.12.39.edf', 'mk_5_2015-05-18_14.04.59.edf', 'mk_6_2015-05-18_14.17.09.edf', 'mk_7_2015-05-18_14.31.38.edf', 'mk_8_2015-05-18_14.42.06.edf', 'mk_9_2015-05-18_14.59.37.edf', 'mk_10_2015-05-18_15.10.54.edf'], 'PE'], #'mk_1_2015-05-07_14.33.47.edf', 'mk_2_2015-05-07_14.47.21.edf', 
					[Subject('mrf', '?', None, None, None), ['mrf_1_2015-05-13_13.13.01.edf', 'mrf_2_2015-05-13_13.29.27.edf', 'mrf_3_2015-05-13_13.41.08.edf', 'mrf_4_2015-05-13_13.52.06.edf', 'mrf_5_2015-05-15_15.25.52.edf', 'mrf_6_2015-05-15_15.39.26.edf', 'mrf_7_2015-05-15_15.50.56.edf', 'mrf_8_2015-05-15_16.03.41.edf', 'mrf_9_2015-05-15_16.15.31.edf', 'mrf_10_2015-05-15_16.27.14.edf'], 'PE'],
					[Subject('vv', '?', None, None, None), ['vv_1_2015-05-07_13.20.36.edf', 'vv_2_2015-05-07_13.34.03.edf', 'vv_3_2015-05-07_13.44.31.edf', 'vv_4_2015-05-15_13.04.12.edf', 'vv_5_2015-05-15_13.16.10.edf', 'vv_6_2015-05-15_13.29.28.edf', 'vv_7_2015-05-15_13.40.57.edf', 'vv_8_2015-05-15_13.51.21.edf', 'vv_9_2015-05-15_14.04.10.edf'], 'PE'],
					[Subject('sg', '?', None, None, None), ['sg_1_2015-05-13_14.27.50.edf', 'sg_2_2015-05-13_14.40.12.edf', 'sg_3_2015-05-13_14.50.26.edf', 'sg_4_2015-05-13_15.00.36.edf', 'sg_5_2015-05-29_14.52.46.edf', 'sg_6_2015-05-29_15.04.00.edf', 'sg_7_2015-05-29_15.14.39.edf', 'sg_8_2015-05-29_15.26.09.edf', 'sg_9_2015-05-29_15.40.14.edf', 'sg_10_2015-05-29_15.50.43.edf'], 'PE'],
					[Subject('emc', '?', None, None, None), ['emc_1_2015-05-13_15.35.04.edf', 'emc_2_2015-05-13_15.45.50.edf', 'emc_3_2015-05-13_15.59.00.edf', 'emc_4_2015-05-13_16.12.52.edf', 'emc_5_2015-06-18_11.07.13.edf', 'emc_6_2015-06-18_11.20.40.edf', 'emc_7_2015-06-18_11.34.48.edf', 'emc_8_2015-06-18_11.46.51.edf', 'emc_9_2015-06-18_12.02.03.edf', 'emc_10_2015-06-18_12.11.40.edf'], 'PE'],
					[Subject('tom', '?', None, None, None), ['tom_6_2015-06-18_09.57.19.edf', 'tom_7_2015-06-18_10.07.28.edf', 'tom_8_2015-06-18_10.17.55.edf', 'tom_9_2015-06-18_10.40.09.edf', 'tom_10_2015-06-18_10.50.31.edf', 'tom_11_2015-07-09_12.45.50.edf', 'tom_12_2015-07-09_12.57.41.edf', 'tom_13_2015-07-09_13.07.54.edf', 'tom_14_2015-07-09_13.23.50.edf', 'tom_15_2015-07-09_13.34.37.edf'], 'PE'], #'tom_1_2015-06-16_10.11.23.edf','tom_2_2015-06-16_10.23.55.edf','tom_5_2015-06-16_11.14.23.edf','tom_3_2015-06-16_10.49.45.edf', 'tom_4_2015-06-16_11.01.15.edf',
					[Subject('des', '?', None, None, None), ['des_1_2015-06-12_14.51.07.edf', 'des_2_2015-06-12_15.02.27.edf', 'des_3_2015-06-12_15.15.32.edf', 'des_4_2015-06-12_15.27.47.edf', 'des_5_2015-06-16_17.20.37.edf', 'des_6_2015-06-16_17.30.52.edf', 'des_7_2015-06-16_17.52.26.edf', 'des_8_2015-06-16_18.02.46.edf', 'des_9_2015-06-18_16.40.46.edf', 'des_10_2015-06-18_16.51.07.edf'], 'PE'],
					[Subject('rmo', '?', None, None, None), ['rmo_1_2015-06-25_11.57.00.edf', 'rmo_3_2015-06-25_12.22.26.edf', 'rmo_6_2015-06-26_10.27.12.edf', 'rmo_7_2015-06-26_10.38.15.edf', 'rmo_8_2015-06-26_10.51.42.edf', 'rmo_9_2015-06-26_11.03.32.edf', 'rmo_10_2015-06-26_11.17.26.edf'], 'PE'], #'rmo_2_2015-06-25_12.09.43.edf', 'rmo_4_2015-06-25_12.33.38.edf', 'rmo_5_2015-06-25_12.49.09.edf',
					[Subject('iv', '?', None, None, None), ['iv_1_2015-06-25_13.29.15.edf', 'iv_2_2015-06-25_13.40.19.edf', 'iv_3_2015-06-25_13.56.27.edf', 'iv_4_2015-06-25_14.11.17.edf', 'iv_5_2015-06-26_13.03.00.edf', 'iv_6_2015-06-26_13.17.43.edf', 'iv_7_2015-06-26_13.31.20.edf', 'iv_8_2015-06-26_13.50.47.edf', 'iv_9_2015-06-26_14.01.17.edf'], 'PE'],
					[Subject('sz', '?', None, None, None), ['sz_1_2015-06-26_12.06.15.edf', 'sz_2_2015-06-26_12.18.43.edf', 'sz_3_2015-06-26_12.33.06.edf', 'sz_4_2015-06-26_12.49.32.edf', 'sz_5_2015-07-03_11.49.47.edf', 'sz_6_2015-07-03_12.00.50.edf', 'sz_7_2015-07-03_12.11.06.edf', 'sz_8_2015-07-03_12.27.09.edf', 'sz_9_2015-07-03_12.39.36.edf', 'sz_10_2015-07-03_12.50.16.edf'], 'PE'],
					[Subject('eo', '?', None, None, None), ['eo_1_2015-06-26_09.27.15.edf', 'eo_2_2015-06-26_09.40.30.edf', 'eo_3_2015-06-26_09.52.05.edf', 'eo_4_2015-06-26_10.06.43.edf', 'eo_5_2015-07-03_10.39.59.edf', 'eo_6_2015-07-03_10.51.16.edf', 'eo_7_2015-07-03_11.05.44.edf', 'eo_8_2015-07-03_11.15.55.edf', 'eo_9_2015-07-03_11.26.33.edf', 'eo_10_2015-07-03_11.37.34.edf'], 'PE'],
					[Subject('oc', '?', None, None, None), ['oc_1_2015-07-09_09.50.41.edf', 'oc_2_2015-07-09_10.06.21.edf', 'oc_3_2015-07-09_10.19.30.edf', 'oc_4_2015-07-09_10.30.40.edf', 'oc_5_2015-07-09_10.43.34.edf', 'oc_6_2015-07-16_10.23.47.edf', 'oc_7_2015-07-16_10.34.31.edf', 'oc_8_2015-07-16_10.49.26.edf', 'oc_9_2015-07-16_10.59.51.edf', 'oc_10_2015-07-16_11.16.34.edf'], 'PE'],
					[Subject('lr', '?', None, None, None), ['lr_1_2015-07-21_15.04.36.edf', 'lr_2_2015-07-21_15.17.50.edf', 'lr_3_2015-07-21_15.31.56.edf', 'lr_4_2015-07-21_15.43.07.edf', 'lr_5_2015-07-21_15.54.26.edf', 'lr_6_2015-07-23_16.30.16.edf', 'lr_7_2015-07-23_16.41.51.edf', 'lr_8_2015-07-23_16.56.20.edf', 'lr_9_2015-07-23_17.11.26.edf', 'lr_10_2015-07-23_17.21.18.edf'], 'PE'],
					[Subject('jw', '?', None, None, None), ['jw_1_2015-08-06_16.25.11.edf', 'jw_2_2015-08-06_16.37.32.edf', 'jw_3_2015-08-06_16.51.28.edf', 'jw_4_2015-08-06_17.30.39.edf', 'jw_5_2015-08-10_09.53.42.edf', 'jw_6_2015-08-10_10.04.40.edf', 'jw_7_2015-08-10_10.17.52.edf', 'jw_8_2015-08-10_10.35.33.edf', 'jw_9_2015-08-10_10.46.27.edf'], 'PE'], #,

# #excluded participants: 
				 ### ma: too much blinks 
				 #[Subject('ma', '?', None, None, None), ['ma_1_2015-04-16_15.48.54.edf', 'ma_2_2015-04-16_16.03.14.edf', 'ma_4_2015-04-22_14.08.02.edf', 'ma_5_2015-04-22_14.22.14.edf', 'ma_6_2015-04-22_14.36.19.edf', 'ma_7_2015-04-22_14.47.06.edf', 'ma_8_2015-04-22_15.06.09.edf', 'ma_9_2015-04-22_15.17.32.edf'], 'PE'], #, 'ma_3_2015-04-16_16.15.49.edf'
				 ### paid no attention to the task 
				 # [Subject('gk', '?', None, None, None), ['gk_1_2015-04-09_15.37.43.edf', 'gk_2_2015-04-09_15.50.35.edf', 'gk_3_2015-04-09_16.04.42.edf', 'gk_4_2015-04-09_16.17.41.edf', 'gk_5_2015-04-17_15.33.36.edf', 'gk_6_2015-04-17_15.44.14.edf', 'gk_7_2015-04-17_15.56.23.edf', 'gk_8_2015-04-17_16.08.36.edf', 'gk_9_2015-04-17_16.27.33.edf', 'gk_10_2015-04-17_16.39.12.edf'], 'PE'],

]

def run_subject(sj, do_preps = False, exp_name = 'reward_prediction_error'):
	 
	 #change data directory depending on the used server
	 if os.uname()[1] == 'aeneas': 
		data_folder = '/home/shared/reward_pupil/5/data/'
	 elif os.uname()[1] == 'login2.lisa.surfsara.nl' or 'lisa': 
		data_folder = '/home/jslooten/projects/pupil_prediction_error/5/data/'

	 raw_data  = [os.path.join(data_folder, 'raw', sj[0].initials, run) for run in sj[1]] 
	 aliases = [sj[2] + '_' + run.split('_')[1] for run in sj[1]]
	 
	 pes = ReversalLearningSession.ReversalLearningSession(subject = sj[0], experiment_name = exp_name, project_directory = data_folder, pupil_hp = 0.1, version = sj[-1], aliases = aliases)
	 pes_B = ReversalLearningSessionB.ReversalLearningSessionB(subject = sj[0], experiment_name = exp_name, project_directory = data_folder, pupil_hp = 0.1, version = sj[-1], aliases = aliases)
	 
	 # PREPROCESSING:
	 if do_preps: # preparations
			#pes.remove_HDF5()
			pes.import_raw_data(raw_data, aliases)
			pes.import_msg_data(aliases)
			pes.import_gaze_data(aliases)
			pass
	 
	 ##methods for analysing pupil signals 
	 # pes.prepocessing_report()
	 # pes.events_and_signals_in_time(do_plot=False)
	 #pes.filter_bank_pupil_signals(do_plot=True, zscore=True)
	 #pes_B.events_and_signals_in_time_behav(plot_reversal_blocks=False)
	 # # pes.events_and_signals_in_time_TD()
	 pes.deconvolve_colour_sounds_nuisance(deconvolution = 'nuisance_standard_events_rewards', analysis_sample_rate=20)
	 #pes.deconvolve_colour_sounds_nuisance(deconvolution = 'nuisance', analysis_sample_rate=20)
	 # for dom in ['first_3split', 'third_3split','full']: #'first_3split', 'third_3split', # ,'first','second','full''first_3split','third_3split'
	 #    ##pes.deconvolve_full(use_domain=dom)
	 #     pes.deconvolve_full_FIR(use_domain=dom, analysis_sample_rate=10)
	 #pes.pupil_baseline_amplitude(signal_for_baseline='phasic_at_fixation')
	 pes.pupil_baselines_phasic_amplitude_correlations()
	 # pes.kernel_fit_gamma()
	 #pes.single_trial_GLM_one_gamma_kernel()
	 #pes.single_trial_GLM_dual_gamma_kernel()
	 #pes.single_trial_GLM_one_gamma_kernel_results()    
	 #pes.single_trial_GLM_dual_gamma_kernel_results()
	 #pes.check_each_trial_condition()


	 #methods for analysing pupil signals + behavioural data 
	  #pes_B.deconvolve_colour_sound_button(microsaccades_added=False)
	 # pes_B.deconvolve_blinks_colour_button()      
	 # for dom in ['clear', 'unclear']:#'clear' 
	 #    pes_B.deconvolve_clear_domain(use_domain=dom)
	 #    pes_B.deconvolve_clear_domain_covariates()
	#for resid in ['sound','no_sound']:  #'sound''no_sound',
	 # for dom in ['first_3split', 'third_3split', 'full']: #'first', 'second', 
		# pes_B.event_related_average(use_domain=dom)
	 # for dom in ['full']: #'first', 'second', 'first_3split','third_3split'
	 #   pes_B.deconvolve_ridge_covariates(use_domain=dom, best_sim_params='average_TD_params')
	 #pes_B.detrend_pupil_signals()
	 #pes_B.filter_detrended_pupil_signals()
	 # pes_B.pupil_around_keypress()
	 # pes_B.calculate_slope_and_acceleration_pupil_around_keypress()
	 # pes_B.pupil_around_keypress_filterbank_signals(zscore=True)
	 # pes_B.pupil_around_keypress_filterbank_signals(zscore=False)
	 # pes_B.calculate_distance_and_amplitude_reversals_to_peak_and_keypresses()
	 # pes_B.calculate_powerspectra_pupil_and_experiment()
	 # pes_B.calculate_time_frequency_spectrum_per_run()
	 # #pes_B.deconvolve_ridge_filtered_detrended_pupil_baseline()
	 # pes_B.correlate_tonic_phasic_pupil_around_keypress()
	 # pes_B.deconvolve_ridge_phasic_pupil_and_tonic_baselines(zscore=True)
	 # pes_B.deconvolve_ridge_phasic_pupil_and_tonic_baselines(zscore=False)


	 #pes_B.TD_states(do_zoom='zoom', do_sim=False, do_plot=True, best_sim_params='average_TD_params') #'individual_TD_params'
	 #pes_B.regress_TD_prediction_error(standard_deconvolution='sound', do_zscore='z_scored') #standard_deconvolution = 'no_sound', do_zscore= 'raw'
	 # #pes_B.single_trial_GLM_dual_kernel_behav(microsaccades_added=False)
	 #pes_B.single_trial_GLM_dual_results_behav() 
	 #pes_B.individual_IRF() 

	 return True
	 


def analyze_subjects(sjs_all, do_preps, run_parallel=False):
	
	#run all selected participants (no initials are given in command line)
	if len(sjs_all) > 1 and len(sys.argv)<2: 
		### run subjects parallel
		if run_parallel == True: 
			
			job_server = pp.Server(ppservers=()) 
	 
			start_time = time.time()
			jobs = [(sj, job_server.submit(run_subject,(sj, do_preps), (), ("ReversalLearningSession", "ReversalLearningSessionB"))) for sj in sjs_all]
			
			results = []
			for s, job in jobs:
				 job()
	 
			print "Time elapsed: ", time.time() - start_time, "s"
			job_server.print_stats()

		 ### run subjects serially 
		else:  
			for idx, sj in enumerate(sjs_all): 

				run_subject(sjs_all[idx], do_preps=do_preps) 
	
	#run specific participant who's initials were given to the command line 
	elif len(sjs_all) > 1 and len(sys.argv)==2:  

		sj_initials = [sj[0].initials for sj in sjs_all] 
	 	sj_index = sj_initials.index(sys.argv[1]) 
		run_subject(sjs_all[sj_index], do_preps = do_preps) 
	
	#normal single subject analysis 
	else:
			run_subject(sjs_all[0], do_preps = do_preps)  

#######################################################################

def group_level_analysis(sjs_all, exp_name='reward_prediction_error'):  
	 data_folder = '/home/shared/reward_pupil/5/data/'
	 sessions = []
	 for sj in sjs_all:
			aliases = [sj[2] + '_' + c.split('_')[1] for c in sj[1]]
			sessions.append(ReversalLearningSession.ReversalLearningSession(subject = sj[0], experiment_name = exp_name, project_directory = data_folder, pupil_hp=0.04, version = sj[-1], aliases = aliases))

	 #group level object
	 pes_gl = group_level(sessions = sessions, data_folder = data_folder, exp_name = exp_name)    

	 #pes_gl.basic_deconv_results_across_subjects(group = 'deconvolve_colour_sound')
	 #pes_gl.deconv_standard_keypress_across_subjects()
	 # #pes_gl.deconv_diff_colours_across_subjects(use_domain='full')
	 #for dom in ['full']: #'first', 'second','first_3split','third_3split'
	     #pes_gl.deconvolve_full_across_subjects(use_domain=dom, baseline=True)
	 	  #pes_gl.deconvolve_full_FIR_across_subjects(use_domain=dom)
	 #       pes_gl.deconv_reward_probabilities_across_subjects(use_domain = dom, standard_deconvolution=standard, microsaccades_added=False) 
	 #pes_gl.compare_first_last_domain_deconvolve_full(baseline=True, use_domain ='first_third_domain')
	 #pes_gl.gather_data_from_hdfs()
	 #pes_gl.gather_data_from_pickles() 
	 #pes_gl.gather_data_from_npzs() 
	 #pes_gl.gather_blink_rates()
	 #pes_gl.inspect_residuals()
	 # pes_gl.pupil_baseline_amplitude_overview(signal_for_baseline='tonic_at_fixation')
	 # pes_gl.pupil_baseline_amplitude_overview(signal_for_baseline='phasic_at_fixation')

	 #pes_gl.inspect_deconvolutions()
	 #pes_gl.single_trial_GLM_one_kernel_group_results()
	 # pes_gl.single_trial_GLM_dual_kernel_group_results()
	 #for resid in ['sound', 'no_sound']: 
	 # for dom in ['clear', 'unclear']: #'clear', 
	 #    pes_gl.clear_domain_group_results(use_domain=dom, baseline =True)
	 #pes_gl.ANOVA(use_domain='first_third_domain', microsaccades_added=False, baseline=True, standard_deconvolution = 'no_sound')
	 #pes_gl.linear_mixed_model(use_domain='clear', microsaccades_added=False, baseline=True, standard_deconvolution = 'sound')
	 #pes_gl.BISBAS(standard_deconvolution='sound', baseline=True)
	 #pes_gl.event_related_average_across_subjects()
	 #pes_gl.TD_states_across_subjects(do_zoom='zoom') #'zoom'
	 #pes_gl.regress_TD_prediction_error_across_subjects(standard_deconvolution='sound', baseline=True)
	 #pes_gl.deconvolve_ridge_covariates_across_subjects(best_sim_params='average_TD_params', baseline=True, which_covariates='detrended_tonic_pupil_baseline_zscore')
	 #pes_gl.pupil_around_keypress_across_subjects(padding=True, detrending=True)
	 #pes_gl.BISBAS(baseline=False)
	 #pes_gl.detrend_pupil_signals_across_subjects(do_plot==False)
	 #pes_gl.deconvolve_detrended_pupil_across_subjects()
	 #pes_gl.calculate_distance_reversals_to_peak_and_keypresses_across_subjects()
	 #pes_gl.deconvolve_sound_using_baseline_slope_across_subjects() 
	 #pes_gl.correlate_tonic_phasic_pupil_around_keypress_across_subjects()
	 #pes_gl.ridge_phasic_pupil_tonic_baselines_across_subjects()
	 #pes_gl.pupil_around_keypress_tonic_baselines_across_subjects(zscore=True)
	 #pes_gl.powerspectra_pupil_and_experiment_across_subjects()
	 #pes_gl.baseline_phasic_pupil_correlations()
	 #pes_gl.baseline_phasic_pupil_correlations_reversal_performance_median_split(split_on='steep_peak') ##'steep_peak','high_performers'
	 for dom in ['full', 'first_3split','third_3split']: #'first_3split', 'third_3split', 'first', 'second' 'first_3split', 'third_3split','full'
	 	#pes_gl.correlate_blinkrate_with_prediction_errors(baseline=False, use_domain=dom)#use_domain = dom,
	 	pes_gl.event_related_average_across_subjects(use_domain=dom)





	 

def main(do_preps=do_preps):
	
	#analyze_subjects(sjs_all, do_preps=do_preps, run_parallel=False)   
	group_level_analysis(sjs_all) 
	

	 


if __name__ == '__main__':
	main()





