#!/usr/bin/env python
# encoding: utf-8
"""
exp.py

Created by Tomas Knapen on 2011-02-16.
Copyright (c) 2011 __MyCompanyName__. All rights reserved.
"""

import os, sys, datetime, readline
import subprocess, logging
import pickle, datetime, time

import scipy as sp
from scipy.stats import *
import numpy as np
import matplotlib.pylab as pl
from math import *

import VisionEgg

from VisionEgg.Core import *
from VisionEgg.MoreStimuli import FilledCircle
from VisionEgg.Text import Text
from VisionEgg.WrappedText import WrappedText

import pygame
from pygame.locals import *

from colorsys import hsv_to_rgb
#from termcolor import colored

sys.path.append( os.environ['EXPERIMENT_HOME'] )
#from experiment import *
from Trial import *
from Session import *


class RMLTrial(Trial):
	"""docstring for RMLTrial"""
	def __init__(self, parameters = {}, phase_durations = [], session = None, screen = None, tracker = None):
		super(RMLTrial, self).__init__(parameters = parameters, phase_durations = phase_durations, session = session, screen = screen, tracker = tracker)
		
		self.create_stimuli()
		 
		

	def create_stimuli(self):
		# most important is the color of the fixation mark
		self.color_rgb = hsv_to_rgb(self.parameters['hue'], self.parameters['standard_saturation'], self.parameters['standard_color_value'])

		self.viewport = Viewport(screen = self.screen, size = self.screen.size, stimuli = [self.session.fixation ], anchor = 'lowerleft') 
		self.run_time = 0.0
		
		self.pre_experiment_time = self.pre_stimulation_time = self.stimulus_time = self.post_stimulus_time = 0.0
		self.stimulus_presentation_frames = 0
		
		#sounds
		bits = 16 #16 bit integer array to work with stereo audio mixer 
		duration_sound = 0.75 #duration in seconds
		sample_rate = 44100 #standard sample rate
		max_sample = 2**(bits-1)-1
		n_samples = int(round(duration_sound * sample_rate))
		
		#set noise tone volume from txtfile Tone_matching_analysis.py 
		self.noise_tone_array = np.array(max_sample * np.random.random(size = n_samples) * 2 ** self.parameters['noise_volume'], dtype = np.uint16)

	
	def draw(self):
		"""docstring for draw"""
		
		if self.phase == 2 and ( self.stimulus_time - self.pre_stimulus_time ) < 0.5:
			self.session.fixation.parameters.color = self.color_rgb
		else:
			self.session.fixation.parameters.color = (self.parameters['base_fixation_luminance'], self.parameters['base_fixation_luminance'], self.parameters['base_fixation_luminance'])

		self.session.fixation.parameters.on = True
		self.stimulus_presentation_frames += 1
		
		super(RMLTrial, self).draw( )
	
	def event(self):
		for event in pygame.event.get():
			super(RMLTrial, self).key_event( event )
			if event.type == KEYDOWN:
				if event.key == pygame.locals.K_ESCAPE:	
					self.events.append([-99,VisionEgg.time_func()-self.start_time])
					self.stopped = True
					self.session.stopped = True
				# it handles both numeric and lettering modes 
				elif event.key == pygame.locals.K_RETURN or event.key == pygame.locals.K_t: #press enter (or TR pulse) to start experiment 
					self.events.append([0,VisionEgg.time_func()-self.start_time])
					if self.phase == 0:
						self.phase_forward()
				elif event.key == pygame.locals.K_SPACE: 
					if self.phase != 0: #only register space bar presses after experiment has started 
						self.events.append([1, VisionEgg.time_func() - self.start_time]) #register space bar presses with '1' and save timestamp 
						self.session.reversal_keypresses +=1 	#save self.reversal_keypresses to be able to print it in the terminal later 
	
	def run(self, ID = 0):
		self.ID = ID
		self.sound_fired = 0
		super(RMLTrial, self).run()
		while not self.stopped:
			self.run_time = VisionEgg.time_func() - self.start_time
			if self.phase == 0:
				self.pre_experiment_time = VisionEgg.time_func()
				# this trial phase is ended with a enter press or TR pulse
				if self.ID != 0:
					self.phase_forward()
			if self.phase == 1:
				self.pre_stimulus_time = VisionEgg.time_func()
				if ( self.pre_stimulus_time - self.pre_experiment_time ) > self.phase_durations[1]:
					self.phase_forward()
			if self.phase == 2:
				self.stimulus_time = VisionEgg.time_func()
				if ( self.stimulus_time - self.pre_stimulus_time ) > self.phase_durations[2]:
					self.phase_forward()
			if self.phase == 3:
				self.tone_time = VisionEgg.time_func()
				if self.sound_fired == 0:
					if self.tracker != None:
						self.tracker.sendMessage('trial ' + str(self.ID) + ' sound ' + str(self.parameters['sound']) + ' at ' + str(self.tone_time) )
					if self.parameters['sound'] == 0:
						self.session.play_np_sound(self.noise_tone_array)
					elif self.parameters['sound'] == 1:	
						self.session.play_sound(sound_index = 'reward')
					self.sound_fired = 1
				# this trial phase is timed
				if ( self.tone_time - self.stimulus_time ) > self.phase_durations[3]:
					self.stopped = True
			
			# events and draw
			self.event()
			self.draw()
		
		self.stop()
	


class RMLSession(EyelinkSession):
	"""docstring for RMLSession"""
	def __init__(self,  subject_initials, index_number, tracker_on = False):
		super(RMLSession, self).__init__( subject_initials, index_number)
		self.create_screen( size = (1280, 1024), full_screen = 1 )
		self.set_screen_params( physical_screen_distance = 80.0, physical_screen_size = (40, 30) )
		# self.create_screen( size = (1024, 768), full_screen = 1 )
		# self.set_screen_params( physical_screen_distance = 80.0, physical_screen_size = (40, 30) )
		# self.create_screen( size = (2560, 1440), full_screen = 1 )
		# self.set_screen_params( physical_screen_distance = 50.0, physical_screen_size = (40, 30) )
		# self.create_screen( size = (800, 600), full_screen = 1 )
		# self.set_screen_params( physical_screen_distance = 80.0, physical_screen_size = (40, 30) )
		self.create_output_file_name()
		self.run_nr = index_number

		if tracker_on:
			self.create_tracker()
		else:
			self.create_tracker(tracker_on = False)
		
		self.standard_parameters = {'standard_saturation': 1.0, 'standard_color_value': 1.0, 'base_fixation_luminance': 0.75, 'noise_volume': 0.0 }
		
		# fixation mark and color
		self.screen.parameters.bgcolor = (0.5,0.5,0.5)
		center = ( self.screen.size[0]/2.0, self.screen.size[1]/2.0 )
		self.fixation = FilledCircle(position=center, color=(self.standard_parameters['base_fixation_luminance'], self.standard_parameters['base_fixation_luminance'], self.standard_parameters['base_fixation_luminance']), radius = self.screen.size[0] / 400.0, num_triangles = 601)

		# self.create_trials()

	def create_trials(self):
		"""create_adaptation_trials creates trials for adaptation runs"""
		# now for some 'arbitrary' parameters
		self.approximate_nr_trials = 80
		self.reward_probabilities = np.array([0.2, 0.8])
		self.hues = np.linspace(0,1,len(self.reward_probabilities), endpoint=False) + 0.25
		np.random.shuffle(self.hues)
		self.nr_blocks_in_this_session = np.random.randint(3)+2 #amount of reversal blocks in a run 
		self.block_length = self.approximate_nr_trials / self.nr_blocks_in_this_session
		self.block_durations = np.ones(self.nr_blocks_in_this_session) * self.block_length + np.random.randint(10, size = (self.nr_blocks_in_this_session)) - 5
		self.domains = np.concatenate([np.ones(self.block_durations[i]) * [0,1][i%2] for i in range(self.nr_blocks_in_this_session)])

		self.nr_trials = len(self.domains)
		# timings
		if self.index_number < 0:
			self.fix_color_mean_delay = 1.0
			self.fix_color_std_delay = 0.3

			self.color_sound_mean_delay = 1.0
			self.color_sound_std_delay = 0.3
		else:
			self.fix_color_mean_delay = 3.0
			self.fix_color_std_delay = 0.3

			self.color_sound_mean_delay = 3.0
			
			self.color_sound_std_delay = 0.3

		# create trials
		self.trials = []		
		
		fix_color_delays = (np.random.randn(self.nr_trials) * self.fix_color_std_delay) + self.fix_color_mean_delay
		color_sound_delays = (np.random.randn(self.nr_trials) * self.color_sound_std_delay) + self.color_sound_mean_delay
		self.total_duration = 0
		self.trial_counter = 0
		reward_trials = 0
		punish_trials = 0 
		self.reward_counter = 0
		self.punish_counter = 0 
		self.reversal_keypresses= 0
		
		# sound level for this subject
		try:
			with open('sound_level/' + self.subject_initials + '.txt') as f:
				self.standard_parameters['noise_volume'] = float(f.read())
		except IOError:
			print 'ERROR: no noise volume file found, or contents not read. \nhas this subject run ToneMatching??'
		
		# rewards already earned
		try:
			with open('reward/' + self.subject_initials + '_reward.txt') as f:
				self.reward_total  = float(f.read())
		except IOError:
			print('hier gaat iets mis')
			self.reward_total = 0
		
		
		# self.ordering = np.arange(self.nr_trials)
		# np.random.shuffle(self.ordering)
		for i in range(self.nr_trials):
			# for pre_time in self.pre_lags_secs:
			phase_durs = [-0.0001, 0.5, fix_color_delays[self.trial_counter], color_sound_delays[self.trial_counter]]
			hue = np.random.randint(2)
			this_reward_probability = np.abs(self.reward_probabilities[hue] - self.domains[i])
			if np.random.binomial(1, p=this_reward_probability, size = 1) == 0:
				sound = 0 # noise tone 
				if self.run_nr > -1: 
					punish_trials += 1 				#accumulate punish_trials and reward_counter after the practice run -1
					# self.reward_counter -= 0.05
					self.punish_counter += 0.05     #use this variable to inform participant in the end about losses 
			else:
				sound = 1					# positive tone
				if self.run_nr > -1: 				#accumulate reward_trials and reward_counter after the practice run -1  
					reward_trials += 1
					self.reward_counter += 0.1 

			params = self.standard_parameters
			params.update({ 'sound': sound, 
							'fix_color_delay': fix_color_delays[self.trial_counter],    
							'color_sound_delay': color_sound_delays[self.trial_counter], 
							'domain': self.domains[i], 
							'hue': self.hues[hue], 
							'reward_probability': this_reward_probability,
							'reward_counter': self.reward_counter, 
							'reward_trials': reward_trials, 
							'punish_trials': punish_trials, 
							'punish_counter': self.punish_counter})
			self.trials.append(RMLTrial(parameters = params, phase_durations = np.array(phase_durs), session = self, screen = self.screen, tracker = self.tracker))
			self.trial_counter += 1
			self.total_duration += np.array(phase_durs).sum()

		# self.trials[self.ordering[-1]].phase_durations[-1] = 10.0
		
		self.reward_total += (self.reward_counter - self.punish_counter)
		try:
			os.system('rm reward/' + self.subject_initials + '_reward.txt')
			with open('reward/' + self.subject_initials + '_reward.txt','w') as f:
				f.write(str(self.reward_total))
		except OSError:
			print 'error saving reward amount. Amount was... %f' % self.reward_total
 
		print str(self.trial_counter) + '  trials generated, of which ' + str(reward_trials) + ' were rewarded and ' + str(punish_trials) + ' were punished \nTotal net trial duration amounts to ' + str( self.total_duration/60 ) + ' min.'
		print ('\n\nYOU EARNED ' + str(self.reward_counter) + ' EUR THIS BLOCK \n \n ')
		print ('YOU LOST ' + str(self.punish_counter) + 'EUR THIS BLOCK \n \n')
		
		

	def run(self):
		"""docstring for fname"""
		self.tracker_setup()
		# cycle through trials
		for i in range(len(self.trials)):
			# self.trials[self.ordering[i]].run(ID = i)
			self.trials[i].run(ID = i)
			if self.stopped == True:
					break
		print 'THERE WERE ' + str(self.nr_blocks_in_this_session-1) + ' REVERSALS THIS BLOCK. \nYOU INDICATED THERE WERE ' + str(self.reversal_keypresses) + ' REVERSALS \n\n'
		print 'YOUR REWARD TOTAL IS: ' + str(self.reward_total) + ' EURO'
		self.screen.clear()
		swap_buffers()
		

def main():
	initials = raw_input('Your initials: ')
	run_nr = int(raw_input('Run number: '))
	nr_blocks = 1
	ts = RMLSession( initials,run_nr, tracker_on = True )
	ts.create_trials()
	ts.run()
	ts.close()
	
if __name__ == '__main__':
	VisionEgg.start_default_logging()
	VisionEgg.watch_exceptions()
	main()
