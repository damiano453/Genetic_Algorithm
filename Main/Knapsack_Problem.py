#!/usr/bin/env python
#-*- coding: utf-8 -*-

import cmath as mat		#Biblioteka matematyczna
import numpy as np 		#Biblioteka do macierzy
import matplotlib.pyplot as plt
from random import seed, random, randint
from operator import add
#import scipy as sc

'Function draws weights and prizes with ascending value of p/w'
def drawLots(numberOfItems,typeOfLoss):
	'draw weights in range <0.1,1>, with 0.1 steps'
	'     prizes           <1,100>       1.0      '
	seed()
	randArray = np.zeros((1,numberOfItems))
	if typeOfLoss == 'weight':
		for i in range(numberOfItems):
			randArray[0,i] = float(randint(1,10))/10
	elif typeOfLoss == 'prize':
		for i in range(numberOfItems):
			randArray[0,i] = float(randint(1,100))
	else:
		randArray = 0
	return randArray

'Function creates Chromosome witch binary values'
'Chromosome is content of knapshack (1- contains; 0- not)'
def chromosome(numberOfItems,probabilityOfChromosome,maxWeight,prizes,weights):
	'draw member of population'
	seed()
	X = np.zeros((1,numberOfItems),dtype=bool)
	for i in range(numberOfItems):
		x = random()
		if x <= float(probabilityOfChromosome):
			if weights[0,i] <= (maxWeight - np.dot(X,weights.T)):
				'case to prevent overloading knapsack'
				X[0,i] = 1
			else:
				X[0,i] = 0
		else:
			X[0,i] = 0
	return X

'Function creates Population of Chromosomes'
def population(numberOfInduviduals,numberOfItems,probabilityOfChromosome,maxWeight,prizes,weights):
	'draw population'
	pop = np.zeros((numberOfInduviduals,numberOfItems),dtype=bool)
	for i in range(numberOfInduviduals):
		pop[i,:] = chromosome(numberOfItems,probabilityOfChromosome,maxWeight,prizes,weights)
	return pop

'Function returns parameters of Fitness function'
'Fval (averge, min., max., variance)'
def avgFitnes(fit):
	'must be used after fitnes function'
	'avg - averge fitness'
	'minim - min fitness'
	'maxim - max fitness'
	avg = np.sum(fit)/np.shape(fit)[0]
	minArg = np.argmin(fit)
	maxArg = np.argmax(fit)
	minim = fit[minArg]
	maxim = fit[maxArg]
	'var - variance of fitness'
	var = (np.sum((fit-avg)**2))/np.shape(fit)[0]
	return avg,minim,maxim,var,maxArg

'Function counts fitness of every individual'
'Repair Algorithm- randomly deleting items of Chromosome'
'until weight of chromosome is less or equal Max. Weight'
def fitnes(pop,prizes,weights,maxWeight,numberOfItems):
	'Fitnes of individual'
	'prize of individual'
	seed()
	sizeOfPop = np.shape(pop)
	fitValue = np.zeros((sizeOfPop[0],1))
	for i in range(sizeOfPop[0]):
		ite = 0				#Used to avoid infinite loop
		while(pop[i,:].dot(weights.T) > maxWeight):
			ite += 1
			'repeair algorithm'
			'heuristic change until weight is correct'
			x = int(random()*numberOfItems)
			if(pop[i,x] == 1):
				pop[i,x] = 0
			else:
				'To avoid infinite loop'
				if(ite > 30):
					arg = np.argwhere(pop[i]==1)
					pop[i,arg[0,0]] = 0
			w = pop[i,:].dot(weights.T)				#Weight of individual
		fitValue[i,0] = np.sum(prizes.dot(pop[i,:]))
	return fitValue,pop

'Selection by max of prize/weight for individual'
def selection(pop,fit,numOfParcip,weights):
	seed()
	children = np.zeros([pop.shape[0],pop.shape[1]],dtype=bool)
	index = np.zeros([1,numOfParcip],dtype=int)
	weightOfPop = np.dot(pop,weights.T)
	tourWeight = weightOfPop == 0
	weightOfPop[tourWeight==1] = 100000
	for i in range(pop.shape[0]):
		'Children population has the same number of Chromosome'
		'as parents population								  '
		tourPop = np.array(range(pop.shape[0]))
		for j in range(numOfParcip):
			x = randint(0,pop.shape[0]-j-1)
			index[0,j] = tourPop[x]
			tourPop = np.delete(tourPop, x, 0)
		maximInd = np.argmax(fit[index]/weightOfPop[index])
		children[i] = pop[maximInd]
	return children

'Crossover with one point of crossing'
def crossover(pop,probOfCross):
	'For every couple draw if it will be crossed'
	'------//-------- draw position to cross	'
	'couple to cross is 1and2; 3and4 and so on			'
	seed()
	cross = np.zeros([pop.shape[0],2],dtype=int)
	children = np.zeros([pop.shape[0],pop.shape[1]],dtype=bool)
	for i in range(pop.shape[0]):
		if (i%2 == 0):
			'Beacasue we count as couple 1and2...'
			x = random()
			if(x <= probOfCross):
				cross[i,0] = 1
				cross[i,1] = int(randint(1,(pop.shape[1]-2)))
			else:
				cross[i,0] = 0
		else:
			cross[i] = cross[i-1]
	for i in range(int(pop.shape[0]/2)):
		'Make crossover'
		i *= 2		#Because cauples 1and2...
		i = int(i)
		if(cross[i,0] == 1):
			'With crossover'
			children[i] = np.concatenate((pop[i,range(cross[i,1])],pop[i+1,range(cross[i,1],pop.shape[1])]))
			children[i+1] = np.concatenate((pop[i+1,range(cross[i,1])],pop[i,range(cross[i,1],pop.shape[1])]))
		else:
			'Without crossover'
			children[i] = pop[i]
			children[i+1] = pop[i+1]
	return children

'Mutation of Chromosomes'
def mutation(pop,probOfMut):
	seed()
	children = np.zeros([pop.shape[0],pop.shape[1]],dtype=bool)
	for i in range(pop.shape[0]):
		for j in range(pop.shape[1]):
			x = random()
			if (x <= probOfMut):
				'Swap value'
				children[i,j] = ~pop[i,j]
			else:
				children[i,j] = pop[i,j]
	return children

'Percent of same fitness   '
'case of break of algorithm'
def percentOfFit(fit):
	'fitval and number of appearing'
	val_num = np.zeros([fit.shape[0],2])
	val_num[0,0] = fit[0,0]
	'check number of apperaing'
	for i in range(fit.shape[0]):
		for j in range(fit.shape[0]):
			if(val_num[j,0] == fit[i,0]):
				val_num[j,1] += 1
				break
			else:
				if(i == j):
					val_num[j,0] = fit[i,0]
					val_num[j,1] += 1
	'count percent of eppearing'
	arg = np.argmax(val_num[:,1])
	percent = val_num[arg,1]/fit.shape[0]
	return percent

'Generate Population and save to File' 
'Used only for test'
def generateData(numberOfItems,numberOfInduviduals,probabilityOfChromosome):
	numberOfItems = 64
	numberOfInduviduals = 512
	probabilityOfChromosome  = 1/(numberOfItems-10)
	
	'Generate input data'
	weights = drawLots(numberOfItems,'weight')
	prizes = drawLots(numberOfItems,'prize')
	maxWeight = 0.3*np.sum(weights)
	print('Input Data:\nw = ',weights,'\np = ',prizes,'\nW = ',maxWeight,'\n\n')
	
	'Generate parent population'
	pop = population(numberOfInduviduals,numberOfItems,probabilityOfChromosome,maxWeight,prizes,weights)
	print("Parent Population:\n",pop,'\n\n')
	
	'Name of Generated File'
	title = "Dane_Item:"+str(numberOfItems)+"_Individ:"+str(numberOfInduviduals)
	
	np.savez(title, weights=weights, prizes=prizes, maxWeight=maxWeight, pop=pop)
	return 0

'Load Population from File'
'Used only for test'
def loadData(numberOfItems,numberOfInduviduals):
	numberOfItems = 64
	numberOfInduviduals = 256
	
	title = "Dane_Item:"+str(numberOfItems)+"_Individ:"+str(numberOfInduviduals)
	data = np.load(title+'.npz')
	#weights=weights, prizes=prizes, maxWeight=maxWeight, pop=pop)
	
	weights = data['weights']
	prizes = data['prizes']
	maxWeight = data['maxWeight']
	pop = data['pop']
	
	print(pop,'\n',weights.T,'\n',prizes.T,'\n',maxWeight)
	
	return weights,prizes,maxWeight,pop

'''
'Main function of Programm'
'Genetic Algorithm and creating input data'
'''
def main():
	'Parameters of Population'
	numberOfItems = 64									#number should be even (for crossing method)
	numberOfInduviduals = 512							#number should be even (for crossing method)
	newData = 0											#if you want new data
	probabilityOfChromosome  = 1/(numberOfItems-10) 	#Probablity of 1 = "take item" in chromosome 
	
	'Parameters of Algorithm'
	probabilityOfCrossover   = 0.75						
	probabilityOfMutation    = 0.1			
	numberOfTournamentParcip = 3						#Participants of Tournament Selection
	percentOfSameFit         = 0.6						#case for stop of Algorithm
	minValOfIterations = 8								#case for stop algorithm
	maxIterations = 100									#Max iterations of algorithm
	
	print ('Probability of Chromosome: ',probabilityOfChromosome)	#Just for info
	
	
	if (newData == 1):
		'Generates new data'
		generateData(numberOfItems,numberOfInduviduals,probabilityOfChromosome)
	
	'load generated data'
	weights,prizes,maxWeight,pop = loadData(numberOfItems,numberOfInduviduals)
	#print(weights,prizes,maxWeight,pop)
	'''
	'Genetic Algorithm'
	'''
	'Declaration of Variables --> help in raport'
	parent = np.zeros((numberOfInduviduals,numberOfItems),dtype=bool)
	children = np.zeros((numberOfInduviduals,numberOfItems),dtype=bool)
	variancePlot = []								#Variance of Fit
	minimum = []									#Min Fitness
	maximum = []									#Max Fitenss
	averge = []										#Averge of Fitness
	
	'Genetic algorithm'
	for i in range(maxIterations):
		
		if((i/maxIterations*100)%10 == np.array(range(10)).any()):
			'Show progress of Algorithm'
			print('#', sep='', end='', flush=True)
		'Prizes of Individuals and averge prize'
		if(i == 0):
			'For the first iteration only'
			'Used for count properties of fitst population'
			fit,pop = fitnes(pop,prizes,weights,maxWeight,numberOfItems)
			avg,minim,maxim,variance,MAX = avgFitnes(fit)
			
			variancePlot.append(variance)
			minimum.append(minim)
			maximum.append(maxim)
			averge.append(avg)
			print('per of fit: \n',percentOfFit(fit))
			print('Fitness function:\n',fit,'\nAverge Prize:\n',avg,'\nVariance: ',variance,'\n\n')
		percent = percentOfFit(fit)
		if((percent >= percentOfSameFit) and (i >= minValOfIterations)):
			'case of break'
			break
		else:
			'Tournament Selection'
			parents = selection(pop,fit,numberOfTournamentParcip,weights)
			
			'Crossover'
			children = crossover(parents,probabilityOfCrossover)
			
			'Mutation'
			children = mutation(children,probabilityOfMutation)
			pop = children
		fit,pop = fitnes(pop,prizes,weights,maxWeight,numberOfItems)
		avg,minim,maxim,variance,MAX = avgFitnes(fit)
		deltaWEIGHT = (pop[MAX,:].dot(weights.T))-maxWeight			#just to check
		variancePlot.append(variance)
		minimum.append(minim)
		maximum.append(maxim)
		averge.append(avg)
		# percent of the same fitnes
		percent = percentOfFit(fit)
		iteration = i
	argument = np.argmax(fit[:,0])
	
	print("Final Population:\n",pop,'\n\n')
	print('per of fit: \n',percentOfFit(fit))
	print('Fitness function:\n',fit,'\nAverge Prize:',avg,'\nMin Prize:  ',minim,'\nMax Prize:  ',maxim,'\nVariance: ',variance,'\n\n')
	print('varLen: ',len(variancePlot))
	print('Iterations: ',iteration)
	
	print('Best Knapshack: \n',pop[argument]*1,"\nDelta Weight = ",deltaWEIGHT,'Max Weight = ',maxWeight)
	
	'Plots of min,max,avg Fitness'
	plt.plot(averge,'*',label='Średnia')
	plt.plot(maximum,'+',label='Maksimum')
	plt.plot(minimum,'.',label='Minimum')
	
	# Place a legend to the right of this smaller subplot.
	plt.legend()			#bbox_to_anchor=(1., 1., 1., .102), loc=0,ncol=1, mode="expand", borderaxespad=0.)
	plt.xlabel('Numer Generacji')
	plt.ylabel('Wartość funkcji dopasowania')
	title = 'Funkcja dopasowania; l. przedmiotów: '+str(numberOfItems)+'; l. populacji: '+str(numberOfInduviduals)
	plt.title(title)
	plt.grid()
	plt.show()
	
	'Plots of Variance'
	plt.plot(variancePlot,'*--')
	plt.xlabel('Numer Generacji')
	plt.ylabel('Wariancja')
	title = 'Wariancja; l. przedmiotów: '+str(numberOfItems)+'; l. populacji: '+str(numberOfInduviduals)
	plt.title(title)
	plt.grid()
	plt.show()


'Just to run programm'
main()
