#!/usr/bin/env python

import cv2.cv as cv;
import time;
import numpy;
import copy;
import string;
import sys;
import math;


#stale
INFTY = 10e10 	#nieskonczonosc
RHO = 1000	#maksymalne przesuniecie obiektu
DELAY = 10		#maksymalny czas 'ukrycia' w kadrze
GATE_WIDTH = 5	#'szerokosc' brarmy
MINPERSONSIZE = 30
AVGPERSONSIZE = 40
MOV_OFFSET = 150
BG_LENGTH = 20
VER_NO = 50
AGING = 100
TOPMARGIN = 20
BOTTOMMARGIN = 20

class FindObject:

	def __init__(self):
		#glowne okno
		cv.NamedWindow('Film', cv.CV_WINDOW_AUTOSIZE)
		cv.MoveWindow('Film', 10, 100)
		#glowne okno
		cv.NamedWindow('Tlo', cv.CV_WINDOW_AUTOSIZE)
		cv.MoveWindow('Tlo', 10, 200)
		#okna do podgladu filtrow
		cv.NamedWindow('Roznica', cv.CV_WINDOW_AUTOSIZE)
		cv.MoveWindow('Roznica', 10, 300)
		cv.NamedWindow('Progowanie', cv.CV_WINDOW_AUTOSIZE)
		cv.MoveWindow('Progowanie', 10, 400)
		cv.NamedWindow('Zamkniecie', cv.CV_WINDOW_AUTOSIZE)
		cv.MoveWindow('Zamkniecie', 10, 500)
		self.koniec = 1



	def configure(self, path, x, y, w, h, l, a, d, e, th, ent, wgate, time, ro, minp, avgp, pp, resf, start, stop):
		self.movie = cv.CreateFileCapture(path)
		self.width = w
		self.height = h
		self.topmargin = TOPMARGIN;
		self.bottommargin = self.height-BOTTOMMARGIN;
		self.left = x
		self.top = y
		self.gate = l
		self.alfa = a
		self.dil_par = d
		self.er_par = e
		self.threshold = th
		#self.threshold = 90
		#lista 'aktywnych' obiektow na poprzednim kadrze
		self.prev_obiekty = list();
		#lista 'aktywnych' obiektow na aktualnym kadrze
		self.new_obiekty = list()
		#lista obiektow ktore opuscily kadr
		self.out = list()
		#zliczone osoby
		self.n = 0
		self.n2 = 0
		self.n3 = 0
		self.enter = ent
		self.rho = ro
		self.delay = time
		self.gate_width = wgate
		self.minpersonsize = minp
		self.avgpersonsize = avgp
		self.predparam = pp
		self.koniec = 0
		self.resultfile = resf
		self.start = start
		self.stop = stop
		movie_dur = cv.GetCaptureProperty(self.movie, cv.CV_CAP_PROP_FRAME_COUNT)
		if self.start > movie_dur:
			self.start = 1
		if (self.stop > movie_dur)or(self.stop==0):
			self.stop = movie_dur
		self.fps = cv.GetCaptureProperty(self.movie, cv.CV_CAP_PROP_FPS)
		#print self.fps

	def cropIm(self):
		self.cropped = cv.CreateImage((self.width, self.height), self.current_frame.depth, self.current_frame.nChannels)
		self.src_region = cv.GetSubRect(self.current_frame, (self.left, self.top, self.width, self.height))
		self.current_frame = cv.CreateImage((self.width, self.height), self.current_frame.depth, self.current_frame.nChannels)
		cv.Copy(self.src_region, self.current_frame)

	def preprocess(self):
		cv.Smooth(self.current_frame,self.current_frame, cv.CV_GAUSSIAN, 3, 0)

	def initialize(self):
		#obraz roznicowy (aktualna ramka - tlo)
		self.difference = cv.CloneImage(self.current_frame)
		#aktualne tlo
		self.background = cv.CloneImage(self.current_frame)
		#srednia kroczaca tla (adaptacyjne tla)
		self.avg_background = cv.CreateImage(cv.GetSize(self.current_frame), cv.IPL_DEPTH_32F, 3)
		#konwersja miedzy typami obrazu
		cv.ConvertScale(self.current_frame, self.avg_background, 1.0, 0.0)
		#szary obraz
		self.grey_image = cv.CreateImage(cv.GetSize(self.current_frame), cv.IPL_DEPTH_8U, 1)

	def bg_adapt(self):
		cv.RunningAvg(self.current_frame, self.avg_background, self.alfa, None)
		cv.ConvertScale(self.avg_background, self.background, 1.0, 0.0)
		cv.ShowImage('Tlo',self.background)

	def bg_detect(self):
		#roznica miedzy aktualna ramka a tlem 4 wersje
		cv.AbsDiff(self.current_frame, self.background, self.difference)
		#konwersja do odcieni szarosci
		cv.CvtColor(self.difference, self.grey_image, cv.CV_RGB2GRAY)
		cv.ShowImage('Roznica',self.grey_image)
		
	def old_bg_table_adapt_detect(self):
		#self.bg_table = [ [ cv.CreateImage((self.width, self.height), self.current_frame.depth, self.current_frame.nChannels) for j in range(VER_NO) ] for i in range(BG_LENGTH) ]
		#self.avg_bg_table = [ cv.CreateImage((self.width, self.height), self.current_frame.depth, self.current_frame.nChannels) for i in range(BG_LENGTH) ]
		#self.bg_table_size = [ 1 for i in range(BG_LENGTH) ]
		mini = 0
		minj = 0
		minval = INFTY
		for i in range(BG_LENGTH):
			for j in range (self.bg_table_size[i]):
				cv.AbsDiff(self.current_frame, self.bg_table[i][j], self.difference)
				val = cv.Avg(self.difference)
				#print i, ":", j, "->", val[0]
				if (minval > val[0]):
					minval = val[0]
					mini = i
					minj = j

		cv.ShowImage('Tlo',self.bg_table[mini][minj])
		cv.AbsDiff(self.current_frame, self.bg_table[mini][minj], self.difference)

		#print "MINi = ", mini, ":", minj
		cv.RunningAvg(self.current_frame, self.avg_bg_table[mini], self.alfa, None)
		cv.ConvertScale(self.avg_bg_table[mini], self.background, 1.0, 0.0)
		cv.Sub(self.difference, self.background, self.difference)		

		#if (self.bg_table_size[mini] < VER_NO):
			#self.bg_table_size[mini] += 1
			#self.bg_table[mini][self.bg_table_size[mini]-1] = cv.CloneImage(self.current_frame)
		#else:
			#self.bg_table[mini][minj] = cv.CloneImage(self.current_frame)	
			
		cv.CvtColor(self.difference, self.grey_image, cv.CV_RGB2GRAY)
		cv.ShowImage('Roznica',self.grey_image)
		
	def bg_table_adapt_detect(self):
		#self.bg_table = [ [ cv.CreateImage((self.width, self.height), self.current_frame.depth, self.current_frame.nChannels) for j in range(VER_NO) ] for i in range(BG_LENGTH) ]
		#self.avg_bg_table = [ cv.CreateImage((self.width, self.height), self.current_frame.depth, self.current_frame.nChannels) for i in range(BG_LENGTH) ]
		#self.bg_table_size = [ 1 for i in range(BG_LENGTH) ]
		mini = 0
		minj = 0
		minval = INFTY
		
		#wyszukiwanie aktuanego indeksujacego tla w wektorze srednich kroczacych tel
		for i in range(BG_LENGTH):
			cv.ConvertScale(self.avg_bg_table[i], self.background, 1.0, 0.0)
			cv.AbsDiff(self.current_frame, self.background, self.difference)
			val = cv.Avg(self.difference)
			#print i, ":", j, "->", val[0]
			if (minval > val[0]):
				minval = val[0]
				mini = i
			for j in range(VER_NO):
				if self.bg_aging[i][j] > 0:
					self.bg_aging[i][j] -= 1


		minval_i = minval
		
		minval = INFTY
		minaging = AGING
		agedFound = False
		#wyszukiwanie najlepiej dopasowanego tla w wektorze najlepiej indeksujacego tla
		for j in range(VER_NO):
			if (agedFound):
				if (self.bg_aging[mini][j] <= 0):
					cv.AbsDiff(self.current_frame, self.bg_table[mini][j], self.difference)
					val = cv.Avg(self.difference)
					if (minval > val[0]):
						minval = val[0]
						minj = j
			else:
				if (self.bg_aging[mini][j] > 0):
					cv.AbsDiff(self.current_frame, self.bg_table[mini][j], self.difference)
					val = cv.Avg(self.difference)
					if (minval > val[0]):
						minval = val[0]
						minj = j
				else:
					cv.AbsDiff(self.current_frame, self.bg_table[mini][j], self.difference)
					val = cv.Avg(self.difference)
					minval = val[0]
					minj = j
					#print "CInd: ", mini," : ", j," : ",self.bg_aging[mini][j] 						
					agedFound = True
		
		minval_j = minval
		#print agedFound
		
		if ((minval_i==0)and(minval_j==0)):
			minval_i = 1;
					
		sum_val = minval_i + minval_j

		#pokazanie tla indeksujacego: AvgTlo oraz najlepiej dopasowanego tla: Tlo
		cv.ConvertScale(self.avg_bg_table[mini], self.background, 1.0, 0.0)
		cv.ShowImage('AvgTlo',self.background)
		cv.MoveWindow('AvgTlo', 230, 200)		

		cv.ShowImage('Tlo', self.bg_table[mini][minj])			
		#wyswietlenie indeksow
		#print "min = ", mini, ":", minj, " -> RA = ", minval_i, " CA = ", minval_j
		
		#uaktualnienie najlepiej indeksujacego tla
		#print "ALFA ", self.alfa
		self.alfa = 0.05
		cv.RunningAvg(self.current_frame, self.avg_bg_table[mini], self.alfa, None)
		cv.ConvertScale(self.avg_bg_table[mini], self.background, 1.0, 0.0)
		#cv.Sub(self.difference, self.background, self.difference)			
		
		cv.AbsDiff(self.current_frame, self.bg_table[mini][minj], self.difference)

		self.avg_difference = cv.CloneImage(self.difference)				
		cv.ConvertScale(self.avg_bg_table[mini], self.background, 1.0, 0.0)
		cv.AbsDiff(self.current_frame, self.background, self.avg_difference)
		
		cv.Min(self.difference,self.avg_difference,self.difference)

		#self.temp = cv.CloneImage(self.difference)		
		#cv.AddWeighted(self.bg_table[mini][minj], (1.0*minval_i/sum_val), self.background, (1.0*minval_j/sum_val), 0, self.temp)
		#cv.AbsDiff(self.current_frame, self.temp, self.difference)
		#cv.ShowImage('WazonaSuma',self.temp)		
		
		
		#ustawienie obrazu do dalszego przetwarzania "self.grey_image"
		cv.CvtColor(self.difference, self.grey_image, cv.CV_RGB2GRAY)
		
		#wyswietlenie obrazu
		cv.ShowImage('Roznica',self.grey_image)
		
		self.temp_grey = cv.CloneImage(self.grey_image)				
		cv.Threshold(self.grey_image, self.temp_grey, self.threshold, 255, cv.CV_THRESH_BINARY)

		w_size = 40;
		h_size = 40;
		w_step = 5;
		h_step = 5;
		det_threshold = 30
		
		possible_objects = list()
		
		#for j in range(0,self.height-h_size,h_step):
		#	for i in range(0,self.width-w_size,w_step):		
		#		self.testowa = cv.CloneImage(self.current_frame)
		#		pixcounter = 0;
		#		for y in range(j, j+h_size):
		#			for x in range(i, i+w_size):
		#				if (self.temp_grey[y,x]>125): 
		#					pixcounter += 1
		#		
		#		detector = 100.0*pixcounter/(w_size*h_size)
		#		#print "x = ", i, ", y = ", j, ", P = ", detector, "%"
		#		
		#		if (detector > det_threshold):
		#			cv.Rectangle(self.testowa, (i,j), (i+w_size, j+h_size), cv.CV_RGB(0, 255, 0), 1)
		#			possible_objects.append([i,j])
		#		else:
		#			cv.Rectangle(self.testowa, (i,j), (i+w_size, j+h_size), cv.CV_RGB(255, 0, 0), 1)
		#							
		#		#cv.ShowImage('TEST',self.testowa)
		#		#cv.MoveWindow('TEST', 230, 100)
		#		#cv.WaitKey(1);
		#		#print self.testowa[j,i],",",
		
		self.testowa = cv.CloneImage(self.current_frame)
		
		for k in range(len(possible_objects)):
			cv.Rectangle(self.testowa, (possible_objects[k][0],possible_objects[k][1]), (possible_objects[k][0]+w_size, possible_objects[k][1]+h_size), cv.CV_RGB(255, 0, 0), 1)

		cv.ShowImage('TEST',self.testowa)
		cv.MoveWindow('TEST', 230, 100)
		#cv.WaitKey(1);
		#print "#", self.licznik, " ", agedFound


		#uaktualnienie wektora tel w wektorze najlepiej indeksujacych tel
		#if (self.bg_table_size[mini] < VER_NO):
		#	self.bg_table_size[mini] += 1
		#	self.bg_table[mini][self.bg_table_size[mini]-1] = cv.CloneImage(self.current_frame)
		#	self.bg_aging[mini][self.bg_table_size[mini]-1] = AGING
		#else:
		#	if (agedFound):
		#		self.bg_table[mini][minj] = cv.CloneImage(self.current_frame)
		#		self.bg_aging[mini][minj] = AGING

		#print "AT: ",self.bg_aging[mini][minj]

		if (agedFound):
			self.bg_table[mini][minj] = cv.CloneImage(self.current_frame)
			self.bg_aging[mini][minj] = AGING

	def processIm(self):
		#binaryzacja
		cv.Threshold(self.grey_image, self.grey_image, self.threshold, 255, cv.CV_THRESH_BINARY)
		cv.ShowImage('Progowanie',self.grey_image)
		#zamkniecie = dylatacja + erozja
		#dylatacja
		cv.Dilate(self.grey_image, self.grey_image, None, self.dil_par)
		#erozja
		cv.Erode(self.grey_image, self.grey_image, None, self.er_par)
		cv.ShowImage('Zamkniecie',self.grey_image)

	def dimensions(self, pt1, pt2, local_frame):
		# pole obiektu
		size = 0;
		for i in range(pt1[0],pt2[0]):
			for j in range(pt1[1], pt2[1]):
				if (local_frame[j,i] > 50):
					size += 1

		#srodek prostokata opisujacego - bedzie reprezentowac obiekt
		#sr_pt = ((pt1[0] + pt2[0])/2, (pt1[1] + pt2[1])/2)

		dim = (pt2[0] - pt1[0], pt2[1] - pt1[1], size)
		return dim

	def countObjects(self,dim):
		#return 1
		return 1+(dim[0]-self.minpersonsize)/self.avgpersonsize

	def centersList(self, dim, objCount, pt1, pt2):
		centers = list()
		for n in range(objCount):
				pt = (int(pt1[0] + (pt2[0]-pt1[0])*(2*n+1)/(2*objCount)), int((pt1[1] + pt2[1])/2))
				#print "PUNKT = ", pt[0], "OBIEKTOW = ", objCount, "SRODEK = ", sr_pt
				obj = (pt, (dim[0]/objCount,dim[1]/objCount,dim[2]/objCount))
				centers.append(obj)
				#oznaczenie kolkiem ruchomego obiektu
				cv.Circle(self.current_frame, pt, (self.minpersonsize/2), cv.CV_RGB(255, 100, 0), 1)
				#oznaczenie prostokatem obszaru, ktory zajmuje
				cv.Rectangle(self.current_frame, pt1, pt2, cv.CV_RGB(255, 200, 0), 1)
		return centers

	def objDetect(self):
		#punkty konturow
		storage = cv.CreateMemStorage(0)
		local_frame = cv.CloneImage(self.grey_image)
		#znajdowanie konturow - metoda modyfikuje obraz zrodlowy!
		#metoda: cv.CV_CHAIN_APPROX_SIMPLE zwraca ciag punktow konturu
		contour = cv.FindContours(self.grey_image, storage, cv.CV_RETR_LIST, cv.CV_CHAIN_APPROX_SIMPLE)
		points = []
		while contour:
			#obraniczamy kontur prostokatem
			bound_rect = cv.BoundingRect(list(contour))
			#nastepny kontur - potrzebne do nastepnej iteracji
			contour = contour.h_next()
			#wspolrzedne lewego, gornego rogu prostokata obejmujacego ksztalt
			pt1 = (bound_rect[0], bound_rect[1])
			#wspolrzedne prawego, dolnego rogu
			pt2 = (bound_rect[0] + bound_rect[2], bound_rect[1] + bound_rect[3])
			#points.append(pt1)
			#points.append(pt2)

			dim = self.dimensions(pt1, pt2, local_frame)
			objCount = self.countObjects(dim)

			#print "SZEROKOSC: ", dim[0]
			#print "OBIEKTOW: ", objCount
			objList = self.centersList(dim, objCount, pt1, pt2)
			for obj in objList:
				self.obiekty.append(obj)

			#print "Wymiary: ", dim


	def predictPos(self, a):
		#predykcja pozycji /predkosci/ na podstawie aktualnej predkosci - mierzonej na podstawie ostatnich 2 pozycji
		vx = 0
		vy = 0
		if (len(a[1]) > 1):
			vx = a[1][len(a[1])-1][0] - a[1][len(a[1])-2][0]
			vy = a[1][len(a[1])-1][1] - a[1][len(a[1])-2][1]
		v = (vx, vy)
		return v

	def predictPosAvg(self, a, w):
		#predyktor pozycji /predkosci/ na podstawie sredniej kroczacej pozycji
		vx = 0
		vy = 0
		if len(a[1]) > 1:
			for i in range(0,len(a[1]) - 1):
				vx = (1-w)*vx + w*(a[1][i][0] - a[1][i-1][0])
				vy = (1-w)*vy + w*(a[1][i][1] - a[1][i-1][1])
		v = (vx, vy)
		return v

	def distObj(self, a, b):
		#predykcja pozycji obiektu z poprzedniego kadru
		v = self.predictPosAvg(a,self.predparam)
		w = 1
		ws = 1
		#odleglosc miedzy pozycja obiektu z aktualnego kadru a predykowana pozycja obiektu z poprzedniego kadru
		distance = (a[0][0][0]+v[0]-b[0][0])*(a[0][0][0]+v[0]-b[0][0])+(a[0][0][1]+v[1]-b[0][1])*(a[0][0][1]+v[1]-b[0][1])
		dim_dist = (a[0][1][0]-b[1][0])*(a[0][1][0]-b[1][0])+(a[0][1][1]-b[1][1])*(a[0][1][1]-b[1][1])+ws*(a[0][1][2]-b[1][2])*(a[0][1][2]-b[1][2])
		similarity = w * distance + (1-w)*dim_dist;
		return similarity

	def SimpleCounter(self, a):
		#wersja 1 - zliczane jest kazde przekroczenie linii przez trajektorie:
		pocz = False
		koniec = False
		if (self.enter == 1):
			for d in a[0][1]:
				if d[1] < self.gate:
					pocz = True
				if pocz:
					if d[1] > self.gate:
						koniec = True
		else:
			for d in a[0][1]:
				if d[1] > self.gate:
					pocz = True
				if pocz:
					if d[1] < self.gate:
						koniec = True
		if pocz and koniec:
			self.n += 1

	def TrajectoryCounter(self, a):
		#wersja 2 - zliczanie na podstawie poczatku i konca trajektorii obiektu:
		if len(a[0][1])>0:
			if (self.enter == 1):
				if (a[0][1][0][1] < self.gate) and (a[0][1][len(a[0][1])-1][1] > self.gate):
					self.n2 += 1
			else:
				if (a[0][1][0][1] > self.gate) and (a[0][1][len(a[0][1])-1][1] < self.gate):
					self.n2 += 1

	def DoubleGateCounter(self, a):
		#wersja 3 (podwojna linia wejscia):
		pocz = False
		first_gate = False
		koniec = False
		if (self.enter == 1):
			for d in a[0][1]:
				if not pocz:
					if d[1] < self.gate - self.gate_width:
						pocz = True
				else:
					if not first_gate:
						if (d[1] > self.gate - self.gate_width)and(d[1] < self.gate + self.gate_width):
							first_gate = True
						elif (d[1] > self.gate + self.gate_width):
							pocz = False
					else:
						if not koniec:
							if d[1] < self.gate - self.gate_width:
								first_gate = False
							else:
								if d[1] > self.gate + self.gate_width:
									koniec = True
			if pocz and first_gate and koniec:
				self.n3 += 1
		else:
			for d in a[0][1]:
				if not pocz:
					if d[1] > self.gate + self.gate_width:
						pocz = True
				else:
					if not first_gate:
						if (d[1] > self.gate - self.gate_width)and(d[1] < self.gate + self.gate_width):
							first_gate = True
						elif (d[1] < self.gate - self.gate_width):
							pocz = False
					else:
						if not koniec:
							if d[1] > self.gate + self.gate_width:
								first_gate = False
							else:
								if d[1] < self.gate - self.gate_width:
									koniec = True
			if pocz and first_gate and koniec:
				self.n3 += 1

	def objTrack(self):
		temp_obiekty = list()
		for a in self.prev_obiekty:
			#sledzenie ruchomego obiektu:
			#dla kazdego obiektu z poprzedniego kadru,
			#szukamy najblizszego mu obiektu na biezacym kadrze
			#obiekt ten musi znajdowac sie nie dalej niz w promieniu RHO
			#WADA_1: analiza na podstawie odleglosci srodka prostokata obwiedniego
			#UWAGA_1_1: nalezaloby brac pod uwage ksztalt, a nie same przesuniecia
			#UWAGA_1_2: nalezaloby uwzglednic nie tylko srodek (i to jeszcze prostokata bez obrotu), ale caly ksztalt
			temp = INFTY
			best = temp
			best_point = ((-1,-1),(0,0,0))
			for b in self.obiekty:
				#nie sledzimy obiektow poruszajacych sie po polu marginesu
				if (not(((a[0][0][1] > self.bottommargin) and (b[0][1] > self.bottommargin))or((a[0][0][1] < self.topmargin) and (b[0][1] < self.topmargin)))):
					temp = self.distObj(a,b)
					if temp < best:
						#uwzgledniamy tylko obiekty, ktore nie oddalily sie bardziej niz o promien RHO
						if temp < self.rho:
							# print "odleglosc: ", temp
							# print "punkt: ", b
							# print "poprzedni punkt: ", a[0]
							best = temp
							best_point = b
			temp_obiekty.append((a, best_point, best))
			
			#print 'Center', a[0][0];
			if best_point[0] != (-1,-1):
				cv.Circle(self.temporaryImg, (a[0][0][0], a[0][0][1]), 5, cv.CV_RGB(255, 0, 0), 1)
				cv.Circle(self.temporaryImg, best_point[0], 5, cv.CV_RGB(255, 0, 0), 1)
				cv.Line(self.temporaryImg, (a[0][0][0], a[0][0][1]), best_point[0], cv.CV_RGB(255, 0, 0),1)
				cv.Circle(self.temporaryImg, (a[0][0][0], a[0][0][1]), 20, cv.CV_RGB(255, 255, 255), 1)
				cv.Circle(self.temporaryImg, best_point[0], 20, cv.CV_RGB(255, 255, 255), 1)				
				
		cv.ShowImage('ADD',self.temporaryImg)
		cv.MoveWindow('ADD', 460, 100)

		temp_obiekty = sorted(temp_obiekty, key = lambda obiekt: obiekt[2])
		for a in temp_obiekty:
			#obiekt pozostaje we wlasciwej czesci skanera (nie jest na marginesie)
			if (a[1][0][1] < self.bottommargin)and(a[1][0][1] > self.topmargin):
				if a[1][0] != (-1,-1):
					a[0][1].append(a[0][0][0])
					self.new_obiekty.append((a[1], copy.copy(a[0][1]), self.delay, copy.copy(a[0][3])))
					self.obiekty = filter (lambda z: z != a[1], self.obiekty)
				else:
					#obiekt 'zniknal' w biezacym kadrze, uruchamia sie odliczanie DELAY kadrow
					#w tym czasie obiekt moze 'wrocic' do kadru
					if (a[0][2] > 0):
						self.new_obiekty.append( (copy.copy(a[0][0]), copy.copy(a[0][1]), copy.copy(a[0][2]-1), copy.copy(a[0][3]) ) )
					else:
						#jesli po uplynieciu czasu DELAY obiekt nie 'wrocil' jest zapisywany
						#na liste obiektow, ktore opuscily obszar
						#weryfikacja czy trajektoria obiektu, ktory definitywnie opuszcza kadr przekroczyl zadana linie
						self.out.append( (copy.copy(a[0][0]), copy.copy(a[0][1]), copy.copy(a[0][2]), copy.copy(a[0][3]), self.licznik) )
						#print "OBIEKT: ", a[0][3], "->", self.licznik, " TRAJEKTORIA: ", a[0][1], a[0][0][0]
						#print self.licznik-a[0][3]," : ",len(a[0][1])
						self.SimpleCounter(a)
						self.TrajectoryCounter(a)
						self.DoubleGateCounter(a)
			#obiekt wszedl w pole marginesu -> konczymy sledzenie trajektorii
			else:
				#jesli po uplynieciu czasu DELAY obiekt nie 'wrocil' jest zapisywany
				#na liste obiektow, ktore opuscily obszar
				#weryfikacja czy trajektoria obiektu, ktory definitywnie opuszcza kadr przekroczyl zadana linie
				self.out.append( (copy.copy(a[0][0]), copy.copy(a[0][1]), copy.copy(a[0][2]), copy.copy(a[0][3]), self.licznik) )
				#print "OBIEKT: ", a[0][3], "->", self.licznik, " TRAJEKTORIA: ", a[0][1], a[0][0][0]
				#print self.licznik-a[0][3]," : ",len(a[0][1])
				self.SimpleCounter(a)
				self.TrajectoryCounter(a)
				self.DoubleGateCounter(a)

		#self.obiekty = filter (lambda z: (z[0][1] > self.bottommargin)or(z[0][1] < self.topmargin), self.obiekty)
		for b in self.obiekty:
			self.new_obiekty.append((b, [], self.delay, self.licznik))

	def results(self):
		s = "n = " + str(self.n) + " n2 = " + str(self.n2) + " n3 = " + str(self.n3) + "\n"
		plik = open(self.resultfile, "wb")
		plik.write(s)
		i = 1
		for ob in self.out:
			temp = ob[1]
			temp.append(ob[0][0])
			plik.write(str(i)+"; "+str(ob[3])+"; "+str(ob[4])+"; "+str(temp)+"\r\n")
			i += 1
		plik.close()
		

	def run(self):
		speed = 1
		s = ""
		self.licznik = self.start - MOV_OFFSET
		if self.licznik < 1:
			self.licznik = 1
		#przewiniecie filmu do poczatku:
		cv.SetCaptureProperty(self.movie, cv.CV_CAP_PROP_POS_FRAMES, self.licznik);
#		self.licznik = MOV_OFFSET+1
		#pierwsze pare kadrow jest zakloconych
		offset = MOV_OFFSET

		#"rozbieg" filmu
		for i in range(offset):
			self.licznik += 1
			self.current_frame = cv.QueryFrame(self.movie)
			#wycinek kadru
			self.cropIm();
			cv.WaitKey(1)
			cv.ShowImage('Film',self.current_frame)

		#wstepne przetwarzanie obrazu
		self.preprocess()
		#inicjowanie potrzebnych zmiennych
		self.initialize()

		self.avg_background = cv.CreateImage(cv.GetSize(self.current_frame), cv.IPL_DEPTH_32F, 3)
		#konwersja miedzy typami obrazu
		cv.ConvertScale(self.current_frame, self.avg_background, 1.0, 0.0)

		self.bg_table = [ [ cv.CreateImage((self.width, self.height), self.current_frame.depth, self.current_frame.nChannels) for j in range(VER_NO) ] for i in range(BG_LENGTH) ]
		self.avg_bg_table = [ cv.CreateImage(cv.GetSize(self.current_frame), cv.IPL_DEPTH_32F, 3) for i in range(BG_LENGTH) ]
		self.bg_table_size = [ 1 for i in range(BG_LENGTH) ]
		self.bg_aging = [ [ 0 for j in range(VER_NO) ] for i in range(BG_LENGTH) ]
			
		for i in range(BG_LENGTH):
			self.licznik += 1
			self.current_frame = cv.QueryFrame(self.movie)
			#cv.Line(self.current_frame, (self.left+1,self.top+30), (self.left+75, self.top+30), cv.CV_RGB(0, 0, 0),12)
			#wycinek kadru
			self.cropIm();
			self.preprocess()
			cv.WaitKey(1)
			#cv.ShowImage('Film',self.current_frame)
			for j in range(VER_NO):
				self.bg_table[i][j] = cv.CloneImage(self.current_frame)
				self.bg_aging[i][j] = AGING
			cv.ConvertScale(self.current_frame, self.avg_bg_table[i], 1.0, 0.0)
		
		cv.SetCaptureProperty(self.movie, cv.CV_CAP_PROP_POS_FRAMES, 1)
		
		self.licznik = 1
		for i in range(offset):
			self.licznik += 1
			self.current_frame = cv.QueryFrame(self.movie)
			#cv.Line(self.current_frame, (self.left+1,self.top+30), (self.left+75, self.top+30), cv.CV_RGB(0, 0, 0),12)
			#wycinek kadru
			self.cropIm();
			cv.WaitKey(1)
			cv.ShowImage('Film',self.current_frame)

		
		
		#for i in range(BG_LENGTH):
			#for j in range(1,VER_NO):
				#self.licznik += 1
				#self.current_frame = cv.QueryFrame(self.movie)
				##wycinek kadru
				#self.cropIm();
				#cv.WaitKey(1)
				##cv.ShowImage('Film',self.current_frame)
				#self.bg_table[i][j] = cv.CloneImage(self.current_frame)
				##print 100.0*(j+1+i*VER_NO)/(BG_LENGTH*VER_NO),"%"
				##print len(self.bg_table[i])

		#cv.ShowImage('7x3',self.avg_bg_table[6] )

		start = time.time()
		while True:
			self.obiekty = list()
			#poprzedni kadr
			previous_frame = cv.CloneImage(self.current_frame)
			
			#self.hsv_frame = cv.CloneImage(self.current_frame)
			#cv.CvtColor(self.current_frame, self.hsv_frame, cv.CV_RGB2HSV)
			#cv.ShowImage("HSV", self.hsv_frame)
			
			key = cv.WaitKey(speed)
			if key==114: #r
				break
			elif key == 27: #ESC
				self.koniec = 1
				break
			elif key == 112: #p
				nkey = cv.WaitKey(0)
				if nkey==114: #r
					break
				elif nkey == 27:
					self.koniec = 1
					break
			elif key == 119: #w
				if speed > 0:
					speed /= 4
				elif speed == 0:
					speed = 256
			elif key == 115: #s
				if speed < 256:
					speed *= 4
				else:
					speed = 0
			elif key == 32: #spacja
				self.n = 0
				self.n2 = 0
				self.n3 = 0
			elif (key == 44)or(key == 120):
				self.licznik -= int(5*self.fps)
				if self.licznik < self.start:
					self.licznik = self.start
				cv.SetCaptureProperty(self.movie, cv.CV_CAP_PROP_POS_FRAMES, self.licznik)
				self.licznik = int(cv.GetCaptureProperty(self.movie, cv.CV_CAP_PROP_POS_FRAMES))
			elif (key == 46)or(key == 99):
				self.licznik += int(5*self.fps)
				if self.licznik > self.stop:
					self.licznik = self.stop
				cv.SetCaptureProperty(self.movie, cv.CV_CAP_PROP_POS_FRAMES, self.licznik)
				self.licznik = int(cv.GetCaptureProperty(self.movie, cv.CV_CAP_PROP_POS_FRAMES))
			elif (key == 45)or(key == 122):
				self.licznik -= int(60*self.fps)
				if self.licznik < self.start:
					self.licznik = self.start
				cv.SetCaptureProperty(self.movie, cv.CV_CAP_PROP_POS_FRAMES, self.licznik)
				self.licznik = int(cv.GetCaptureProperty(self.movie, cv.CV_CAP_PROP_POS_FRAMES))
			elif (key == 61)or(key == 118):
				self.licznik += int(60*self.fps)
				if self.licznik > self.stop:
					self.licznik = self.stop
				cv.SetCaptureProperty(self.movie, cv.CV_CAP_PROP_POS_FRAMES, self.licznik)
				self.licznik = int(cv.GetCaptureProperty(self.movie, cv.CV_CAP_PROP_POS_FRAMES))				
			if key != -1:
				print "DELAY = ", speed
				#print "FRAME = ", self.licznik, cv.GetCaptureProperty(self.movie, cv.CV_CAP_PROP_POS_FRAMES), cv.GetCaptureProperty(self.movie, cv.CV_CAP_PROP_FRAME_COUNT), cv.GetCaptureProperty(self.movie, cv.CV_CAP_PROP_FOURCC)

			#aktualny kadr
			self.current_frame = cv.QueryFrame(self.movie)
			#cv.Line(self.current_frame, (self.left+1,self.top+30), (self.left+75, self.top+30), cv.CV_RGB(0, 0, 0),12)

			self.licznik = self.licznik + 1;
			#warunek konca filmu
			if (self.licznik > self.stop):
				self.koniec = 1
				break;
			#wycinanie kadru do fragmentu, w ktorym nie ma zaklocen (4 linie)
			self.cropIm()
			#wstepne przetwarzanie obrazu
			self.preprocess()
			
			self.temporaryImg = cv.CloneImage(self.current_frame)
			
			#adaptacja tla
			#self.bg_adapt()
			#roznica miedzy aktualna ramka a tlem 4 wersje
			#self.bg_detect()
			self.bg_table_adapt_detect()
			#przetwarzanie obrazu
			self.processIm()
			#lista obiektow na poprzednim kadrze
			self.prev_obiekty = self.new_obiekty[:]
			#lista obiektow na biezacym kadrze
			self.new_obiekty = list()
			#wykrywanie obiektow
			self.objDetect()
			#sledzenie i zliczanie obiektow
			self.objTrack()

			#rysowanie sciezki (lamanej) przemieszczania sie obiektu
			for ob in self.new_obiekty:
				if len(ob)>2:
					if len(ob[1])>1:
						for ind in range(len(ob[1])-1):
							cv.Line(self.current_frame, ob[1][ind], ob[1][ind+1], cv.CV_RGB(255, 100, 0))
			#wyswietlanie zadanej linii na rysunku
			cv.Line(self.current_frame, (1,self.gate-self.gate_width), (self.current_frame.width, self.gate-self.gate_width), cv.CV_RGB(100, 255, 0))
			cv.Line(self.current_frame, (1,self.gate+self.gate_width), (self.current_frame.width, self.gate+self.gate_width), cv.CV_RGB(100, 255, 0))
			cv.Line(self.current_frame, (1,self.gate), (self.current_frame.width, self.gate), cv.CV_RGB(0, 200, 100))
			cv.Line(self.current_frame, (1,self.topmargin), (self.current_frame.width, self.topmargin), cv.CV_RGB(0, 0, 255))
			cv.Line(self.current_frame, (1,self.bottommargin), (self.current_frame.width, self.bottommargin), cv.CV_RGB(0, 0, 255))


			#wyswietlanie tekstu: zliczanie osob przekraczajacych zadana linie
			font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 3, 8)
			text = "# = " + str(self.licznik) + " n2 =" + str(self.n2)

			#ZLICZANIE
			s1 = "n = " + str(self.n) + " n2 = " + str(self.n2) + " n3 = " + str(self.n3)
			if s!=s1:
				print self.licznik, s1
			s = copy.copy(s1)
			
			cv.PutText(self.current_frame, text, (5,20), font, 255)
			#wyswietlanie
			cv.ShowImage('Film',self.current_frame)
			stop = time.time()
			#print "TIME: ",(stop-start)
			start = stop
			#cv.ShowImage('Filtry',self.grey_image)

if __name__=="__main__":
	if len(sys.argv)>1:
		confile = sys.argv[1]
	else:
		confile = "config.ini"
	while True:
		path = "/Users/anihilati/Downloads/test3.avi"
		x = 200
		y = 30
		width = 160
		height = 70
		line = 35
		alfa = 0.02
		dil = 10
		erod = 5
		thresh = 30
		entr = 0
		gate = GATE_WIDTH
		pdelay = DELAY
		prho = RHO
		pminpersonsize = MINPERSONSIZE
		pavgpersonsize = AVGPERSONSIZE
		predparam = 0.8
		resf = "result"
		start = 1
		stop = 0
		
		f = open(confile, "rb") 		
		while True:
			a = f.readline()	
			if len(a) == 0:
				break;
			b = string.lower(a)
			c = string.split(b," = ")
			if c[0][0]!="#":
				if c[0]=="path":
					path = c[1]
				elif c[0]=="scanner":
					confile1 = c[1]
				elif c[0]=="algorithm":
					confile2 = c[1]
		path = path.replace("\n", "")
		path = path.replace("\r", "")
		confile1 = confile1.replace("\n", "")
		confile2 = confile2.replace("\n", "")
		confile1 = confile1.replace("\r", "")
		confile2 = confile2.replace("\r", "")
		f.close()
		
		#print confile1
				
		f = open(confile1, "rb") 		
		while True:
			a = f.readline()	
			if len(a) == 0:
				break;
			b = string.lower(a)
			#    b = string.replace(b, " ", "")			
			c = string.split(b," = ")
			if c[0][0]!="#":
				if c[0]=='x':
					x = int(c[1])
				elif c[0]=='y':
					y = int(c[1])
				elif c[0]=='width':
					width = int(c[1])
				elif c[0]=='height':
					height = int(c[1])
				elif c[0]=='line':
					line = int(c[1])
				elif c[0]=='entr':
					entr = int(c[1])
				elif c[0]=='gate':
					gate = int(c[1])
				elif c[0]=='resfile':
					resf = c[1]
				elif c[0]=='start':
					start = int(c[1])
				elif c[0]=='stop':
					stop = int(c[1])
		resf = resf.replace("\n", "")
		resf = resf.replace("\r", "")
		f.close()
		
		f = open(confile2, "rb") 	
		while True:
			a = f.readline()	
			if len(a) == 0:
				break;
			b = string.lower(a)
			#    b = string.replace(b, " ", "")			
			c = string.split(b," = ")
			if c[0][0]!="#":
				if c[0]=='alfa':
					alfa = float(c[1])
				elif c[0]=='dil':
					dil = int(c[1])
				elif c[0]=='erod':
					erod = int(c[1])
				elif c[0]=='thresh':
					thresh = int(c[1])
				elif c[0]=='delay':
					pdelay = int(c[1])
				elif c[0]=='rho':
					prho = int(c[1])
				elif c[0]=='minpersonsize':
					pminpersonsize = int(c[1])
				elif c[0]=='avgpersonsize':
					pavgpersonsize = int(c[1])
				elif c[0]=='predparam':
					predparam = int(c[1])
		f.close()
		
		app = FindObject()

		if start>stop:
			stop = 0

		try:
			with open(path):
				app.configure(path, x, y, width, height, line, alfa, dil, erod, thresh, entr, gate, pdelay, prho, pminpersonsize, pavgpersonsize, predparam, resf, start, stop)
				app.run()
				if app.koniec == 1:
					app.results()
					break

		except IOError:
			print 'Brak wskazanego pliku z filmem'
			if app.koniec == 1:
					break
		
		f.close()
		
	cv.DestroyAllWindows()
