import math
import numpy
import sys
import heapq
import time
from timeit import default_timer as timer
import pygame as pg

SCALE = 5
OSBTACLE = (44, 31, 65)
WALL = (105, 112, 95)
MOVE = [(221, 79, 67), (34, 176, 188), (39, 110, 33), (188, 176, 34)]
START = (79, 221, 67)
GOAL = (255, 104, 0)
OPEN = (174, 148, 216)
CLOSED = (79, 67, 221)
screen = pg.display.set_mode()
WAIT = 25

class PriorityQueue:
	def __init__ (self):
		self.items = []
	def isEmpty(self):
		return len(self.items) == 0
	def push(self, item, value):
		heapq.heappush(self.items, (value, item))
	def pop(self):		
		return heapq.heappop(self.items)
	def pushpop(self, item, value):
		return heapq.heappushpop(self.items, (value, item))
	def containBetter(self, item, value):
		for i in self.items:
			(val, pos) = i
			if item == pos and val <= value:
				return True
		return False
	def contain(self, item):
		for i in self.items:
			(val, pos) = i
			if item == pos:
				return True	
		return False
	def peek(self):
		(val, item) = self.pop()
		self.push(item,val)
		return item


class Map:
	def __init__ (self, size, data):
		self.size = size
		self.data = data
		self.parents = [[(-1, -1) for x in range(size)] for y in range(size)]
	
	def resetParents(self):
		self.parents = [[(-1, -1) for x in range(self.size)] for y in range(self.size)]

	def surroundNodesPosition(self, node):
		(x, y) = node
		return [(x-1,y-1),(x-1,y),(x-1,y+1),(x,y-1),(x,y+1),(x+1,y-1),(x+1,y),(x+1,y+1)]

	def isBound(self, node):
		(x, y) = node
		return 0 <= x < self.size and 0 <= y < self.size

	def isObstacle(self, node):
		(x, y) = node
		return self.data[x][y] == '1'
	
	# Find surrounds node which is not an obstacle
	def successors(self, node):
		ans = []
		surrounds = self.surroundNodesPosition(node)
		for n in surrounds:
			if self.isBound(n) and not self.isObstacle(n):
				ans.append(n)
		return ans

def heuristic(map, goal, n):
	return EuclidDistance(goal, n)

def EuclidDistance(a, b):
	(x0, y0) = a
	(x1, y1) = b
	return math.sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0))


def AStar(start, goal, ourMap):	
	openList = PriorityQueue()
	#closedList = PriorityQueue()		

	f = [[math.inf for x in range(ourMap.size)] for y in range(ourMap.size)]
	g = [[math.inf for x in range(ourMap.size)] for y in range(ourMap.size)]
	f[start[0]][start[1]] = 0
	g[start[0]][start[1]] = 0

	openList.push(start, 0)

	ourMap.parents[start[0]][start[1]] = (-1, -1)	
	
	while(not openList.isEmpty()):
		(val, q) = openList.pop()
		(qx, qy) = q
		successor = ourMap.successors(q)
		for s in successor:
			(x,y) = s			
			
			if s == goal:
				#print("YAY")
				ourMap.parents[x][y] = q
				path = trackPath(ourMap, s)
				drawPath(path)
				#print(path)
				return path
			if g[x][y] > g[qx][qy] + 1:
				g[x][y] = g[qx][qy] + 1
				f[x][y] = g[x][y] + heuristic(ourMap, goal, s)
				openList.push(s, f[x][y])
				drawOpenBlock(s)
				ourMap.parents[x][y] = q
					
		#closedList.push(q, f[qx][qy])
		if q != start:
			drawClosedBlock(q)
	print('No possible path!')
	return []

def fvalue(ourMap, goal, n, g, eps):
	(x, y) = n
	if g[x][y] == math.inf or n == goal:
		return g[x][y]
	return g[x][y] + eps * heuristic(ourMap, goal, n)

def ImprovePath(start, goal, ourMap, g, openList, closedList, inconsList, time_remain, eps = 1):	
	#while f(goal) > min(fvalue in OPEN)
	start_time = timer()
	while not openList.isEmpty() and fvalue(ourMap, goal, goal, g, eps) > openList.peek()[0]:
		cur_time = timer()
		if 1000*(cur_time - start_time) > time_remain:
			return []
		(val, q) = openList.pop()
		(qx, qy) = q
		closedList.push(q, fvalue(ourMap, goal, q, g, eps))
		if q != start:
			drawClosedBlock(q)
		successor = ourMap.successors(q)
		for s in successor:
			(x,y) = s			
			
			if s == goal:
				#print("YAY")
				ourMap.parents[x][y] = q
				path = trackPath(ourMap, s)
				#print(path)
				return path
			if g[x][y] > g[qx][qy] + 1:
				g[x][y] = g[qx][qy] + 1
				if not closedList.contain(s):
					openList.push(s, fvalue(ourMap, goal, s, g, eps))
					drawOpenBlock(s)
				else:
					inconsList.push(s, fvalue(ourMap, goal, s, g, eps))
				ourMap.parents[x][y] = q	
	return []

def ARAStarMain(start, goal, ourMap, file_path, tmax):	
	file = open(file_path, 'w')
	openList = PriorityQueue()
	closedList = PriorityQueue()	
	inconsList = PriorityQueue()
	start_time = timer()

	#Initial value for epsilon
	eps = 3.5
	#Paths found
	path_count = 0
	paths = []
	openList.push(start, eps * heuristic(ourMap, goal, start))

	#Set g(all) = max since there is no path yet
	g = [[math.inf for x in range(ourMap.size)] for y in range(ourMap.size)]
	#Except start position
	g[start[0]][start[1]] = 0
	ourMap.parents[start[0]][start[1]] = (-1, -1)		
	
	
	path = ImprovePath(start, goal, ourMap, g, openList, closedList, inconsList, tmax, eps)
	
	openAndInconsVals = []
	# Get g[s] + h[s] for every s in OPEN and INCONS (value in OPEN and INCONS = fvalue = g[s] + eps * h[s]
	for (val, s) in openList.items:
		openAndInconsVals.append(g[s[0]][s[1]] + heuristic(ourMap, goal, s))
	for (val, s) in inconsList.items:
		openAndInconsVals.append(g[s[0]][s[1]] + heuristic(ourMap, goal, s))
	
	if len(openAndInconsVals) == 0:
		neweps = eps
	else:
		neweps = min(eps, g[goal[0]][goal[1]]/min(openAndInconsVals))
	if len(path) != 0:
		path_count+=1
	else:
		file.write('Out Of Time')
		print("Can't find path")
		return []
	print(path, len(path), neweps)
	writeARAOutput(file, ourMap, path, neweps)
	drawPath(path, path_count - 1)
	paths.append(path)

	cur_time = timer()
	# Increase the quality of answer until found best solution or out of time
	while neweps > 1 and 1000*(cur_time - start_time) < tmax:
		noNewPath = False
		eps = eps - min(eps - 1, 1)
		while not inconsList.isEmpty():
			(v, s) = inconsList.pop()
			openList.push(s, v)
		
		for i in range(len(openList.items)):
			(val, (x, y)) = openList.items[i]
			val = g[x][y] + eps * heuristic(ourMap, goal, (x,y))
			openList.items[i] = (val, (x,y))
	
		heapq.heapify(openList.items)
		for (val, pos) in closedList.items:
			if(pos != start): undrawClosedBlock(pos, path)
		
		cur_time = timer()
		if(1000*(cur_time - start_time) > tmax): 
			return
		# ClosedList = []
		closedList = PriorityQueue()

		#Set improved path, if there is no improved path then keep the old path and return
		path_tmp = ImprovePath(start, goal, ourMap, g, openList, closedList, inconsList, tmax - (cur_time - start_time)*1000, eps)
		if len(path_tmp) != 0 and len(path_tmp) <= len(path):
			path = path_tmp
			paths.append(path)
			path_count += 1
		else:
			# When there is no chance to find a better path 
			if len(path_tmp) == 0 and openList.isEmpty() and inconsList.isEmpty():
				file.write('Out Of Time')
				return paths
			# When cannot find a better path at current epsilon but still can better
			elif len(path_tmp) == 0:
				noNewPath = True
			# When the new path is longer than the old path
			else:
				noNewPath = True
				#return path

		openAndInconsVals = []
		# Get g[s] + h[s] for every s in OPEN and INCONS (value in OPEN and INCONS = fvalue = g[s] + eps * h[s]
		for (val, s) in openList.items:
			openAndInconsVals.append(g[s[0]][s[1]] + heuristic(ourMap, goal, s))
		for (val, s) in inconsList.items:
			openAndInconsVals.append(g[s[0]][s[1]] + heuristic(ourMap, goal, s))
	
		if len(openAndInconsVals) == 0:
			neweps = eps
		else:
			neweps = min(eps, g[goal[0]][goal[1]]/min(openAndInconsVals))
		
		writeARAOutput(file, ourMap, path, neweps)
		
		if noNewPath:
			print("No different path, eps = ", neweps)
		else:
			print(path, len(path), " eps = ", neweps)
			drawPath(path, path_count - 1)
		cur_time = timer()
	
	return paths
		

def trackPath(ourMap, node):
	(x,y) = node
	path = []
	current = node
	while(ourMap.parents[x][y] != (-1,-1)):
		path.append(current)
		current = ourMap.parents[x][y]
		(x, y) = current
	path.append(current)
	path.reverse()	
	return path
	
def readInput(path):
	file = open(path,'r')
	data = file.read()
	data = data.strip().split()
	data = [int(x) for x in data]
	size = data[0]

	start = (data[1],data[2])
	goal = (data[3],data[4])
	data = [str(x) for x in data[5:]]
	ourMapData = numpy.reshape(data, (size,size))
	return size, start, goal, ourMapData

def generateRandomTest(path):
	n=numpy.random.randint(15) + 20;

	start = (numpy.random.randint(int(n/6)),numpy.random.randint(int(n/6)))
	goal = (numpy.random.randint(int(n/6)) + int(n-n/6),numpy.random.randint(int(n/6)) + int(n-n/6))
	outputFile = open(path,"w")
	outputFile.write('%d\n'%n)
	outputFile.write('{} {}\n'.format(start[0],start[1]))
	outputFile.write('{} {}\n'.format(goal[0],goal[1]))

	matrix = numpy.random.randint(2,size=(n,n))
	for i in range(int(n*3/7)**2):
		x = numpy.random.randint(n)
		y = numpy.random.randint(n)
		matrix[x][y]=0
	matrix[start[0]][start[1]]=0
	matrix[goal[0]][goal[1]]=0

	for (x,y),value in numpy.ndenumerate(matrix):
		outputFile.write('%d '%value)
		if(y == (n-1)):
			outputFile.write('\n')

	print('Generated random test file!')

def writeOutput(path, ourMap, solution):
	file = open(path, 'w')
	if solution == []:
		file.writelines(str(-1))
		return
	ansMap = ourMap
	file.write("%d\n"%len(solution))
	file.write(str(solution)[1:-1] + "\n")
	#put solution on map
	for (x,y) in solution:
		ansMap.data[x][y] = 'x'
	(x, y) = solution[0]
	ansMap.data[x][y] = 'S'
	(x, y) = solution[-1]
	ansMap.data[x][y] = 'G'
	for i in range(ansMap.size):
		line = ""
		for j in range(ansMap.size):
			line = line + ansMap.data[i][j]
			if not j == ansMap.size - 1:
				line = line + " "
			else:
				line = line + "\n"
		line = line.replace('0','-').replace('1','o')
		file.write(line)

def writeARAOutput(file, ourMap, solution, eps):
	if solution == []:
		file.writelines(str(-1))
		return
	ansMap = []
	for i in range(ourMap.size):
		ansMap.append([])
		for j in range(ourMap.size):
			ansMap[i].append(ourMap.data[i][j])
	file.write("eps : %f\n"%eps)
	file.write("%d\n"%len(solution))
	file.write(str(solution)[1:-1] + "\n")
	#put solution on map
	for (x,y) in solution:
		ansMap[x][y] = 'x'
	(x, y) = solution[0]
	ansMap[x][y] = 'S'
	(x, y) = solution[-1]
	ansMap[x][y] = 'G'
	for i in range(ourMap.size):
		line = ""
		for j in range(ourMap.size):
			line = line + ansMap[i][j]
			if not j == ourMap.size - 1:
				line = line + " "
			else:
				line = line + "\n"
		line = line.replace('0','-').replace('1','o')
		file.write(line)

def rectPosition(x, y):
	top = 30 + SCALE * x
	left = 30 + SCALE * y
	return (left, top, SCALE - 2, SCALE - 2)

def wait():
	while True:
		for event in pg.event.get():
			if event.type == pg.MOUSEBUTTONDOWN or event.type == pg.QUIT:
				return

def drawMap(ourMap, start, goal):
	global SCALE
	SCALE = int(600 / ourMap.size)
	n = ourMap.size
	screen = pg.display.set_mode([n*SCALE + 60, n*SCALE + 60])
	screen.fill(WALL)
	for x in range(ourMap.size):
		for y in range(ourMap.size):
			if ourMap.data[x][y] == '1':
				color = OSBTACLE
			else:
				color = (255,255,255)
			pg.draw.rect(screen, color, rectPosition(x,y))
	
	(x,y) = start
	color = START
	pg.draw.rect(screen, color, rectPosition(x, y))
	
	(x,y) = goal
	color = GOAL
	pg.draw.rect(screen, color, rectPosition(x, y))

	pg.display.flip()
	wait()

def drawOpenBlock(block):
	pg.event.get()
	(x, y) = block
	color = OPEN
	pg.draw.rect(screen, color, rectPosition(x,y))
	pg.display.flip()
	pg.time.wait(WAIT)

def drawClosedBlock(block):
	pg.event.get()
	(x, y) = block
	color = CLOSED
	pg.draw.rect(screen, color, rectPosition(x,y))
	pg.display.flip()
	pg.time.wait(WAIT)

def undrawClosedBlock(block, path):
	if block in path:
		return
	pg.event.get()
	(x, y) = block
	color = (255, 255, 255)
	pg.draw.rect(screen, color, rectPosition(x,y))
	pg.display.flip()
	pg.time.wait(int(WAIT/3))

def drawPath(path, i = 0):	
	path_copy = path.copy()
	if path == []:
		return
	path_copy.reverse()
	path_copy = path_copy[1:-1]
	for n in path_copy:
		pg.event.get()
		(x,y) = n
		color = MOVE[i%len(MOVE)]
		pg.draw.rect(screen, color, rectPosition(x,y))
		pg.display.flip()
		pg.time.wait(WAIT)
	#wait()
	
def drawMultiPaths(paths):
	for i in range(len(paths)):
		drawPath(paths[i], i)

def main(inp = 'input.txt', out = 'output.txt'):	
	
	flag = input('Want to create new test? (Y/N) ')

	if flag.upper() == "Y":
		generateRandomTest(inp)


	flag = input('A* or ARA*? (1, 0) ')
	tlimit = 1000
	if flag != "1":
		global WAIT
		tlimit = int(input('Enter the time limit (in ms): '))
		if tlimit == 0:
			tlimit = math.inf
		#WAIT = 0

	pg.init()
	size, start, goal, ourMapData = readInput(inp)
	ourMap = Map(size, ourMapData)
	drawMap(ourMap, start, goal)

	start_time = timer()
	if flag == "1": 
		path = AStar(start, goal, ourMap)
		print(path, len(path))
	else:
		path = ARAStarMain(start, goal, ourMap, file_path = out, tmax = tlimit)
	end_time = timer()
	print("time elapsed: ", (end_time - start_time)*1000, "ms")

	# ARA* has output function inside
	if flag == "1": writeOutput(out, ourMap, path)
	#drawPath(path)	
	else: drawMultiPaths(path)
	wait()
	pg.quit()
	return 0

if __name__ == '__main__':
	if len(sys.argv) == 3:
		main(inp = sys.argv[1], out = sys.argv[2])
	else:
		main()	