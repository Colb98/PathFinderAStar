import math
import numpy
import heapq
import time
import pygame as pg

SCALE = 5
OSBTACLE = (44, 31, 65)
WALL = (105, 112, 95)
MOVE = (221, 79, 67)
START = (79, 221, 67)
GOAL = (255, 104, 0)
OPEN = (174, 148, 216)
CLOSED = (79, 67, 221)
screen = pg.display.set_mode()
WAIT = 150

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


class Map:
	def __init__ (self, size, data):
		self.size = size
		self.data = data
		self.parents = [[(-1, -1) for x in range(size)] for y in range(size)]

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
	return math.sqrt((x1-x0)**2 + (y1-y0)**2)


def AStar(start, goal, ourMap):	
	intData = [[int(x)*200 for x in ourMap.data[i]] for i in range(ourMap.size)]	
	


	openList = PriorityQueue()
	closedList = PriorityQueue()	

	openList.push(start, 0)

	g = f = h = []
	
	g = f = h = [[0 for x in range(ourMap.size)] for y in range(ourMap.size)]
	ourMap.parents[start[0]][start[1]] = (-1, -1)
	
	
	while(not openList.isEmpty()):
		(val, q) = openList.pop()
		(qx, qy) = q
		successor = ourMap.successors(q)
		for s in successor:
			(x,y) = s
			
			
			if s == goal:
				print("YAY")
				ourMap.parents[x][y] = q
				path = trackPath(ourMap, s)
				print(path)
				return path
			g[x][y] = g[qx][qy] + 1
			h[x][y] = heuristic(ourMap, goal, s)
			g[x][y] = g[x][y] + h[x][y]
			if not openList.containBetter(s, f[x][y]):
				if not closedList.containBetter(s, f[x][y]):
					openList.push(s, f[x][y])
					drawOpenBlock(s)
					ourMap.parents[x][y] = q
					
		closedList.push(q, f[qx][qy])
		if q != start:
			drawClosedBlock(q)
	print('No possible path!')
	return []

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

def rectPosition(x, y):
	top = 30 + SCALE * x
	left = 30 + SCALE * y
	return (left, top, SCALE, SCALE)

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

def drawPath(path):	
	if path == []:
		return
	path.reverse()
	path = path[1:-1]
	for n in path:
		pg.event.get()
		(x,y) = n
		color = MOVE
		pg.draw.rect(screen, color, rectPosition(x,y))
		pg.display.flip()
		pg.time.wait(WAIT)
	wait()
	

def main():	
	
	flag = input("Want to create new test? (Y/N) ")

	if flag.upper() == "Y":
		generateRandomTest('input.txt')

	pg.init()
	size, start, goal, ourMapData = readInput('input.txt')
	ourMap = Map(size, ourMapData)
	drawMap(ourMap, start, goal)
	path = AStar(start,goal,ourMap)
	writeOutput('output.txt', ourMap, path)
	drawPath(path)
	pg.quit()
	return 0

if __name__ == '__main__':
	main()	