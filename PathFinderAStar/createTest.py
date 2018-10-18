import numpy as np

n=np.random.randint(30);
start = (np.random.randint(n),np.random.randint(n))
goal = (np.random.randint(n),np.random.randint(n))
outputFile = open("input.txt","w")
outputFile.write('%d\n'%n)
outputFile.write('{} {}\n'.format(start[0],start[1]))
outputFile.write('{} {}\n'.format(goal[0],goal[1]))

matrix = np.random.randint(2,size=(n,n))
for i in range(int(n*2/3)**2):
    x = np.random.randint(n)
    y = np.random.randint(n)
    matrix[x][y]=0
matrix[start[0]][start[1]]=0
matrix[goal[0]][goal[1]]=0

for (x,y),value in np.ndenumerate(matrix):
    outputFile.write('%d '%value)
    if(y == (n-1)):
        outputFile.write('\n')

print('Generated random test file!')