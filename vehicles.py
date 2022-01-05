from random import randint
import time

class Car:
    tracks=[]
    def __init__(self,i,xi,yi,max_age): #initial for car obj
        self.i=i
        self.x=xi
        self.y=yi
        self.tracks=[]
        self.R=randint(0,255) # random colour code for red
        self.G=randint(0,255) # random colour code for green
        self.B=randint(0,255) # random colour code for blue
        self.done=False # do not pass the check line -> return 0
        self.state='0'
        self.age=0
        self.max_age=max_age
        self.dir=None

    def getRGB(self):  #For the RGB colour
        return (self.R,self.G,self.B)
    def getTracks(self):
        return self.tracks
    
    def getId(self): #For the ID
        return self.i

    def getState(self): #return state of the obj
        return self.state

    def getDir(self): #return direction
        return self.dir

    def getX(self):  #for x coordinate -> return x coordinate
        return self.x

    def getY(self):  #for y coordinate -> return y coordinate
        return self.y

    def updateCoords(self, xn, yn):
        self.age = 0
        self.tracks.append([self.x, self.y])
        self.x = xn
        self.y = yn

    def setDone(self):
        self.done = True # pass the check line -> return 1

    def timedOut(self):
        return self.done

    def going_UP(self, mid_start, mid_end):
        if len(self.tracks)>=2: # check if the obj has been moved
            if self.state=='0': # do not pass the check line
                if self.tracks[-1][1]<mid_end and self.tracks[-2][1]>=mid_end:
                    state='1' # change state to passed the check line
                    self.dir='up' # init its direction
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def going_DOWN(self,mid_start,mid_end): # same as going_UP
        if len(self.tracks)>=2:
            if self.state=='0':
                if self.tracks[-1][1]>mid_start and self.tracks[-2][1]<=mid_start:
                    start='1'
                    self.dir='down'
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def age_one(self):
        self.age+=1
        if self.age>self.max_age:
            self.done=True
        return  True

#Class2

class MultiCar:
    def __init__(self,cars,xi,yi):
        self.cars=cars
        self.x=xi
        self.y=yi
        self.tracks=[]
        self.R=randint(0,255)
        self.G=randint(0,255)
        self.B=randint(0,255)
        self.done=False
