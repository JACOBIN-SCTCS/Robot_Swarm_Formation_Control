
import math
import pygame
from pygame.locals import *
import sys
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

pygame.init()
pygame.font.init()
import numpy as np


NUM_ROBOTS = 4
WINDOW_WIDTH = 700
WINDOW_HEIGHT = 700
ROBOT_RADIUS = 10
SAFE_REGION = ROBOT_RADIUS+80
K_p = 0.004
ALPHA = 4
BETA = 0.03

square_template = np.array([[1,-1],[-1,-1],[1,1],[-1,1]])

b_matrix = -K_p*80*square_template
laplacian = np.array([[3,-1,-1,-1],[-1,3,-1,-1],[-1,-1,3,-1],[-1,-1,-1,3]])
SAFE_REGION_EXPONENTIAL = np.exp(-BETA*SAFE_REGION) * np.ones((NUM_ROBOTS,NUM_ROBOTS))
#print(SAFE_REGION_EXPONENTIAL)

positions = np.random.randint(ROBOT_RADIUS,WINDOW_WIDTH-ROBOT_RADIUS,(NUM_ROBOTS,2))
screen = pygame.display.set_mode((WINDOW_WIDTH,WINDOW_HEIGHT))
screen.fill((255,255,255))

dt = 0.2

desired_position = np.random.randint(ROBOT_RADIUS,WINDOW_WIDTH-ROBOT_RADIUS,(1,2))
old_positions = np.copy(positions)

while True:
    
    pygame.time.delay(10)

    #displacement = (velocity[0]*dt,velocity[1]*dt)
    #position = (position[0] + displacement[0],position[1]+displacement[1])
    screen.fill((255,255,255))
    new_velocity = -K_p * np.matmul(laplacian,positions) + b_matrix
      
    pairwise_distances = squareform( pdist(positions))
    SAFE_REGION_MATRIX = SAFE_REGION*np.eye(NUM_ROBOTS)
    pairwise_distances = pairwise_distances + SAFE_REGION_MATRIX
    pairwise_distances = np.clip(pairwise_distances,0,SAFE_REGION)
    pairwise_distances = np.exp(-BETA * pairwise_distances)
    difference = ALPHA * (pairwise_distances - SAFE_REGION_EXPONENTIAL)


    pairwise_repulsive_velocities = np.zeros((NUM_ROBOTS,NUM_ROBOTS,2))
    for i in range(NUM_ROBOTS):
        for j in range(NUM_ROBOTS):
            difference_vector = positions[i] - positions[j]
       
            normalized_distance = np.linalg.norm(difference_vector)
      

            if(normalized_distance>0):
                difference_vector = difference_vector/normalized_distance

            pairwise_repulsive_velocities[i,j,:] = difference[j][i] *difference_vector
  
    
    repulsive_velocity = np.sum(pairwise_repulsive_velocities,axis=1)
    displacement = (new_velocity+repulsive_velocity)*dt
    positions = positions + displacement


    mean_position = np.mean(positions,axis=0)
    

    velocity_direction = (new_velocity+repulsive_velocity) 
    norm = np.linalg.norm(velocity_direction,axis=1)
    velocity_vector = 15 *velocity_direction / norm[:,None]
    
    for i in range(len(positions)):
        pygame.draw.circle(screen,(255,0,0),positions[i].tolist(),ROBOT_RADIUS,width=3)
        pygame.draw.line(screen,(0,255,0),positions[i],positions[i] + velocity_vector[i],3)
    
    pygame.draw.circle(screen,(0,0,0),desired_position[0].tolist(),4)
    pygame.display.update()


    for e in pygame.event.get():

        if e.type==pygame.MOUSEBUTTONUP:
            pos = pygame.mouse.get_pos()
            desired_position = np.array([[pos[0],pos[1]]])
        if e.type==QUIT:
            pygame.quit()
            sys.exit()
