from __future__ import division
from numba import cuda
import numpy as np
import math
import pygame
import colorsys
import sys

nparticles = 3000
pygame.init()
screensize = (1200, 1200)
screen = pygame.display.set_mode(screensize)


@cuda.jit
def grav_2D(velArr, posArr):
    axis, particle = cuda.grid(2)
    if axis == 0 and particle < posArr.shape[0]:
        for part2 in range(particle, posArr.shape[0]):
            dirX = posArr[part2, 0] - posArr[particle, 0]
            dirY = posArr[part2, 1] - posArr[particle, 1]
            magSq = dirX**2 + dirY**2 + 5
            xgrav = (dirX / math.sqrt(magSq)) / magSq
            ygrav = (dirY / math.sqrt(magSq)) / magSq
            velArr[particle, 0] = velArr[particle, 0] + xgrav
            velArr[particle, 1] = velArr[particle, 1] + ygrav
            velArr[part2, 0] = velArr[part2, 0] - xgrav
            velArr[part2, 1] = velArr[part2, 1] - ygrav
            

particlePos = np.zeros((nparticles, 2), dtype=np.float64)
particleVels = np.zeros((nparticles, 2), dtype=np.float64)

particlePos = np.random.rand(nparticles, 2) * 1200
print(particlePos.shape)

particlePos = particlePos.astype(np.float64)

threadsperblock = (1, 1)

blockspergrid_x = math.ceil(particlePos.shape[1] / threadsperblock[0])
blockspergrid_y = math.ceil(particlePos.shape[0] / threadsperblock[1])

blockspergrid = (blockspergrid_x, blockspergrid_y)

def hsv2rgb(h,s,v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

clock = pygame.time.Clock()

while True:
    s = pygame.Surface(screensize, pygame.SRCALPHA)
    s.fill((0, 0, 0, 5))
    screen.blit(s, (0,0)) 
        
    for particle in range(len(particlePos)):

        pygame.draw.circle(screen, (0, 120, 255), (particlePos[particle, 0], particlePos[particle, 1]), 5)

    grav_2D[blockspergrid, threadsperblock](particleVels, particlePos)

    '''
    for particle in range(len(particlePos)):
        for part2 in range(particle, particlePos.shape[0]):
            dirX = particlePos[part2, 0] - particlePos[particle, 0]
            dirY = particlePos[part2, 1] - particlePos[particle, 1]
            magSq = dirX**2 + dirY**2 + 5
            xgrav = (dirX / math.sqrt(magSq)) / magSq
            ygrav = (dirY / math.sqrt(magSq)) / magSq
            particleVels[particle, 0] = particleVels[particle, 0] + xgrav
            particleVels[particle, 1] = particleVels[particle, 1] + ygrav
            particleVels[part2, 0] = particleVels[part2, 0] - xgrav
            particleVels[part2, 1] = particleVels[part2, 1] - ygrav
    '''
    

    if pygame.mouse.get_pressed()[0]:
        mX, mY = pygame.mouse.get_pos()
        for particle in range(len(particlePos)):
            dirX = mX - particlePos[particle, 0]
            dirY = mY - particlePos[particle, 1]
            magSq = dirX**2 + dirY**2 + 5
            xgrav = 100 * (dirX / math.sqrt(magSq)) / magSq
            ygrav = 100 * (dirY / math.sqrt(magSq)) / magSq
            particleVels[particle, 0] = particleVels[particle, 0] + xgrav
            particleVels[particle, 1] = particleVels[particle, 1] + ygrav

    particlePos = particlePos + particleVels

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    pygame.display.update()

    clock.tick()
    print(clock.get_fps())


