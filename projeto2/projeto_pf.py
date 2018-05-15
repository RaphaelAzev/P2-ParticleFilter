#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esta classe deve conter todas as suas implementações relevantes para seu filtro de partículas
"""

from pf import Particle, create_particles, draw_random_sample
import numpy as np
import inspercles # necessário para o a função nb_lidar que simula o laser
import math
import scipy.stats as stats


largura = 775 # largura do mapa
altura = 748  # altura do mapa

# Robo
robot = Particle(largura/2, altura/2, math.pi/4, 1.0)

# Nuvem de particulas
particulas = []

num_particulas = 500


# Os angulos em que o robo simulado vai ter sensores
angles = np.linspace(0.0, 2*math.pi, num=8, endpoint=False)

# Lista mais longa
movimentos_longos = [[-10, -10, 0], [-10, 10, 0], [-10,0,0], [-10, 0, 0],
              [0,0,math.pi/12.0], [0, 0, math.pi/12.0], [0, 0, math.pi/12],[0,0,-math.pi/4],
              [-5, 0, 0],[-5,0,0], [-5,0,0], [-10,0,0],[-10,0,0], [-10,0,0],[-10,0,0],[-10,0,0],[-15,0,0],
              [0,0,-math.pi/4],[0, 10, 0], [0,10,0], [0, 10, 0], [0,10,0], [0,0,math.pi/8], [0,10,0], [0,10,0], 
              [0,10,0], [0,10,0], [0,10,0],[0,10,0],
              [0,0,-math.radians(90)],
              [math.cos(math.pi/3)*10, math.sin(math.pi/3),0],[math.cos(math.pi/3)*10, math.sin(math.pi/3),0],[math.cos(math.pi/3)*10, math.sin(math.pi/3),0],
              [math.cos(math.pi/3)*10, math.sin(math.pi/3),0]]

# Lista curta
movimentos_curtos = [[-10, -10, 0], [-10, 10, 0], [-10,0,0], [-10, 0, 0]]

movimentos_relativos = [[0, -math.pi/3],[10, 0],[10, 0], [10, 0], [10, 0],[15, 0],[15, 0],[15, 0],[0, -math.pi/2],[10, 0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [0, -math.pi/2], 
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [0, -math.pi/2], 
                       [10,0], [0, -math.pi/4], [10,0], [10,0], [10,0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0]]



movimentos = movimentos_relativos



def cria_particulas(minx=0, miny=0, maxx=largura, maxy=altura, n_particulas=num_particulas):
    """
        Cria uma lista de partículas distribuídas de forma uniforme entre minx, miny, maxx e maxy
    """
    for i in range(n_particulas):
      xp = np.random.uniform(minx, maxx, n_particulas) #posição x da partícula, aleatório entre 0 e 775 
      yp = np.random.uniform(miny, maxy, n_particulas) #posição y da partícula, aleatório entre 0 e 748
      ang = np.random.uniform(0, 2*math.pi) #angulo da partícula (direção que está apontada) aleatório entre 0 e 360º
      part = Particle(xp, yp, ang, w=1.0)
      particulas.append(part) #appendando as particulas criadas na lista declarada no início
    return create_particles(robot.pose(),largura/2,altura/2,math.pi,n_particulas)
    #return particulas
      
def move_particulas(particulas, movimento):
    """
        Recebe um movimento na forma [deslocamento, theta]  e o aplica a todas as partículas
        Assumindo um desvio padrão para cada um dos valores
        Esta função não precisa devolver nada, e sim alterar as partículas recebidas.
        
        Sugestão: aplicar move_relative(movimento) a cada partícula
        
        Você não precisa mover o robô. O código fornecido pelos professores fará isso
        
    """
    for part in particulas:
      movimento[0] = stats.norm.pdf(np.random.uniform(movimento[0]-0.2,movimento[0]+0.2),movimento[0],0.2)
      movimento[1] = stats.norm.pdf(np.random.uniform(movimento[1]-0.15,movimento[1]+0.15),movimento[1],0.15) 
      #or
      #movimento[0] = np.random.uniform(movimento[0]-0.2,movimento[0]+0.2)
      #movimento[1] = np.random.uniform(movimento[1]-0.15,movimento[1]+0.15)
      part.move_relative(movimento)

    return particulas
    
def leituras_laser_evidencias(robot, particulas):
    """
        Realiza leituras simuladas do laser para o robo e as particulas
        Depois incorpora a evidência calculando
        P(H|D) para todas as particulas
        Lembre-se de que a formula $P(z_t | x_t) = \alpha \prod_{j}^M{e^{\frac{-(z_j - \hat{z_j})}{2\sigma^2}}}$ 
        responde somente P(D|Hi), em que H é a hi
        
        Esta função não precisa retornar nada, mas as partículas precisa ter o seu w recalculado. 
        
        Você vai precisar calcular para o robo
        
    """
    sumfor1part = 0
    leitura_robo = inspercles.nb_lidar(robot, angles)
    dicParts = []
    probPDH = []   ## alpha = 1/soma de todas as probabilidades
    #print(inspercles.nb_lidar(particulas[0],angles))
    for part in particulas:
      le_part = inspercles.nb_lidar(part,angles)
      dicParts.append(le_part)
      #print(le_part)

    for i in range(len(particulas)):
      for f in range(8):
        sumfor1part = sumfor1part + stats.norm.pdf(dicParts[i][angles[f]],leitura_robo[angles[f]],scale = 7)  #soma das 8 direções dos lasers
      probPDH.append(sumfor1part)
      sumfor1part = 0  

    alpha = 1/sum(probPDH)                 # alpha * pdh * ph = Phd
    #print(probPDH)
    #print(sum(probPDH))
    for j in range(len(probPDH)):
      part.w = alpha*probPDH[j]*part.w
    
    # Voce vai precisar calcular a leitura para cada particula usando inspercles.nb_lidar e depois atualizar as probabilidades


    
    
def reamostrar(particulas, n_particulas = num_particulas):
    """
        Reamostra as partículas devolvendo novas particulas sorteadas
        de acordo com a probabilidade e deslocadas de acordo com uma variação normal    
        
        O notebook como_sortear tem dicas que podem ser úteis
        
        Depois de reamostradas todas as partículas precisam novamente ser deixadas com probabilidade igual
        
        Use 1/n ou 1, não importa desde que seja a mesma
    """
    partweight = [part.w for part in particulas]

    partsAmostradas = draw_random_sample(particulas, partweight, num_particulas)
    for part in partsAmostradas:
      part.w = 1
    return particulas


    







