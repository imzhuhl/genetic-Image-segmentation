# -*- coding: utf-8 -*-
import numpy as np
from otsu import otsu, fast_ostu

np.random.seed(8)

class GA:
    def __init__(self, image, N = 10):
        """
        genetic algorithm
        :param image: image feature
        :param N: num of population
        :param population: N population
        """
        self.image = image
        self.N = N
        self.population = np.random.randint(0, 256, self.N)
        self.retain_rate = 0.2
        self.random_select_rate = 0.5
        self.mutation_rate = 0.1
        self.length = 8
        
    def evolve(self):
        parents = self.selection()
        self.crossover(parents)
        self.mutation(self.mutation_rate)

    def selection(self):
        graded = [(self.fitness(chromosome), chromosome) for chromosome in self.population]
        graded = [x[1] for x in sorted(graded, reverse=True)]
        # 选出适应性强的染色体
        retain_length = int(len(graded) * self.retain_rate)
        parents = graded[:retain_length]
        # 选出适应性不强，但是幸存的染色体
        for chromosome in graded[retain_length:]:
            if np.random.random() < self.random_select_rate:
                parents.append(chromosome)
        return parents

    def fitness(self, chromosome):
        fitness = fast_ostu(self.image, chromosome)
        return fitness

    def crossover(self, parents):
        children = []
        # 需要繁殖的孩子的量
        target_count = len(self.population) - len(parents)
        while len(children) < target_count:
            male = np.random.randint(0, len(parents)-1)
            female = np.random.randint(0, len(parents)-1)
            if male != female:
                # 随机选取交叉点
                cross_pos = np.random.randint(0, self.length)
                # 生成掩码，方便位操作
                mask = 0
                for i in xrange(cross_pos):
                    mask |= (1 << i) 
                male = parents[male]
                female = parents[female]
                # 孩子将获得父亲在交叉点前的基因和母亲在交叉点后（包括交叉点）的基因
                child = ((male & mask) | (female & ~mask)) & ((1 << self.length) - 1)
                children.append(child)
        # 经过繁殖后，孩子和父母的数量与原始种群数量相等，在这里可以更新种群。
        self.population = parents + children
    
    def mutation(self, rate):
        for i in xrange(len(self.population)):
            if np.random.random() < rate:
                j = np.random.randint(0, self.length - 1)
                self.population[i] ^= 1 << j

    def result(self):
        graded = [(self.fitness(chromosome), chromosome) for chromosome in self.population]
        graded = [x[1] for x in sorted(graded, reverse=True)]
        return graded[0]
