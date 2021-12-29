from deap import tools
import random


def varOr(population, toolbox, lambda_, cxpb, mutpb):
    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            #
            #
            # копируем два случайных элемента (всего так сделаем lambda раз)
            ind1, ind2 = list(map(toolbox.clone, random.sample(population, 2)))
            # делаем из них кроссовер
            ind1, ind2 = toolbox.mate(ind1, ind2)
            # удаляем их фитнесы??? А они были? Ну да ладно
            # видимо дальше будут проверяться на valid
            del ind1.fitness.values
            del ind2.fitness.values
            # добавляем к потомству
            offspring.append(ind1)

        # elif op_choice < cxpb + mutpb:  # Apply mutation
        # вот зачем тут плюс? Это специальный прекол?
        # эта ж штука в любом случае (почти) будет больше единицы и всегда прокатит
        elif op_choice < cxpb + mutpb:  # Apply mutation

            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            #
            del ind.fitness.values
            # и этого хрена тоже в потомство
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(toolbox.clone(random.choice(population)))

    return offspring


#     mu - размер популяции
#     lambda_=int(self.pop_size * 0.8)
#       какое то количество, определенное как процент от размера популяции
#       вероятно максимальное количество мутаций кроссоверов (даже при вероятностях =1)
#     cxpb =self.cross_prob вероятность применения кроссовера к индивиду
#     mutpb =self.mut_prob вероятность применения мутации к индивиду

def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):
    #
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    # зачем брать индивиды с неверным фитнесом?
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    # для каждого индивида с неверными фитнесом?
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    # мы же проверяем, так? или что это за херня...

    # вообще еще не было даже основной рализации (если брать первую итерацию..)
    for ind, fit in zip(invalid_ind, fitnesses):
        # добавляем значение фитнеса индивиду
        #
        #
        ind.fitness.values = fit

    #
    if halloffame is not None:
        #
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    # есть ген=0, соответственно код выше только для первой итерации
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process

    for gen in range(1, ngen + 1):

        # Vary the population
        if halloffame is not None:
            for ind in halloffame:
                population.append(toolbox.clone(ind))
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        # турнир или рулетка ...
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

