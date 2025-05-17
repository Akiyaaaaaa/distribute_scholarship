import random
from collections import defaultdict


def genetic_algorithm(
    students_data,
    SCHOLARSHIPS,
    nilai_weight,
    pendapatan_weight,
    number_of_generation,
    population_size,
):
    def initialize_population():
        return [
            random.sample(students_data, SCHOLARSHIPS) for _ in range(population_size)
        ]

    def calculate_fitness(individual):
        total_score = sum(s["nilai"] * nilai_weight for s in individual)
        total_income = sum(
            (1 / s["pendapatan"]) * pendapatan_weight for s in individual
        )
        class_count = defaultdict(int)
        for s in individual:
            class_count[s["kelas"]] += 1
        deviation = sum(
            (count - (SCHOLARSHIPS / 3)) ** 2 for count in class_count.values()
        )
        return (total_score * total_income) - deviation

    def selection(populations):
        return sorted(populations, key=calculate_fitness, reverse=True)[:2]

    def crossover(p1, p2):
        merged = {s["id"]: s for s in p1 + p2}
        return random.sample(list(merged.values()), SCHOLARSHIPS)

    def mutation(ind, all_students):
        existing_ids = {s["id"] for s in ind}
        candidates = [s for s in all_students if s["id"] not in existing_ids]
        if candidates:
            idx = random.randint(0, SCHOLARSHIPS - 1)
            ind[idx] = random.choice(candidates)
        return ind

    population = initialize_population()
    log = []

    for _ in range(number_of_generation):
        parents = selection(population)
        child = crossover(parents[0], parents[1])
        child = mutation(child, students_data)
        population.append(child)
        best = max(population, key=calculate_fitness)
        log.append((best, calculate_fitness(best)))

    best_final = max(population, key=calculate_fitness)
    return best_final, log
