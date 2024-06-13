import heapq
import math
import pandas as pd
import geopandas as gpd
import re
from geopy.distance import geodesic
from math import ceil
import random

# Define the column names
column_names = ['countyname', 'neighborname']

# Read the CSV file without headers, specifying the column names
df = pd.read_csv(r'C:\שנה ג\יסודות בינה\תרגילי בית\תרגיל 1\adjacency.csv', names=column_names)

# Initialize an empty dictionary to store the neighbors of each county
neighbors_dict = {}

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    county = row['countyname']
    neighbor = row['neighborname']
    
    # Ignore if county is its own neighbor
    if neighbor != county:
        # Append the neighbor to the county's list of neighbors
        neighbors_dict.setdefault(county, []).append(neighbor)


# Load data from 'county_centers_coordinates' file
df = pd.read_csv('county_centers_coordinates.csv')

# Create 'county_centers' dictionary from the loaded data
county_centers = {}
for index, row in df.iterrows():
    county_name = row['County']
    county_coords = (row['Latitude'], row['Longitude'])
    county_centers[county_name] = county_coords



def heuristic(start_location, goal_location):
    # Get coordinates for the start and goal locations from the county_centers dictionary
    start_coords = county_centers.get(start_location)
    goal_coords = county_centers.get(goal_location)

    if not start_coords or not goal_coords:
        return float('inf')  # Return a high number to indicate infeasible path

    # Calculate the geodesic distance between the two coordinates
    start_lat, start_long = start_coords
    goal_lat, goal_long = goal_coords

    distance = geodesic((start_lat, start_long), (goal_lat, goal_long)).kilometers
    estimated_distance = ceil(distance / 100)
    return estimated_distance

def heuristic2(start_location, goal_location):
    # Get coordinates for the start and goal locations from the county_centers dictionary
    start_coords = county_centers.get(start_location)
    goal_coords = county_centers.get(goal_location)

    if not start_coords or not goal_coords:
        return float('inf')  # Return a high number to indicate infeasible path

    # Calculate the geodesic distance between the two coordinates
    start_lat, start_long = start_coords
    goal_lat, goal_long = goal_coords

    distance = geodesic((start_lat, start_long), (goal_lat, goal_long)).kilometers
    estimated_distance = distance / 100
    return estimated_distance

    

def A_Star_Algo_1T1(starting_location, goal_location):
    # open_list entries are tuples of the form (f, g, h, current_location, path)
    open_list = [(heuristic(starting_location, goal_location), 0, heuristic(starting_location, goal_location), starting_location, [])]
    closed_set = set()
    open_set = set([starting_location])

    while open_list:
        # Prioritize nodes based on their f value
        current = heapq.heappop(open_list)
        current_f, current_g, current_h, current_loc, current_path = current

        if current_loc == goal_location:
            return current_path + [current_loc]  # Return the path if the goal is reached

        closed_set.add(current_loc)
        open_set.discard(current_loc)

        for neighbor in neighbors_dict.get(current_loc, []):
            if neighbor not in closed_set:
                g = current_g + 1  # Increase g by 1 for each step
                h = heuristic(neighbor, goal_location)
                f = g + h
                new_path = current_path + [current_loc]

                already_in_open_set = any(n[3] == neighbor for n in open_list)
                if not already_in_open_set or any(n[3] == neighbor and n[0] > f for n in open_list):
                    heapq.heappush(open_list, (f, g, h, neighbor, new_path))
                    open_set.add(neighbor)

    return None  # No path found


def hill_climbing(start_location, goal_location, max_restarts=5):
    def attempt_climb(current_location):
        path = [current_location]

        while current_location != goal_location:
            neighbors = neighbors_dict.get(current_location, [])
            if not neighbors:
                return None  # No path found

            # Find the neighbor with the lowest heuristic value
            next_location = min(neighbors, key=lambda loc: heuristic2(loc, goal_location))
            if heuristic2(next_location, goal_location) >= heuristic2(current_location, goal_location):
                return None  # No path found

            current_location = next_location
            path.append(current_location)

        return path

    result = attempt_climb(start_location)
    if result:
        return result
    
    tried_neighbors = set()

    for _ in range(max_restarts - 1):

        # Refresh the list of neighbors, skipping already tried ones
        potential_neighbors = [n for n in neighbors_dict.get(start_location, []) if n not in tried_neighbors]

        if potential_neighbors:
            start_neighbore = random.choice(potential_neighbors)
            tried_neighbors.add(start_neighbore)
        else:
            break  # No more unique neighbors to try
    
        result = attempt_climb(start_neighbore)
        if result:
            result = [start_location] + result
            return result  # Path found

    return None  # No path found after max_restarts attempts




def simulated_annealing(start_state, goal_state, max_iterations=10000, initial_temp=100, alpha=0.94):
    current_state = start_state
    path = [current_state]
    visited = {current_state: 0}
    t = initial_temp  # Initial temperature

    for iteration in range(max_iterations):
        if current_state == goal_state and detail_output_choose ==False:
            return remove_cycles(path),None
        if current_state == goal_state and detail_output_choose ==True:
            return remove_cycles(path),[start_state, first_county_consider, first_probability]


        neighbors = neighbors_dict.get(current_state, [])
        if not neighbors:
            break  # No possible moves, stop this run

        next_state = random.choice(neighbors)

        current_cost = heuristic2(current_state, goal_state)
        next_cost = heuristic2(next_state, goal_state)

        # Move to the next state based on the acceptance probability
        delta_cost = next_cost - current_cost
        if t == 0:  # Avoid division by zero
            acceptance_probability = 0
        else:
            # Cap the exponent to avoid overflow
            try:
                acceptance_probability = min(1, math.exp(min(-delta_cost / t, 700)))
            except OverflowError:
                acceptance_probability = 0

        if detail_output_choose == True and iteration == 0:  ## for True case in the main. saving the first iteration decision
            first_county_consider = next_state
            first_probability = acceptance_probability   

        if delta_cost < 0 or random.random() < acceptance_probability:
            current_state = next_state
            path.append(current_state)
            visited[current_state] = len(path) - 1  # Update the latest position

        # Adaptive cooling: Reduce temperature more gradually initially, then rapidly
        t *= alpha  # Slower cooling schedule
    
    if detail_output_choose == False:
        if current_state == goal_state:
             return remove_cycles(path),None   
        else:
            return None,None
    
    if current_state == goal_state:
         return remove_cycles(path), [start_state, first_county_consider, first_probability]
    else:
        return None,None


def remove_cycles(path):
    seen = set()
    cleaned_path = []
    for node in path:
        if node not in seen:
            cleaned_path.append(node)
            seen.add(node)
        else:
            index = cleaned_path.index(node)
            cleaned_path = cleaned_path[:index+1]  # Trim the path from the repeated location
            seen.clear()
            for n in cleaned_path:
                seen.add(n)
    return cleaned_path


def k_beam(start_location, goal_location, k=3):
    visited = set()  # Set to store visited nodes
    beam = [(0, start_location, [])]  # Initialize the beam with the starting location
    while beam:
        new_beam = []  # Temporary beam for the next iteration
        for beam_item in beam:
            _, current_location, path = beam_item
            visited.add(current_location)
            if current_location == goal_location:
                return path + [current_location]
            neighbors = [(heuristic2(neighbor, goal_location), neighbor) for neighbor in neighbors_dict.get(current_location, []) if neighbor not in visited]
            neighbors.sort()  # Sort the neighbors based on heuristic value
            new_beam += [(h, neighbor, path + [current_location]) for h, neighbor in neighbors]
            new_beam.sort()  # Sort the new beam
            beam = new_beam[:k]  # Take the best k paths for the next iteration
    return None  # No path found

#genetic algoritem
##########################
class Individual:
    def __init__(self, path, goal_location):
        self.path = path
        self.goal_location = goal_location
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        # Fitness is defined as the path length plus the heuristic estimate
        if not self.path or not is_valid_path(self.path, self.goal_location):
            return float('inf')
        heuristic_value = heuristic2(self.path[-1], self.goal_location)
        path_length = len(self.path)
        return path_length + heuristic_value

def generate_initial_population(start_location, goal_location, population_size=10):
    population = []
    counter = 0
    max_steps = 100  # You can adjust this to limit the length of random paths
    
    while len(population) < population_size:
        if counter == 10000:
            return None
        path = generate_random_path(start_location, max_steps)
        if path:
            population.append(Individual(path, goal_location))
        counter += 1
    
    return population

def generate_random_path(start_node, max_steps=100):
    path = [start_node]
    current_location = start_node
    visited = set(path)
    for _ in range(max_steps):  # Arbitrary limit to avoid infinite loops
        neighbors = [neighbor for neighbor in neighbors_dict.get(current_location, []) if neighbor not in visited]
        if not neighbors:
            break
        current_location = random.choice(neighbors)
        path.append(current_location)
        visited.add(current_location)
    return compress_path(path)  # Compress path to remove cycles

def compress_path(path):
    seen = {}
    compressed = []
    for index, node in enumerate(path):
        if node in seen:
            # If the node is seen, remove the cycle
            cycle_start = seen[node]
            compressed = compressed[:cycle_start]
        seen[node] = len(compressed)
        compressed.append(node)
    return compressed

def calculate_probabilities(population):
    fitness_scores = [(1.0 / ind.fitness if ind.fitness != float('inf') else 0) for ind in population]
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores] if total_fitness > 0 else [1 / len(population)] * len(population)
    return probabilities

def probabilistic_selection(population):
    probabilities = calculate_probabilities(population)
    selected_index = random.choices(range(len(population)), weights=probabilities, k=1)[0]
    return population[selected_index]

def selection(population, num_selected=5):
    return [probabilistic_selection(population) for _ in range(num_selected)]

def crossover(parent1, parent2):
    min_len_path = min(len(parent1.path), len(parent2.path))
    
    # Ensure valid range for random selection of crossover point
    if min_len_path > 1:
        start = random.randint(1, min_len_path - 1)
        child_path = parent1.path[:start]
        next_node = parent2.path[start]
        
        # Avoid starting an already duplicated node
        while next_node in child_path and len(parent2.path) > start + 1:
            start += 1
            next_node = parent2.path[start]
        
        child_path.extend(parent2.path[start:])
        child_path = compress_path(child_path)
        
        if len(child_path) != len(set(child_path)):  # Check for duplication
            return Individual(parent1.path, parent1.goal_location)  # Fallback to one of the parents
        return Individual(child_path, parent1.goal_location)
    
    # Fall back to one parent if the path length is insufficient
    return Individual(parent1.path, parent1.goal_location) if random.random() < 0.5 else Individual(parent2.path, parent1.goal_location)

def mutate(individual, max_steps=100):
    if len(individual.path) < 2:
        return
    mutation_point = random.randint(0, len(individual.path) - 2)
    current_location = individual.path[mutation_point]
    if current_location in neighbors_dict:
        next_location = random.choice(neighbors_dict[current_location])
        new_path = generate_random_path(next_location, max_steps)  # Use max_steps
        if new_path:
            new_segment = individual.path[:mutation_point + 1] + new_path  # Keep start, mutate rest
            individual.path = compress_path(new_segment)
            individual.fitness = individual.calculate_fitness()

# Check if path has elements in the right order without any cycles and ends with the goal location
def is_valid_path(path, goal_location):
    if path[-1] != goal_location:
        return False
    for i in range(len(path) - 1):
        if path[i + 1] not in neighbors_dict.get(path[i], []):
            return False
    return True

def genetic(start_location, goal_location, population_size=10, generations=7000, mutation_rate=0.1):
    population = generate_initial_population(start_location, goal_location, population_size)
    first_actions_Genetic = None

    if population == None:
        return None,None

    for generation in range(generations):
        selected_individuals = selection(population, num_selected=5)
        next_generation = selected_individuals[:]

        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(selected_individuals, 2)
            child = crossover(parent1, parent2)
            
            if random.random() < mutation_rate:
                mutate(child, max_steps=100)  # Pass max_steps here
            
            next_generation.append(child)

        population = next_generation


        # Save the paths of the population after the first iteration
        if generation == 0 and detail_output_choose==True:
            first_actions_Genetic = [ind.path for ind in population]   


        # Check for the best solution in the current population
        best_individual = min(population, key=lambda x: x.fitness)
        if is_valid_path(best_individual.path, goal_location):
            return best_individual.path, first_actions_Genetic

    return None,None
###################################


def shortest1Path_Many_T_Many(start_counties, goal_counties):
    shortest_path = None
    shortest_path_length = float('inf')

    for start_county in start_counties:
        start_color = start_county.split(',')[0].strip()
        
        for goal_county in goal_counties:
            goal_color = goal_county.split(',')[0].strip()
            
            if start_color == goal_color:
                startParts = start_county.split(', ')
                Start_county_state = ', '.join(startParts[1:])
                goalParts = goal_county.split(', ')
                goal_county_state = ', '.join(goalParts[1:])

                if search_method_choose == 1:
                    path = A_Star_Algo_1T1(Start_county_state, goal_county_state)

                if search_method_choose == 2:
                    path = hill_climbing(Start_county_state, goal_county_state)

                if search_method_choose == 3:
                    path, first_actions = simulated_annealing(Start_county_state, goal_county_state)

                
                if search_method_choose == 4:
                    path = k_beam(Start_county_state, goal_county_state)

                if search_method_choose ==5:
                    path, trueGen  = genetic(Start_county_state, goal_county_state)   
    

                if path and len(path) < shortest_path_length:

                    if search_method_choose == 3 and detail_output_choose == True: ##for true in main problem 3
                        for i in range (len(first_iteration_actions)):
                           if first_actions[0] == first_iteration_actions[i][0]:
                               first_iteration_actions.pop(i)
                               break      
                        first_iteration_actions.append(first_actions) 


                    if search_method_choose == 5 and detail_output_choose == True: ##for true in main problem 3
                        for i in range (len(trueCaseGenetic)):
                           if trueGen[0][0] == trueCaseGenetic[i][0][0]:
                               trueCaseGenetic.pop(i)
                               break      
                        trueCaseGenetic.append(trueGen)
                        

                    color = start_color   
                    shortest_path = path
                    shortest_path_length = len(path)

    # Add color to each node in the shortest_path if it is not None
    if shortest_path:
        shortest_path = [f"{color}, {county}" for county in shortest_path]            

    return shortest_path

def Pach_Many_T_Many(start_counties, goal_counties):
    all_paths = []
    initial_start_counties = start_counties.copy()
    start_counties_for_workon = start_counties.copy()
    goal_counties_for_workon = goal_counties.copy()
    initial_length = len(start_counties)

    for i in range(initial_length):
        # Finding the shortest path using shortest1Path_Many_T_Many
        shortest_path = shortest1Path_Many_T_Many(start_counties_for_workon, goal_counties_for_workon)
        if shortest_path:
            all_paths.append(shortest_path)
            #removing the start and finish that were matched
            start_counties_for_workon.remove(shortest_path[0])
            goal_counties_for_workon.remove(shortest_path[-1])
        else:
            all_paths.append(['No path found'])
        

    # Sort all_paths according to the order in initial_start_counties
    sorted_paths = []
    for start in initial_start_counties:
        found_path = False
        for path in all_paths:
            if path[0] == start:
                sorted_paths.append(path)
                found_path = True
                break
        if not found_path:
            # Add 'No path found' if no path matches the start county
            sorted_paths.append(['No path found'])
        
 # Extract county names and add the first letter of the color in parentheses
    for path_index in range(len(sorted_paths)):
        for county_index in range(len(sorted_paths[path_index])):
            county = sorted_paths[path_index][county_index]
            if county != 'No path found':
                color = county.split(', ')[0]
                state = county.split(', ')[-1]
                county_name = ', '.join(county.split(', ')[1:-1])
                color_letter = color[0].upper()
                sorted_paths[path_index][county_index] = f"{county_name}, {state} ({color_letter})"
            else:
                sorted_paths[path_index][county_index] = county

    return sorted_paths


def transpose_and_fill(final_paths):
    # Find the maximum length of the paths
    max_length = max(len(path) for path in final_paths)

    # Extend each path to the maximum length by repeating the last element
    extended_paths = [
        path + [path[-1]] * (max_length - len(path))
        for path in final_paths
    ]

    # Transpose the extended matrix
    transposed_paths = list(map(list, zip(*extended_paths)))

    return transposed_paths


def format_paths_with_heuristics(final_paths): ## for A* TRUE
    transposed_paths = transpose_and_fill(final_paths)
    heuristic_row = []

    for col in range(len(transposed_paths[0])):
        # Check if 'No path found' is in the column
        if 'No path found' in [transposed_paths[row][col] for row in range(len(transposed_paths))]:
            heuristic_row.append('N/A')
        else:
            # Process the location to remove any region part
            location_parts = transposed_paths[0][col].split(', ')[:2]
            location = re.sub(r'\s*\([\w\s]*\)', '', ', '.join(location_parts))
            
            # Process the goal location to remove any region part
            goal_location_parts = transposed_paths[-1][col].split(', ')[:2]
            goal_location = re.sub(r'\s*\([\w\s]*\)', '', ', '.join(goal_location_parts))

            heuristic_value = heuristic(location, goal_location)
            heuristic_row.append(f"{heuristic_value}")

    # Insert the heuristic row after the second row
    result = []
    for row_index in range(len(transposed_paths)):
        result.append(f"{{{' ; '.join(transposed_paths[row_index])}}}")
        if row_index == 1:
            result.append(f"Heuristic: {{{' ; '.join(heuristic_row)}}}")

    return result


def format_paths_with_custom_data(final_paths, first_iteration_actions): ## For sumlate aneling True print

    for row1 in range (len(final_paths)):
        for row2 in range (len(first_iteration_actions)):

            final_location_name = final_paths[row1][0]
            final_location_name = re.sub(r'\s*\([^)]*\)', '', final_location_name).strip()


            first_iteration_location_name = first_iteration_actions[row2][0]

                    # Compare the extracted names
            if final_location_name == first_iteration_location_name:
                concatenated_string = f"{first_iteration_actions[row2][1]}:{first_iteration_actions[row2][2]}"
                final_paths[row1].insert(2, concatenated_string)  # Insert at index 1 in the row

    return final_paths


def best_locations_for_starting_locations_k_beam(final_Paths): #### For kbeam True print. checking the heuristic
    best_locations = []
    for i in range (len(final_Paths)):
        startLoc = final_Paths[i][0]
        startLoc = re.sub(r'\s*\([^)]*\)', '', startLoc).strip()
        finishLock = final_Paths[i][-1]
        finishLock = re.sub(r'\s*\([^)]*\)', '', finishLock).strip()
        
        if startLoc != "No path found":
            neighbors = [(round(heuristic2(neighbor, finishLock), 2), neighbor) for neighbor in neighbors_dict.get(startLoc, [])]
            neighbors.sort()  # Sort the neighbors based on heuristic value
            best_locations.append({startLoc: neighbors[:3]})
        else:
             best_locations.append("No path found")       

    return best_locations


def process_final_paths(final_Paths, best_locations_dict): #### For kbeam True print. adding the 3 row
    new_paths = []
    
    for row in final_Paths:
        if row[0] != "No path found":
            location_key = row[0]
            simplified_location_key = re.sub(r'\([^)]*\)', '', location_key).strip()
            
            if simplified_location_key in best_locations_dict:
                key_values = best_locations_dict[simplified_location_key]
                
                values_as_string = ', '.join([f"{value[0]}, '{value[1]}'" for value in key_values])
                
                new_row = [row[0], row[1]] + [values_as_string] + row[2:]
            else:
                new_row = row
        else:
            new_row = ["No path found"] + row[1:]
            
        new_paths.append(new_row)
    return new_paths


def process_final_paths_Genetic(final_Paths, trueCaseGenetic): #### For genetic true case. second population
    for i, row in enumerate(final_Paths):
        if row[0] != "No path found":
            location_key = row[0]
            simplified_location_key = re.sub(r'\([^)]*\)', '', location_key).strip()
            
            # Find the corresponding index in trueCaseGenetic
            for j in range(len(trueCaseGenetic)):
                if simplified_location_key == trueCaseGenetic[j][0][0]:
                    # Build the chain string for trueCaseGenetic[j]
                    chains = []
                    for k in range(10):  # There are always 10 lists within each trueCaseGenetic list
                        chain = f"{k}: " + ", ".join(trueCaseGenetic[j][k])
                        chains.append(chain)
                    # Concatenate all chains into one string
                    final_chain_string = " ".join(chains)
                    # Append this chain string to the row
                    final_Paths[i].insert(2, final_chain_string)
                    break  # Stop searching once the correct list is found

    return final_Paths



def find_path(starting_locations, goal_locations, search_method, detail_output):
     global search_method_choose
     global detail_output_choose
     
          
     search_method_choose = search_method
     detail_output_choose = detail_output
     

     if search_method ==3 and detail_output==True:
          global first_iteration_actions   # List to store the first iteration actions      
          first_iteration_actions = []

     if search_method ==5 and detail_output==True:
        global trueCaseGenetic # List to store the first iteration actions 
        trueCaseGenetic = []

     final_Paths = Pach_Many_T_Many(starting_locations, goal_locations)

     if search_method==1 and detail_output==False:   ##A*
         if final_Paths:
             transposed_paths = transpose_and_fill(final_Paths)
             for row in transposed_paths:
                 formatted_row = " ; ".join(row)
                 print(f"{{{formatted_row}}}")
         else:
            print("No path found")           

     if search_method==1 and detail_output==True: ##A*
         if final_Paths:
            formatted_paths = format_paths_with_heuristics(final_Paths)
            for line in formatted_paths:
                print(line)
         else:
           print("No path found")    

     if search_method==2:  ##hill climbing
         if final_Paths:
             transposed_paths = transpose_and_fill(final_Paths)
             for row in transposed_paths:
                 formatted_row = " ; ".join(row)
                 print(f"{{{formatted_row}}}")
         else:
            print("No path found")                          

     if search_method==3 and detail_output==False:  ##simulated annealing
         if final_Paths:
             transposed_paths = transpose_and_fill(final_Paths)
             for row in transposed_paths:
                 formatted_row = " ; ".join(row)
                 print(f"{{{formatted_row}}}")
         else:
            print("No path found")            

     if search_method==3 and detail_output==True:   ##simulated annealing
        if final_Paths:
            formatted_paths = format_paths_with_custom_data(final_Paths, first_iteration_actions)
            transposed_paths = transpose_and_fill(formatted_paths)
            for row in transposed_paths:
                 formatted_row = " ; ".join(row)
                 print(f"{{{formatted_row}}}")   
        else:
            print("No path found")

     if search_method==4 and detail_output==False: ##kbeam
         if final_Paths:
             transposed_paths = transpose_and_fill(final_Paths)
             for row in transposed_paths:
                 formatted_row = " ; ".join(row)
                 print(f"{{{formatted_row}}}")
         else:
            print("No path found")

     if search_method==4 and detail_output==True:  ##kbeam
         if final_Paths:
            best_locations = best_locations_for_starting_locations_k_beam(final_Paths)
            best_locations_dict = {list(location.keys())[0]: list(location.values())[0] for location in best_locations if isinstance(location, dict)}
            final_Paths = process_final_paths(final_Paths,best_locations_dict)
            transposed_paths = transpose_and_fill(final_Paths)
            for row in transposed_paths:
                formatted_row = " ; ".join(row)
                print(f"{{{formatted_row}}}")
         else:
            print("No path found") 

     if search_method==5 and detail_output==False:  ##genetic
         if final_Paths:
             transposed_paths = transpose_and_fill(final_Paths)
             for row in transposed_paths:
                 formatted_row = " ; ".join(row)
                 print(f"{{{formatted_row}}}")
         else:
            print("No path found")                

     if search_method==5 and detail_output==True:  ##genetic
         if final_Paths:
             finalP = process_final_paths_Genetic(final_Paths, trueCaseGenetic) 
             transposed_paths = transpose_and_fill(finalP)
             for row in transposed_paths:
                 formatted_row = " ; ".join(row)
                 print(f"{{{formatted_row}}}")
         else:
            print("No path found")                         
           





         

# totalsum=0
# for g in range(1, 11):
#     for i in range(1, 11):
        # path =  genetic("Johnston County, OK", "Lincoln Parish, LA")
#         if path != None:
#             length = len(path)
#             print(length)
#             totalsum += length
#     print(totalsum)
#     print("")
#     totalsum=0
# path =  genetic("Johnston County, OK", "Lincoln Parish, LA")
# path=genetic("Kingfisher County, OK", "Yauco Municipio, PR")

# def harel(path, neighbors_dict):
#     for i in range(len(path) - 1):
#         current_location = path[i]
#         next_location = path[i + 1]
#         # Check if the next location is a neighbor of the current location
#         if next_location not in neighbors_dict.get(current_location, []):
#             return False
#     return True
  
# print(path)
# print
# print(harel(path, neighbors_dict))


start_locations = ["Blue, Washington County, UT", "Blue, Chicot County, AR", "Red, Fairfield County, CT", "Red, Otsego County, NY", "Blue, Moody County, SD"]
goal_locations = ["Blue, San Diego County, CA", "Blue, Bienville Parish, LA", "Red, Rensselaer County, NY", "Red, Columbia County, FL", "Blue, McDowell County, WV"]


# start_locations = ["Blue, Washington County, UT", "Blue, Chicot County, AR", "Red, Fairfield County, CT"]
# goal_locations = ["Blue, San Diego County, CA", "Blue, Bienville Parish, LA", "Red, Rensselaer County, NY"]


print("")
print("")
find_path(start_locations,goal_locations,4,True)

# # print("")
# # print("")
# # find_path(start_locations,goal_locations,4,True)

# # print("")
# # print("")
# # find_path(start_locations,goal_locations,4,True)

