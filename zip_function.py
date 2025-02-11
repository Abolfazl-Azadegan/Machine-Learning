# Iterating Over Multiple Lists Simultaneously
names = ["Alice", "Bob", "Charlie"]
scores = [85, 92, 78]

for name, score in zip(names, scores):
    print(f"{name} scored {score}")
    


# Creating Dictionaries from Two Lists 
keys = ["name", "age", "city"]
values = ["Alice", 30, "New York"]

person_dict = dict(zip(keys, values))
print(person_dict)



# Important Notes:
    
# Returns an Iterator:
# zip() returns an iterator, which means it doesn’t produce the final result immediately. You need to convert it to a list or tuple 
# to view the output, like list(zip(...)).


# Memory Efficient:
# Since it returns an iterator, zip() is memory-efficient for large datasets.


from itertools import zip_longest

names = ["Alice", "Bob"]
scores = [85, 92, 78]

zipped = zip_longest(names, scores, fillvalue='N/A')
print(list(zipped))


