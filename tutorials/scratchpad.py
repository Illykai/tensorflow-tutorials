import math

initial = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
sorted = []

count = len(initial) + 1
rows = 4
columns = int(math.ceil(count / rows))
full_columns = count % columns if not count % columns == 0 else columns
partial_columns = columns - full_columns 

# Row 1
index = rows - 1
column = 1
while column < columns:
    sorted.append(index)
    if column < full_columns:
        index += rows
    else:
        index += (rows - 1)
    column = column + 1

row_start = 0
row = 1
while row < rows:
    index = row - 1
    column = 0
    while column < columns:
        sorted.append(index)
        if column < full_columns:
            index += rows
        else:
            index += (rows - 1)
        column = column + 1

        # If we're in the final row and we've finished
        # the full columns, then we're done
        if row == rows - 1 and column == full_columns:
            break
    row = row + 1

print(sorted)

table_string = ""
sorted = [str(i) for i in sorted]
sorted = ["X"] + sorted

column = 0
for value in sorted:
    table_string += value + " "
    column += 1
    if column == columns:
        column = 0
        table_string += "\n"

print(table_string)