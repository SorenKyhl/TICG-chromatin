import json
import sys

'''
json sed program for quickly editing json files
usage: python3 jsed.py filename key val type
type is the data type of the value, necessary for distinguishing between
the string "true" and the boolean true, and
the string "0.1" and the float 0.1
'''

try:
	filename = sys.argv[1]
	key = sys.argv[2]
	val = sys.argv[3]
	typ = sys.argv[4]
except IndexError:
	print(sys.argv)
	raise

if typ == "i":
	val = int(val)
elif typ == "f":
	val = float(val)
elif typ == "s":
	val = str(val)
elif typ == "b":
	if val.lower() == "true":
		val = True
	elif val.lower() == "false":
		val = False
	else:
		print("boolean values enter true or false")

else:
	print("indicate the type of variable")

with open(filename, 'r') as f:
	data = json.load(f)

with open(filename, 'w') as f:
	data[key] = val
	json.dump(data, f, indent=4)
	f.write("\n")
