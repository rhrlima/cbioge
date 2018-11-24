import argparse
import sys

print(sys.argv)

parser = argparse.ArgumentParser(prog='EXP', description='Testing the Argument Parser')
parser.add_argument('-s', '--str', 
	nargs=1, 
	default='foo', 
	type=str, 
	help='some string', 
	dest='foo')

parser.add_argument('-i', '--int', 
	default=2, 
	type=int, 
	help='a integer number', 
	dest='bar')

parser.add_argument('-n', '--nodef', 
	help='a integer without default',
	dest='nodef', 
	required=True)

parser.add_argument('-d',
	help='a var without dest')

args = parser.parse_args()
print(args.foo)
print(args.bar)
#for arg in args:
#	print(arg, type(arg))