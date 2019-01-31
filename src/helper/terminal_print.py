
import sys

def clear_current_line():
	sys.stdout.write("\r")
	sys.stdout.write("\033[K")
	sys.stdout.flush()

def write_to_current_line(txt):
	clear_current_line()
	sys.stdout.write(txt)
	#print(txt)

def clear_previous_line():
	clear_current_line()
	sys.stdout.write("\r")
	sys.stdout.write("\033[F")
	sys.stdout.flush()
	clear_current_line()


