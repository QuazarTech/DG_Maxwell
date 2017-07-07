import arrayfire as af

def add(a, b):
	'''
	'''
	return a + b


def divide(a, b):
	'''
	'''
	return a / b


def multiply(a, b):
	'''
	'''
	return a * b


def linspace(start, end, number_of_points):
	'''
	Linspace implementation using arrayfire.
	'''
	X = af.range(number_of_points, dtype = af.Dtype.f64)
	d = (end - start) / (number_of_points - 1)
	X = X * d
	X = X + start
	
	return X
