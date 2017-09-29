import arrayfire as af
af.set_backend('opencl')
af.set_device(1)

def add(a, b):
    '''
    For broadcasting purposes, To sum two arrays of different
    shapes, A function which can sum two variables is required.
    
    Parameters
    ----------
    a : arrayfire.Array [N M 1 1]
        One of the arrays which need to be broadcasted and summed.
    
    b : arrayfire.Array [1 M L 1]
        One of the arrays which need to be broadcasted and summed.
    
    Returns
    -------
    sum : arrayfire.Array
          returns the sum of a and b. When used along with af.broadcast
          can be used to sum different size arrays.
    '''
    sum = a + b
    
    return sum


def divide(a, b):
    '''
    
    For broadcasting purposes, To divide two arrays of different
    shapes, A function which can sum two variables is required.
    
    Parameters
    ----------
    a : arrayfire.Array [N M 1 1]
        One of the arrays which need to be broadcasted and divided.
    
    b : arrayfire.Array [1 M L 1]
        One of the arrays which need to be broadcasted and divided.
    
    Returns
    -------
    quotient : arrayfire.Array
               returns the quotient a / b. When used along with af.broadcast
               can be used to give quotient of two different size arrays
               by dividing elements of the broadcasted array.
    '''
    quotient = a / b
    
    return quotient


def multiply(a, b):
    '''
       
    For broadcasting purposes, To divide two arrays of different
    shapes, A function which can sum two variables is required.
    
    Parameters
    ----------
    a : arrayfire.Array [N M 1 1]
        One of the arrays which need to be broadcasted and multiplying.
    
    b : arrayfire.Array [1 M L 1]
        One of the arrays which need to be broadcasted and multiplying.
    
    Returns
    -------
    product : arrayfire.Array
              returns the quotient a / b. When used along with af.broadcast
              can be used to give quotient of two different size arrays
              by multiplying elements of the broadcasted array.
    '''
    product = a* b
    
    return product

def power(a, b):
    '''
       
    For broadcasting purposes, To divide two arrays of different
    shapes, A function which can sum two variables is required.
    
    Parameters
    ----------
    a : arrayfire.Array [N M 1 1]
        One of the arrays which need to be broadcasted and multiplying.
    
    b : arrayfire.Array [1 M L 1]
        One of the arrays which need to be broadcasted and multiplying.
    
    Returns
    -------
    product : arrayfire.Array
              returns the quotient a / b. When used along with af.broadcast
              can be used to give quotient of two different size arrays
              by multiplying elements of the broadcasted array.
    '''
    pow = a ** b
    
    return pow



def linspace(start, end, number_of_points):
    '''
    Linspace implementation using arrayfire.
    
    Returns
    -------
    X : arrayfire.Array
        An array which contains 'number_of_points' evenly spaced points
        between 'start' and 'end'
    '''
    X = af.range(number_of_points, dtype = af.Dtype.f64)
    d = (end - start) / (number_of_points - 1)
    X = X * d
    X = X + start
    
    return X

