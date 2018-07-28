# Imports
from fractions import Fraction
import math
import numbers
import numpy as np
import os
import pickle

# https://stackoverflow.com/a/9758173/6026013
def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

# https://stackoverflow.com/a/9758173/6026013
def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % m

# Returns n in base b as a list
# Taken from http://stackoverflow.com/a/28666223
def to_base(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]

# Returns the roots of a polynomial function where args are the parameters of the function
def roots(*args):
    params = np.array(args)
    return np.poly1d(params).r

# Returns n clamped to min and max
def clamp(n, min, max):
    if n > max:
        return max
    elif n < min:
        return min
    else:
        return n

    # The cool math way is this but it is slower (1.9 x slower)
    # return (abs(n+1) - abs(n-1))/2

# Returns the fraction form of a floating point number
def to_fraction(n):
    return str(Fraction(n).limit_denominator())

# Returns z = x*y
def xyz_mult(x='?', y='?', z='?'):
    if z == '?':
        return x*y
    elif y == '?':
        return z/x
    elif x == '?':
        return z/y
    else:
        print("You must have one, and only one unknown")

# Prints string s in a frame of characaters
def print_in_frame(s, frame='*'):
    s = str(s)
    # Find the longest word
    biggest = 0
    for word in s.split(' '):
        if len(word) > biggest:
            biggest = len(word)

    print((frame)*(biggest+4))

    for word in s.split(' '):
        string = frame+" " + word
        string += " "*(biggest-len(word))
        string += " "+frame
        print(string)

    print((frame)*(biggest+4))

# Returns whether or not n is an int
def is_int(n):
    return isinstance(n, numbers.Integral)

# Returns whether or not a number is prime
def is_prime(n):
    if not is_int(n):
        return False

    if n < 2:
        return False

    elif n < 4:
        return True

    elif n % 2 == 0 or n % 3 == 0:
        return False

    i = 5
    while i**2 <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6

    return True

# Returns the factors of n (in pairs) as a list of pairs
def factors(n):
    factors = []
    for i in range(1, math.floor(math.sqrt(n) + 1)):
        if n % i == 0:
            factors.append([i, int(n/i)])

    return factors

# Retuns whether or not n is a perfect square (ie. x^2 = n)
def is_perfect_square(n):
    s = math.sqrt(n)
    if math.floor(s) == s:
        return True
    return False

# Returns the minimum perimeter of a given area
def min_perim(area):

    # min_perim(x^2 - 1) = [x-1, x+1]
    # Note: this could be done for any differnece of squares
    if is_perfect_square(area + 1):
        return [int(math.sqrt(area-1)) - 1, int(math.sqrt(area-1)) + 1]

    # If the area is a perfect square, return its square root
    elif is_perfect_square(area):
        s = math.sqrt(area)
        return [int(s), int(s)]

    # If the area is prime, return 1 and area
    elif is_prime(area):
        return [1, area]

    # Otherwise return the smallest perimeter (smallest sum of pair of factors)
    return factors(area)[-1]
    '''
    min_pair = [1, area]
    for pair in factors(area):
        if sum(pair) < sum(min_pair):
            min_pair = pair
    return min_pair
    '''

# Solves a linear system of equations. Input is two lists which hold the equations of the form (Ax + By = C) where A, B, and C are the contents of one of the lists
def solve_linear_system(eq1, eq2):
    a = np.array([[eq1[0], eq1[1]], [eq2[0], eq2[1]]])
    b = np.array([eq1[2], eq2[2]])
    try:
        solution = np.linalg.solve(a, b)
        if np.allclose(np.dot(a, solution), b) == True:
            return solution
    except np.linalg.linalg.LinAlgError:
        print("No solution found")
        return []

# Prints all numbers from "min" to "max" with increment "increment" on a new line
def count(mn, mx, increment=1):
    if mn > mx:
        increment *= -1
    for i in range(min, max, increment):
        print(i)

# Returns the product of a list
def product(list):
    product = 1
    for item in list:
        product *= item
    return product

# Computes distance (d), initial and final velocity (v1) (v2), acceleration (a) and delta time (t)
def kinematics(d='?', v1='?', v2='?', a='?', t='?'):

    d_eqn = "Given"
    v1_eqn = "Given"
    v2_eqn = "Given"
    a_eqn = "Given"
    t_eqn = "Given"

    if d == '?':
        if a == '?':
            # d = (v1 + v2)/2 * t
            d = (v1+ v2) * (t/2)
            d_eqn = "d = (v1 + v2)/2 * t"

        elif t == '?':
            # v2^2 = v1^2 + 2ad
            d = (v2**2 - v1**2)/(2*a)
            d_eqn = "v2^2 = v1^2 + 2ad"

        elif v2 == '?':
            # d = v1*t + (a*t^2)/2
            d = v1*t + (a * (t**2))/2
            d_eqn = "d = v1*t + (a*t^2)/2"

        else:
            # v1 = v2 - a*t
            # d = v1*t + (a*t^2)/2
            d = (v1*t) + (a*t*t)/2
            d_eqn = "d = v1*t + (a*t^2)/2"

    if v1 == '?':
        if v2 == '?':
            # v1 = (d - (a*t^2)/2)/t
            v1 = (d - (a * (t**2)/2)/t)
            v1_eqn = "v1 = (d - (a*t^2)/2)/t"

        elif d == '?':
            # v1 = v2 - (a*t)
            v1 = v2 - (a * t)
            v1_eqn = "v1 = v2 - (a*t)"

        elif a == '?':
            # v1 = 2*(d/t) - v2
            v1 = 2*(d/t) - v2
            v1_eqn = "v1 = 2*(d/t) - v2"

        elif t == '?':
            # v1 = sqrt(v2^2 - (2*ad))
            v1 = math.sqrt(v2**2 - 2*(a*d))
            v1_eqn = "v1 = sqrt(v2^2 - (2*ad))"

        else:
            # v1 = (d - (a*t^2)/2)/t
            v1 = (d - (a * (t**2)/2)/t)
            v1_eqn = "v1 = (d - (a*t^2)/2)/t"

    if v2 == '?':
        if d == '?':
            # v2 = v1 + (a*t)
            v2 = v1 + (a*t)
            v2_eqn = "v2 = v1 + (a*t)"

        elif a == '?':
            # v2 = 2*(d/t) - v1
            v2 = 2*(d/t) - v1
            v2_eqn = "v2 = 2*(d/t) - v1"

        elif t == '?':
            # v2 = sqrt(v1^2 + 2ad)
            v2 = math.sqrt(v1**2 + (2*a*d))
            v2_eqn = "v2 = sqrt(v1^2 + 2ad)"

        else:
            # v2 = sqrt(v1^2 + 2ad)
            v2 = math.sqrt(v1**2 + (2*a*d))
            v2_eqn = "v2 = sqrt(v1^2 + 2ad)"

    if a == '?':
        if d == '?':
            # a = (v2 - v1)/t
            a = (v2 - v2)/t
            a_eqn = "a = (v2 - v1)/t"

        elif t == '?':
            # a = (v2^2 - v1^2)/(2d)
            a = (v2**2 - v1**2)/(2*d)
            a_eqn = "a = (v2^2 - v1^2)/(2d)"

        elif v2 == '?':
            # a = 2*(d - (v1*t))/(t^2)
            a = 2*(d - (v1*t))/(t**2)
            a_eqn = "a = 2*(d - (v1*t))/(t^2)"

        else:
            # a = 2*(d - (v1*t))/(t^2)
            a = 2*(d - (v1*t))/(t**2)
            a_qn = "a = 2*(d - (v1*t))/(t^2)"

    second_t = '~'

    if t == '?':
        if d == '?':
            # t = (v2 - v1)/a
            t = (v2 - v1)/a
            t_eqn = "t = (v2 - v1)/a"

        elif a == '?':
            # t = (2*d)/(v1+v2)
            t = (2 * t)/(v1 + v2)
            t_eqn = "t = (2*d)/(v1+v2)"

        elif v2 == '?':
            t = (-1/a)*(math.sqrt((2*a*d) + (v1**2)) + v1)
            second_t = (1/a)*(math.sqrt((2*a*d) + (v1**2)) - v1)
            if t == second_t or second_t < 0:
                second_t = '~'
            t_eqn = "t = (-1/a) * (sqrt(2ad + v1^2) + v1)"

        else:
            t = (-1/a)*(math.sqrt((2*a*d) + (v1**2)) + v1)
            second_t = (1/a)*(math.sqrt((2*a*d) + (v1**2)) - v1)
            if t == second_t or second_t < 0:
                second_t = '~'
            t_eqn = "t = (-1/a) * (sqrt(2ad + v1^2) + v1)"

    print("Acceleration: {0}\t\t------\t\tEquation: {1}\nDisplacement: {2}\t\t------\t\tEquation: {3}\nInitial Velocity: {4}\t\t------\t\tEquation: {5}\nFinal Velocity: {6}\t\t------\t\tEquation: {7}".format(a, a_eqn, d, d_eqn, v1, v1_eqn, v2, v2_eqn))
    if second_t != '~':
        print("Time: {0} and {1}\t\t------\t\tEquation: {2}".format(t, second_t, t_eqn))
    else:
        print("Time: {0}\t------\t\tEquation: {1}".format(t, t_eqn))

    return {'a': a, 'd': d, 'v1': v1, 'v2': v2, 't': t}

# Computes distance (d), initial and final velocity (v1) (v2), acceleration (a) and delta time (t) in a two dimensional kinematics
def kinematics2d(d='?', v1='?', v2='?', a='?', t='?'):

    # Complete all the vectors
    if d != '?':
        d.calc_unknown()
    if v1 != '?':
        v1.calc_unknown()
    if v2 != '?':
        v2.calc_unknown()
    if a != '?':
        a.calc_unknown()

    # Calculate all the information in the horizontal component
    horizontal = kinematics(d.x, v1.x, v2.x, a.x, t)

    # Calculate all the information in the vertical component
    vertical = kinematics(d.y, v1.y, v2.y, a.y, t)

    # Time is the only variable that the two components must share
    if horizontal['t'] != '?':
        vertical['t'] = horizontal['t']
    else:
        horizontal['t'] = vertical['t']

    horizontal = kinematics(horizontal['d'], horizontal['v1'], horizontal['v2'], horizontal['a'], horizontal['t'])
    vertical = kinematics(vertical['d'], vertical['v1'], vertical['v2'], vertical['a'], vertical['t'])

    return {'a':  Vec2(horizontal['a'], vertical['a']),
            'd':  Vec2(horizontal['d'], vertical['d']),
            'v1': Vec2(horizontal['v1'], vertical['v1']),
            'v2': Vec2(horizontal['v2'], vertical['v2']),
            't':  Vec2(horizontal['t'], vertical['t'])}


# Returns wether or not s is a palindrome (s can be a string or a number)
def is_palindrome(s):
    s = str(s)
    if s == reverse(s):
        return True

    else:
        return False

# Scales an ascii image (as a list) according to the scale factor
def scale_ascii(image, scale_factor = 1):
    print("I need to finish this")
    width = 0
    height = len(image) # The number of rows
    for row in image:
        width = len(row) # The number of cols

    scaled = []
    temp_col = []
    for i in range(0, height):
        for j in range(0, scale_factor):
            for k in range(0, width):
                for l in range(0, scale_factor):
                    temp_col.append(image[i][k])
                scaled.append(temp_col)
    return scaled

def print_grid(grid):
    width = 0
    height = len(grid) # The number of rows
    for row in grid:
        width = len(row) # The number of cols

    for i in range(0, width):
        for j in range(0, height):
            print(grid[i][j], end="")
        print()

# Returns the slope of a line that intersects (x1, y1) and (x2, y2)
def slope(x1, x2, y1, y2):
    a = 0
    try:
        a = (y2 - y1) / (x2 - x1)
    except ZeroDivisionError:
        return "Undefined slope"
    return a

# Returns the prime factors of n
def prime_factors(n):
    is_odd = n % 2
    i = 2
    if is_odd:
        i = 3
    factors = []
    while i * i <= n:
        if n % i != 0:
            if is_odd:
                i += 2
            else:
                i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

class Vec2(object):

    # Constructor
    def __init__(self, x='?', y='?', magnitude='?', theta_x='?', theta_y='?'):
        self.x = x
        self.y = y
        self.magnitude = magnitude
        self.theta_x = theta_x
        self.theta_y = theta_y

        if self.x == self.y == self.magnitude == self.theta_x == self.theta_y == '?':
            self.x = 0
            self.y = 0

        self.calc_unknown()

    def calc_unknown(self):

        if self.x == self.y == 0:
            self.magnitude = 0
            self.theta_x = 0
            self.theta_y = 0
            return

        self.theta_x = math.radians(self.theta_x) if self.theta_x != '?' else '?'
        self.theta_y = math.radians(self.theta_y) if self.theta_y != '?' else '?'

        variables = [self.x, self.y, self.magnitude, self.theta_x, self.theta_y]
        if variables.count('?') > 3:
            #print("[WARNING] Too many unknowns")
            return
        if variables[:3].count('?') == 3:
            #print("[WARNING] No information about the size of any length components")
            return

        if self.x == '?':
            if self.y != '?':
                if self.magnitude != '?':
                    self.x = math.sqrt(self.magnitude**2 - self.y**2)
                elif self.theta_x != '?':
                    self.x = self.y/math.tan(self.theta_x)
                elif self.theta_y != '?':
                    self.x = self.y*math.tan(self.theta_y)

            elif self.magnitude != '?':
                if self.theta_x != '?':
                    self.x = self.magnitude*math.cos(self.theta_x)
                elif self.theta_y != '?':
                    self.x = self.magnitude*math.sin(self.theta_y)

        if self.y == '?':
            if self.x != '?':
                if self.magnitude != '?':
                    self.y = math.sqrt(self.magnitude**2 - self.x**2)
                elif self.theta_x != '?':
                    self.y = self.x * math.tan(self.theta_x)
                elif self.theta_y != '?':
                    self.y = math.tan(self.theta_y)/self.x

            elif self.magnitude != '?':
                if self.theta_x != '?':
                    self.y = math.sin(theta_x)*self.magnitude
                elif self.theta_y != '?':
                    self.y = math.cos(theta_y)*self.magnitude

        if self.magnitude == '?':
            self.magnitude = math.sqrt(self.x**2 + self.y**2)

        if self.theta_x == '?':
            self.theta_x = math.asin(self.y/self.magnitude)

        if self.theta_y == '?':
            self.theta_y = math.radians(90) - self.theta_x

        self.theta_x = math.degrees(self.theta_x)
        self.theta_y = math.degrees(self.theta_y)

    # Prints all of the components and angles of the vector
    def print_info(self):
        print("||<{0}, {1}>|| = {2}".format(self.x, self.y, self.magnitude))
        theta = u'\N{GREEK SMALL LETTER THETA}'
        print("x {0}: {1}\ny {0}: {2}".format(theta, self.x_theta, self.y_theta))


    # Dot product
    def dot(self, other):
        self.calc_unknown()
        other.calc_unknown()
        return self.x*other.x + self.y*other.y

    # Normalizes 'sel'
    def normalize(self):
        self.calc_unknown()
        self /= self.magnitude
        self.calc_unknown()

    # Returns 'self' if it were to be normalized
    def normalized(self):
        self.calc_unknown()
        v = Vec2(x=self.x/self.magnitude, y=self.y/self.magnitude)
        v.calc_unknown()
        return v

    # Resizes 'self' to a vector with magnitude 'sz'
    def resize(self, sz):
        self.calc_unknown()
        self.normalize()
        self *= sz
        self.calc_unknown()

    # Returns 'self' if it were to be resized to a vector with magnitude 'sz'
    def resized(self, sz):
        self.calc_unknown()
        v = self.normalized()
        v *= sz
        v.calc_unknown()
        return v

    # Rotates 'self' by 'angle' degrees around point 'pivot'
    def rotate(self, angle, pivot=None):
        if pivot == None:
            pivot = self

        rad = math.radians(angle)

        s = math.sin(rad)
        c = math.cos(rad)

        self -= pivot

        new_x = (self.x * c) - (self.y * s)
        new_y = (self.x * s) + (self.x * c)

        self.x = new_x + pivot.x
        self.y = new_y + pivot.y

    # Returns 'self' if it were to have been roatetd by 'angle' degrees around point 'pivot'
    def rotated(self, angle, pivot=None):
        v = Vec2()
        if pivot == None:
            pivot = self

        rad = math.radians(angle)

        s = math.sin(rad)
        c = math.cos(rad)

        self -= pivot

        v.x = (self.x * c) - (self.y * s)
        v.y = (self.x * s) + (self.x * c)

        v += pivot

        return v

    # Translates 'self' in direction 'dir' by 'n' units
    def translate(self, n, dir):
        print("I should probably program this function...")

    # Returns 'self' if it were
    def translated(self, n, dir):
        print("I should probably program this function...")

    # Addition operator overload
    def __add__(self, other):
        self.calc_unknown()
        other.calc_unknown()
        v = Vec2()
        v.x = self.x + other.x
        v.y = self.y + other.y
        v.calc_unknown()
        return v

    # Right addition operator overload
    def __radd__(self, other):
        self.calc_unknown()
        other.calc_unknown()
        v = Vec2()
        v.x = self.x + other.x
        v.y = self.y + other.y
        v.calc_unknown()
        return v

    # Subtraction operator overload
    def __sub__(self, other):
        self.calc_unknown()
        other.calc_unknown()
        v = Vec2()
        v.x = self.x - other.x
        v.y = self.y - other.y
        v.calc_unknown()
        return v

    # Right subtraction operator overload
    def __rsub__(self, other):
        self.calc_unknown()
        other.calc_unknown()
        v = Vec2()
        v.x = self.x - other.x
        v.y = self.y - other.y
        v.calc_unknown()
        return v

    # Multiplication operator overload
    def __mul__(self, other):
        self.calc_unknown()
        v = Vec2()
        v.x = self.x * other
        v.y = self.y * other
        v.calc_unknown()
        return v

    # Right multiplication operator overload
    def __rmul__(self, other):
        self.calc_unknown()
        v = Vec2()
        v.x = other * self.x
        v.y = other * self.y
        v.calc_unknown()
        return v

    # Division operator overload
    def __div__(self, other):
        self.calc_unknown()
        v = Vec2()
        v.x = self.x / other
        v.y = self.y / other
        v.calc_unknown()
        return v

    # Right division operator overload
    def __rdiv__(self, other):
        self.calc_unknown()
        v = Vec2()
        v.x = self.x / other
        v.y = self.y / other
        v.calc_unknown()
        return v

    # String operator overload
    def __str__(self):
        self.calc_unknown()
        theta = u'\N{GREEK SMALL LETTER THETA}'
        s = "||<{0}, {1}>|| = {2}\nx {3}: {4}\ny {3}: {5}".format(self.x, self.y, self.magnitude, theta, self.theta_x, self.theta_y)
        return s

    # String operator overload
    def __repr__(self):
        return self.__str__()

    # Less-than operator overload
    def __lt__(self, other):
        self.calc_unknown()
        other.calc_unknown()
        return self.magnitude < other.magnitude

    # Less-than or equal to operator overload
    def __le__(self, other):
        self.calc_unknown()
        other.calc_unknown()
        return self.magnitude <= other.magnitude

    # Greater-than operator overload
    def __gt__(self, other):
        self.calc_unknown()
        other.calc_unknown()
        return self.magnitude > other.magnitude

    # Greater-than or equal to operator overload
    def __ge__(self, other):
        self.calc_unknown()
        other.calc_unknown()
        return self.magnitude >= other.magnitude

# Fully connected, feedforward, artificial neural network
class FC_FF_ANN(object):

    # Constructor
    def __init__(self, topology, alpha=1):

        self.topology = topology
        self.num_synapses = len(self.topology)-1
        self.num_layers = self.num_synapses + 1
        self.alpha = alpha

        self.synapses = []
        for i in range(self.num_synapses):
            self.synapses.append((2*np.random.rand(topology[i], topology[i+1]))-1)

        self.layers = []
        self.layer_errors = []
        self.layer_deltas = []

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def feedforward(self, inpt):

        # Reset
        self.layers = []

        # Fill in data
        self.layers.append(inpt)
        for i in range(self.num_layers-1): # -1 because we already did the first layer
            self.layers.append(self.sigmoid(np.dot(self.layers[i], self.synapses[i])))

        return self.layers[-1]

    def train(self, inpt, outpt, print_error=False):

        # Feed forward
        self.feedforward(inpt)

        # Reset
        self.layer_errors = []
        self.layer_deltas = []

        self.layer_errors.append(self.layers[-1] - outpt)
        self.layer_deltas.append(self.layer_errors[-1] * self.sigmoid_prime(self.layers[-1]))

        if print_error:
            print("Error: {0}".format(str(np.mean(np.abs(self.layer_errors[-1])))))

        for i in range(self.num_layers - 1): # -1 because we already did the first layer
            self.layer_errors.append(self.layer_deltas[-1].dot(self.synapses[(-1*i)-1].T))
            self.layer_deltas.append(self.layer_errors[-1] * self.sigmoid_prime(self.layers[(-1*i)-2]))

        for i in range(self.num_synapses):
            self.synapses[(-1*i)-1] -= self.alpha * self.layers[(-1*i)-2].T.dot(self.layer_deltas[i])

    def save_state(self, file_path):

        with open(file_path, 'wb') as outfile:
            pickle.dump(self.synapses, outfile)

    def read_state(self, file_path):

        with open(file_path, 'rb') as infile:
            self.synapses = pickle.load(infile)

'''
TO BE REMOVED:

class Vec2(object):

    # Constructor
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    # Translate by (dx, dy)
    def translate(dx=0, dy=0):
        self.x += dx
        self.y += dy


    # Rotate by angle degrees around point p
    def rotate(self, angle, pivot):
        s = math.sin(angle)
        c = math.cos(angle)

        self.x -= pivot.x
        self.y -= pivot.y

        new_x = (self.x * c) - (self.y * s)
        new_y = (self.x * s) + (self.x * c)

        self.x = new_x + pivot.x
        self.y = new_y + pivot.y

    # Addition operator overload
    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    # Subtraction operator overload
    def __sub__(self, other):
        return point(self.x - other.x, self.y - other.y)

    # Multiplication operator overload
    def __mul__(self, other):
        return point(self.x * other.x, self.y * other.y)

    # Division operator overload
    def __div__(self, other):
        return point(self.x / other.x, self.y / other.y)

    # String operator overload
    def __str__(self):
        return "({}, {})".format(self.x, self.y)

    # String operator overload
    def __repr__(self):
        return "({}, {})".format(self.x, self.y)

    # Less-than operator overload
    def __lt__(self, other):
        return math.sqrt(self.x**2 + self.y**2) < math.sqrt(other.x**2 + other.y**2)

    # Less-than or equal to operator overload
    def __le__(self, other):
        return math.sqrt(self.x**2 + self.y**2) <= math.sqrt(other.x**2 + other.y**2)

    # Greater-than operator overload
    def __gt__(self, other):
        return math.sqrt(self.x**2 + self.y**2) > math.sqrt(other.x**2 + other.y**2)

    # Greater-than or equal to operator overload
    def __ge__(self, other):
        return math.sqrt(self.x**2 + self.y**2) >= math.sqrt(other.x**2 + other.y**2)

    # Equality operator overload
    def __eq__(self, other):
        return math.sqrt(self.x**2 + self.y**2) == math.sqrt(other.x**2 + other.y**2)

    # Not equality operator overload
    def __ne__(self, other):
        return math.sqrt(self.x**2 + self.y**2) !=  math.sqrt(other.x**2 + other.y**2)

# Vector class
class vector:

    # Constructor
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.magnitude = '?'
        self.x_theta = '?'
        self.y_theta = '?'
        self.right_angle = 90
        self.calc_unknown()

    # Calculate x component based on magnitude and y component
    def calc_x(self):
        self.x = math.sqrt(self.magnitude*self.magnitude - self.y*self.y)
        return self.x

    # Calculate y component based on magnitude and x component
    def calc_y(self):
        self.y = math.sqrt(self.magnitude*self.magnitude - self.x*self.x)
        return self.y

    # Calculate magnitude based on x and y component
    def calc_magnitude(self):
        self.magnitude = math.sqrt(self.x*self.x + self.y*self.y)
        return self.magnitude

    # Calculate angle of x component and magnitude of vector
    def calc_x_theta(self):
        if self.y != 0 and self.magnitude != 0:
            self.x_theta = math.degrees(math.asin(self.y / self.magnitude))

        elif self.x == 0 and self.magnitude != 0:
            self.x_theta = math.degrees(math.acos(self.x / self.magnitude))

        elif self.x != 0 and self.y != 0:
            self.x_theta = math.degrees(math.atan(self.y / self.x))

        else:
            print("Error: calc_x_theta needs at least two components to be non-zero")

    # Calculate angle of x component and magnitude of vector
    def calc_y_theta(self):
        if self.y != 0 and self.magnitude != 0:
            self.y_theta = math.degrees(math.acos(self.y / self.magnitude))

        elif self.x == 0 and self.magnitude != 0:
            self.y_theta = math.degrees(math.asin(self.x / self.magnitude))

        elif self.x != 0 and self.y != 0:
            self.y_theta = math.degrees(math.atan(self.x / self.y))

        else:
            print("Error: calc_y_theta needs at least two components to be non-zero")

    # Prints all of the components and angles of the vector
    def print_info(self):
        print("||<{}, {}>|| = {}".format(self.x, self.y, self.magnitude))
        theta = u'\N{GREEK SMALL LETTER THETA}'
        print("x {}: {}\ny {}: {}".format(theta, self.x_theta, theta, self.y_theta))

    # Calculates all unknowns ('?') in the vector
    def calc_unknown(self):
        unknown_legs = 0
        if self.x == '?':
            unknown_legs += 1
        if self.y == '?':
            unknown_legs += 1
        if self.magnitude == '?':
            unknown_legs += 1

        if unknown_legs == 3:
            print("You must know at least one component to calculate other unknowns")

        if unknown_legs == 2:
            if self.x != '?':
                if self.x_theta == '?':
                    self.magnitude = self.x/(math.sin(math.radians(self.y_theta)))
                else:
                    self.magnitude = self.x/(math.cos(math.radians(self.x_theta)))

            elif self.y != '?':
                if self.x_theta == '?':
                    self.magnitude = self.y/(math.cos(math.radians(self.y_theta)))
                else:
                    self.magnitude = self.y/(math.sin(math.radians(self.x_theta)))

            elif self.magnitude != '?':
                if self.x_theta == '?':
                    self.x = math.sin(math.radians(self.y_theta))*self.magnitude
                else:
                    self.x = math.cos(math.radians(self.x_theta))*self.magnitude

            else:
                print("You must know at least one angle to calculate other unknowns")

        if self.x == '?':
            self.x = math.sqrt(self.magnitude*self.magnitude - self.y*self.y)
        elif self.y == '?':
            self.y = math.sqrt(self.magnitude*self.magnitude - self.x*self.x)
        else:
            self.magnitude = math.sqrt(self.x*self.x + self.y*self.y)

        self.x_theta = math.degrees(math.asin(self.y/self.magnitude))
        self.y_theta = 90 - self.x_theta

    # Dot product
    def dot(self, other):
        self.calc_unknown()
        other.calc_unknown()
        return self.x*other.x + self.y*other.y

    # Addition operator overload
    def __add__(self, other):
        self.calc_unknown()
        other.calc_unknown()
        new_vec = vector()
        new_vec.x = self.x + other.x
        new_vec.y = self.y + other.y
        new_vec.calc_magnitude()
        return new_vec

    # Subtraction operator overload
    def __sub__(self, other):
        self.calc_unknown()
        other.calc_unknown()
        new_vec = vector()
        new_vec.x = self.x - other.x
        new_vec.y = self.y - other.y
        new_vec.calc_magnitude()
        return new_vec

    # Multiplication operator overload
    def __mul__(self, other):
        self.calc_unknown()
        other.calc_unknown()
        new_vec = vector()
        new_vec.x = self.x * other.x
        new_vec.y = self.y * other.y
        new_vec.calc_magnitude()
        return new_vec

    # Division operator overload
    def __div__(self, other):
        self.calc_unknown()
        other.calc_unknown()
        new_vec = vector()
        new_vec.x = self.x / other.x
        new_vec.y = self.y / other.y
        new_vec.calc_magnitude() return new_vec

    # String operator overload
    def __str__(self):
        self.calc_unknown()
        theta = u'\N{GREEK SMALL LETTER THETA}'
        string = "||<{}, {}>|| = {}\nx {}: {}\ny {}: {}".format(self.x, self.y, self.magnitude, theta, self.x_theta, theta, self.y_theta)
        return string

    # String operator overload
    def __repr__(self):
        self.calc_unknown()
        theta = u'\N{GREEK SMALL LETTER THETA}'
        string = "||<{}, {}>|| = {}\nx {}: {}\ny {}: {}".format(self.x, self.y, self.magnitude, theta, self.x_theta, theta, self.y_theta)
        return string

    # Less-than operator overload
    def __lt__(self, other):
        self.calc_unknown()
        other.calc_unknown()
        return self.magnitude < other.magnitude

    # Less-than or equal to operator overload
    def __le__(self, other):
        self.calc_unknown()
        other.calc_unknown()
        return self.magnitude <= other.magnitude

    # Greater-than operator overload
    def __gt__(self, other):
        self.calc_unknown()
        other.calc_unknown()
        return self.magnitude > other.magnitude

    # Greater-than or equal to operator overload
    def __ge__(self, other):
        self.calc_unknown()
        other.calc_unknown()
        return self.magnitude >= other.magnitude

    # Equality operator overload
    def __eq__(self, other):
        self.calc_unknown()
        other.calc_unknown()
        return self.magnitude == other.magnitude

    # Not equality operator overload
    def __ne__(self, other):
        self.calc_unknown()
        other.calc_unknown()
        return self.magnitude != other.magnitude

'''
