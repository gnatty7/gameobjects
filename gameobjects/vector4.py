from math import sqrt
from .util import format_number
from .vector3 import Vector3

class Vector4(object):

    __slots__ = ('_v',)

    _gameobjects_vector = 4


    def __init__(self, *args):
        """Creates a Vector4 from 4 numeric values or a list-like object
        containing at least 3 values. No arguments result in a null vector.
        """

        if len(args) == 4:
            self._v = list(map(float, args[:4]))
            return

        if not args:
            self._v = [0., 0., 0., 0.]
        elif len(args) == 1:
            if isinstance(args[0], Vector3):
                x, y, z = args[0]
                self._v = [x, y, z, 0.]
            else:
                self._v = list(map(float, args[0][:4]))
        else:
            raise ValueError("Vector4.__init__ takes 0, 1 or 4 parameters")


    @classmethod
    def from_points(cls, p1, p2):

        v = cls.__new__(cls, object)
        ax, ay, az, aw = p1
        bx, by, bz, bw = p2
        v._v = [bx-ax, by-ay, bz-az, bw-aw]

        return v

    @classmethod
    def from_floats(cls, x, y, z, w):
        """Creates a Vector4 from individual float values.
        Warning: There is no checking (for efficiency) here: x, y, z, w _must_ be
        floats.

        """
        v = cls.__new__(cls, object)
        v._v = [x, y, z, w]
        return v


    @classmethod
    def from_iter(cls, iterable):
        """Creates a Vector4 from an iterable containing at least 4 values."""
        next = iter(iterable).__next__
        v = cls.__new__(cls, object)
        v._v = [ float(next()), float(next()), float(next()), float(next()) ]
        return v

    @classmethod
    def _from_float_sequence(cls, sequence):
        v = cls.__new__(cls, object)
        v._v = list(sequence[:4])
        return v

    def copy(self):
        """Returns a copy of this vector."""

        v = self.__new__(self.__class__, object)
        v._v = self._v[:]
        return v
        #return self.from_floats(self._v[0], self._v[1], self._v[2], self._v[3])

    __copy__ = copy

    def _get_x(self):
        return self._v[0]        
    def _set_x(self, x):
        try:
            self._v[0] = 1.0 * x
        except:
            raise TypeError("Must be a number")
    x = property(_get_x, _set_x, None, "x component.")

    def _get_y(self):
        return self._v[1]
    def _set_y(self, y):
        try:
            self._v[1] = 1.0 * y
        except:
            raise TypeError("Must be a number")
    y = property(_get_y, _set_y, None, "y component.")

    def _get_z(self):
        return self._v[2]
    def _set_z(self, z):
        try:
            self._v[2] = 1.0 * z
        except:
            raise TypeError("Must be a number")
    z = property(_get_z, _set_z, None, "z component.")

    def _get_w(self):
        return self._v[3]
    def _set_w(self, w):
        try:
            self._v[3] = 1.0 * w
        except:
            raise TypeError("Must be a number")
    w = property(_get_w, _set_w, None, "w component.")

    def _get_length(self):
        x, y, z, w = self._v
        return sqrt(x*x + y*y + z*z + w*w)

    def _set_length(self, length):
        v = self._v
        try:
            x, y, z, w = v
            l = length / sqrt(x*x + y*y +z*z + w*w)
        except ZeroDivisionError:
            v[0] = 0.
            v[1] = 0.
            v[2] = 0.
            v[3] = 0.
            return self

        v[0] = x*l
        v[1] = y*l
        v[2] = z*l
        v[3] = w*l

    length = property(_get_length, _set_length, None, "Length of the vector")

    def unit(self):
        """Returns a unit vector."""
        x, y, z, w = self._v
        l = sqrt(x*x + y*y + z*z + w*w)
        return self.from_floats(x/l, y/l, z/l, w/1)


    def set(self, x, y, z, w):
        """Sets the components of this vector.
        x -- x component
        y -- y component
        z -- z component
        w -- w component

        """

        v = self._v
        try:
            v[0] = x * 1.0
            v[1] = y * 1.0
            v[2] = z * 1.0
            v[3] = w * 1.0
        except TypeError:
            raise TypeError("Must be a number")
        return self


    def __str__(self):

        x, y, z, w = self._v
        return "({:8}, {:8}, {:8}, {:8})".format(format_number(x),
                                                     format_number(y),
                                                     format_number(z),
                                                     format_number(w))


    def __repr__(self):

        x, y, z, w = self._v
        return "Vector4(%s, %s, %s, %s)" % (x, y, z, w)


    def __len__(self):

        return 4

    def __iter__(self):
        """Iterates the components in x, y, z, w order."""
        return iter(self._v[:])

    def __getitem__(self, index):
        """Retrieves a component, given its index.

        index -- 0, 1 or 2 for x, y, z or w

        """
        try:
            return self._v[index]
        except IndexError:
            raise IndexError("There are 4 values in this object, index should be 0, 1, 2 or 3!")

    def __setitem__(self, index, value):
        """Sets a component, given its index.

        index -- 0, 1 or 2 for x, y, z or w
        value -- New (float) value of component

        """

        try:
            self._v[index] = 1.0 * value
        except IndexError:
            raise IndexError("There are 4 values in this object, index should be 0, 1, 2 or 3!")
        except TypeError:
            raise TypeError("Must be a number")


    def __eq__(self, rhs):

        """Test for equality

        rhs -- Vector or sequence of 4 values

        """

        x, y, z, w = self._v
        xx, yy, zz, ww = rhs
        return x==xx and y==yy and z==zz and w==ww

    def __ne__(self, rhs):

        """Test of inequality

        rhs -- Vector or sequenece of 3 values

        """

        x, y, z, w = self._v
        xx, yy, zz, ww = rhs
        return x!=xx or y!=yy or z!=zz or w!=ww

    def __hash__(self):

        return hash(self._v)

    def __add__(self, rhs):
        """Returns the result of adding a vector (or collection of 4 numbers)
        from this vector.

        rhs -- Vector or sequence of 2 values

        """

        x, y, z, w = self._v
        ox, oy, oz, ow = rhs
        return self.from_floats(x+ox, y+oy, z+oz, w+ow)


    def __iadd__(self, rhs):
        """Adds another vector (or a collection of 4 numbers) to this vector.

        rhs -- Vector or sequence of 2 values

        """
        ox, oy, oz, ow = rhs
        v = self._v
        v[0] += ox
        v[1] += oy
        v[2] += oz
        v[3] += ow
        return self


    def __radd__(self, lhs):

        """Adds vector to this vector (right version)

        lhs -- Left hand side vector or sequence

        """

        x, y, z, w = self._v
        ox, oy, oz, ow = lhs
        return self.from_floats(x+ox, y+oy, z+oz, w+ow)



    def __sub__(self, rhs):
        """Returns the result of subtracting a vector (or collection of
        4 numbers) from this vector.

        rhs -- 4 values

        """

        x, y, z, w = self._v
        ox, oy, oz, ow = rhs
        return self.from_floats(x-ox, y-oy, z-oz, w-ow)


    def _isub__(self, rhs):
        """Subtracts another vector (or a collection of 4 numbers) from this
        vector.

        rhs -- Vector or sequence of 4 values

        """

        ox, oy, oz, ow = rhs
        v = self._v
        v[0] -= ox
        v[1] -= oy
        v[2] -= oz
        v[3] -= ow
        return self

    def __rsub__(self, lhs):

        """Subtracts a vector (right version)

        lhs -- Left hand side vector or sequence

        """

        x, y, z, w = self._v
        ox, oy, oz, ow = lhs
        return self.from_floats(ox-x, oy-y, oz-z, ow-w)

    def scalar_mul(self, scalar):

        v = self._v
        v[0] *= scalar
        v[1] *= scalar
        v[2] *= scalar
        v[3] *= scalar

    def vector_mul(self, vector):

        x, y, z, w = vector
        v= self._v
        v[0] *= x
        v[1] *= y
        v[2] *= z
        v[3] *= w

    def get_scalar_mul(self, scalar):

        x, y, z, w = self._v
        return self.from_floats(x*scalar, y*scalar, z*scalar, w*scalar)

    def get_vector_mul(self, vector):

        x, y, z, w = self._v
        xx, yy, zz, ww = vector
        return self.from_floats(x * xx, y * yy, z * zz, w * ww)

    def __mul__(self, rhs):
        """Return the result of multiplying this vector by another vector, or
        a scalar (single number).


        rhs -- Vector, sequence or single value.

        """

        x, y, z, w = self._v
        if hasattr(rhs, "__getitem__"):
            ox, oy, oz, ow = rhs
            return self.from_floats(x*ox, y*oy, z*oz, w*ow)
        else:
            return self.from_floats(x*rhs, y*rhs, z*rhs, w*rhs)


    def __imul__(self, rhs):
        """Multiply this vector by another vector, or a scalar
        (single number).

        rhs -- Vector, sequence or single value.

        """

        v = self._v
        if hasattr(rhs, "__getitem__"):
            ox, oy, oz, ow = rhs
            v[0] *= ox
            v[1] *= oy
            v[2] *= oz
            v[3] *= ow
        else:
            v[0] *= rhs
            v[1] *= rhs
            v[2] *= rhs
            v[3] *= rhs

        return self

    def __rmul__(self, lhs):

        x, y, z, w = self._v
        if hasattr(lhs, "__getitem__"):
            ox, oy, oz, ow = lhs
            return self.from_floats(x*ox, y*oy, z*oz, w*ow)
        else:
            return self.from_floats(x*lhs, y*lhs, z*lhs, w*lhs)


    def __div__(self, rhs):
        """Return the result of dividing this vector by another vector, or a scalar (single number)."""

        x, y, z, w = self._v
        if hasattr(rhs, "__getitem__"):
            ox, oy, oz, ow = rhs
            return self.from_floats(x/ox, y/oy, z/oz, w/ow)
        else:
            return self.from_floats(x/rhs, y/rhs, z/rhs, w/rhs)


    def __idiv__(self, rhs):
        """Divide this vector by another vector, or a scalar (single number)."""

        v = self._v
        if hasattr(rhs, "__getitem__"):
            ox, oy, oz, ow = rhs
            v[0] /= ox
            v[1] /= oy
            v[2] /= oz
            v[3] /= ow
        else:
            v[0] /= rhs
            v[1] /= rhs
            v[2] /= rhs
            v[3] /= rhs

        return self


    def __rdiv__(self, lhs):

        x, y, z, w = self._v
        if hasattr(lhs, "__getitem__"):
            ox, oy, oz, ow = lhs
            return self.from_floats(ox/x, oy/y, oz/z, ow/w)
        else:
            return self.from_floats(lhs/x, lhs/y, lhs/z, lhs/w)

    def scalar_div(self, scalar):

        v = self._v
        v[0] /= scalar
        v[1] /= scalar
        v[2] /= scalar
        v[3] /= scalar

    def vector_div(self, vector):

        x, y, z, w = vector
        v= self._v
        v[0] /= x
        v[1] /= y
        v[2] /= z
        v[3] /= w

    def get_scalar_div(self, scalar):

        x, y, z, w = scalar
        return self.from_floats(x/scalar, y/scalar, z/scalar, w/scalar)

    def get_vector_div(self, vector):

        x, y, z, w = self._v
        xx, yy, zz, ww = vector
        return self.from_floats(x/xx, y/yy, z/zz, w/ww)

    def __neg__(self):
        """Returns the negation of this vector (a vector pointing in the opposite direction.
        eg v1 = Vector(1,2,3,4)
        print -v1
        >>> (-1,-2,-3,-4)

        """
        x, y, z, w = self._v
        return self.from_floats(-x, -y, -z, -w)

    def __pos__(self):

        return self.copy()


    def __bool__(self):

        x, y, z, w = self._v
        return bool(x or y or z or w)


    def __call__(self, keys):
        """Returns a tuple of the values in a vector

        keys -- An iterable containing the keys (x, y, z or w)
        eg v = Vector4(1.0, 2.0, 3.0, 4.0)
        v('wzyx') -> (4.0, 3.0, 2.0, 1.0)

        """
        ord_x = ord('x')
        v = self._v
        return tuple( v[ord(c)-ord_x] for c in keys )


    def as_tuple(self):
        """Returns a tuple of the x, y, z, w components. A little quicker than
        tuple(vector)."""

        return tuple(self._v)


    def scale(self, scale):
        """Scales the vector by onther vector or a scalar. Same as the
        *= operator.

        scale -- Value to scale the vector by

        """
        v = self._v
        if hasattr(scale, "__getitem__"):
            ox, oy, oz, ow = scale
            v[0] *= ox
            v[1] *= oy
            v[2] *= oz
            v[3] *= ow
        else:
            v[0] *= scale
            v[1] *= scale
            v[2] *= scale
            v[3] *= scale

        return self


    def get_length(self):
        """Calculates the length of the vector."""

        x, y, z, w = self._v
        return sqrt(x*x + y*y + z*z + w*w)
    get_magnitude = get_length

    def set_length(self, new_length):
        """Sets the length of the vector. (Normalises it then scales it)

        new_length -- The new length of the vector.

        """
        v = self._v
        try:
            x, y, z, w = v
            l = new_length / sqrt(x*x + y*y + z*z, w*w)
        except ZeroDivisionError:
            v[0] = 0.0
            v[1] = 0.0
            v[2] = 0.0
            v[3] = 0.0
            return self

        v[0] = x*l
        v[1] = y*l
        v[2] = z*l
        v[3] = w*l

        return self


    def get_distance_to(self, p):
        """Returns the distance of this vector to a point.

        p -- A position as a vector, or collection of 4 values.

        """
        ax, ay, az, aw = self._v
        bx, by, bz, bw = p
        dx = ax-bx
        dy = ay-by
        dz = az-bz
        dw = aw-bw
        return sqrt( dx*dx + dy*dy + dz*dz + dw*dw )


    def get_distance_to_squared(self, p):
        """Returns the squared distance of this vector to a point.

        p -- A position as a vector, or collection of 3 values.

        """
        ax, ay, az, aw = self._v
        bx, by, bz, bw = p
        dx = ax-bx
        dy = ay-by
        dz = az-bz
        dw = aw-bw
        return dx*dx + dy*dy + dz*dz + dw*dw


    def normalise(self):
        """Scales the vector to be length 1."""
        v = self._v
        x, y, z, w = v
        l = sqrt(x*x + y*y + z*z + w*w)
        try:
            v[0] /= l
            v[1] /= l
            v[2] /= l
            v[3] /= l
        except ZeroDivisionError:
            v[0] = 0.0
            v[1] = 0.0
            v[2] = 0.0
            v[3] = 0.0
        return self
    normalize = normalise

    def get_normalised(self):

        x, y, z, w = self._v
        l = sqrt(x*x + y*y + z*z, w*w)
        return self.from_floats(x/l, y/l, z/l, w/l)
    get_normalized = get_normalised

    def dot(self, other):

        """Returns the dot product of this vector with another.

        other -- A vector or tuple

        """
        x, y, z, w = self._v
        ox, oy, oz, ow = other
        return x*ox + y*oy + z*oz + w*ow

def distance4d_squared(p1, p2):

    x, y, z, w = p1
    xx, yy, zz, ww = p2
    dx = x - xx
    dy = y - yy
    dz = z - zz
    dw = w - ww

    return dx*dx + dy*dy +dz*dz + dw*dw


def distance4d(p1, p2):

    x, y, z, w = p1
    xx, yy, zz, ww = p2
    dx = x - xx
    dy = y - yy
    dz = z - zz
    dw = w - ww

    return sqrt(dx*dx + dy*dy +dz*dz + dw*dw)

def centre_point4d(points):

    return sum( Vector4(p) for p in points ) / len(points)


if __name__ == "__main__":

    v1 = Vector4(2.2323, 3.43242, 1., 1.)

    print(3*v1)
    print((2, 4, 6, 7)*v1)

    print((1, 2, 3, 4)+v1)
    print(v1('xxxyyyzzzwww'))
    print(v1[2])
    print(v1.w)    
    v1[2]=5.
    print(v1)
    v2= Vector4(1.2, 5, 10, 12)
    print(v2)
    v1 += v2
    print(v1.get_length())
    print(repr(v1))
    print(v1[1])

    p1 = Vector4(1, 2, 3, 4)
    print(p1)
    print(repr(p1))

    for v in p1:
        print(v)

    #print p1[6]

    ptest = Vector4( [1, 2, 3, 4] )
    print(ptest)

    z = Vector4()
    print(z)

    file("test.txt", "w").write( "\n".join(str(float(n)) for n in range(20)) )
    f = file("test.txt")
    v1 = Vector4.from_iter( f )
    v2 = Vector4.from_iter( f )
    v3 = Vector4.from_iter( f )
    print(v1, v2, v3)

    print("--")
    print(v1)
    print(v1 + (10, 20, 30, 40))

    print(v1('xw'))

    print(-v1)

    #print tuple(ptest)
    #p1.set( (4, 5, 6) )
    #print p1

    print(Vector4(10, 10, 30, 40)+v1)

    print(Vector4((0, 0, 0, 1, 10)))
    
    print(Vector4(1, 2, 3, 4).scale(3))
    
    print(Vector4(1, 2, 3, 4).scale((2, 4, 6, 8)))
    
    print(bool(v1))
