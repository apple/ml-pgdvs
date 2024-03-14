import numpy as np
from math import sqrt, sin, cos, acos, asin


def rot_mat_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a = (vec1 / np.linalg.norm(vec1)).reshape(3)
    b = (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)

    flag_parallel = not np.any(v)

    c = np.dot(a, b)

    # flag_parallel = np.isclose(np.abs(c), 1)

    if flag_parallel:
        if c > 0:
            sign = 1
        else:
            sign = -1
        rot_mat = np.zeros(
            (3, 3)
        )  # cross of all zeros only occurs on identical directions
        rot_mat[0, 0] = sign
        rot_mat[1, 1] = sign
        rot_mat[2, 2] = sign
    else:
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rot_mat = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))

    return rot_mat


# -------------------------------------------------------------------------------------
# The following is modified from https://github.com/zhengqili/Neural-Scene-Flow-Fields/blob/d4001759a39b056c95d8bc22da34b10b4fb85afb/nsff_exp/load_llff.py#L163


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def recenter_poses(poses):
    poses_ = poses + 0  # [N, 3, 5]. NOTE: this will create a copy of the original array
    bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
    c2w = poses_avg(poses)  # [3, 5]
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)  # [4, 4]
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])  # [N, 1, 4]
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)  # [N, 4, 4]

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


# -------------------------------------------------------------------------------------
# The following is modified from https://github.com/zhengqili/Neural-Scene-Flow-Fields/blob/d4001759a39b056c95d8bc22da34b10b4fb85afb/nsff_exp/Q_Slerp.py#L9


class quaternion:
    """A quaternion is a compact method of representing a 3D rotation that has
    computational advantages including speed and numerical robustness.

    A quaternion has 2 parts, a scalar s, and a vector v and is typically written::

        q = s <vx vy vz>

    A unit quaternion is one for which M{s^2+vx^2+vy^2+vz^2 = 1}.

    A quaternion can be considered as a rotation about a vector in space where
    q = cos (theta/2) sin(theta/2) <vx vy vz>
    where <vx vy vz> is a unit vector.

    Various functions such as INV, NORM, UNIT and PLOT are overloaded for
    quaternion objects.

    Arithmetic operators are also overloaded to allow quaternion multiplication,
    division, exponentiaton, and quaternion-vector multiplication (rotation).
    """

    def __init__(self, *args):
        """
        Constructor for quaternion objects:
        - q = quaternion()                 object initialization
        - q = quaternion(s, v1, v2, v3)    from 4 elements
        """

        self.vec = []

        if len(args) == 0:
            # default is a null rotation
            self.s = 1.0
            self.v = np.matrix([0.0, 0.0, 0.0])

        elif len(args) == 4:
            self.s = args[0]
            self.v = np.mat(args[1:4])

        else:
            print("error")
            return None

    def __repr__(self):
        return "%f <%f, %f, %f>" % (self.s, self.v[0, 0], self.v[0, 1], self.v[0, 2])

    def tr2q(self, t):
        # TR2Q   Convert homogeneous transform to a unit-quaternion
        #
        #   Q = tr2q(T)
        #
        #   Return a unit quaternion corresponding to the rotational part of the
        #   homogeneous transform T.

        qs = sqrt(np.trace(t) + 1) / 2.0
        kx = t[2, 1] - t[1, 2]  # Oz - Ay
        ky = t[0, 2] - t[2, 0]  # Ax - Nz
        kz = t[1, 0] - t[0, 1]  # Ny - Ox

        if (t[0, 0] >= t[1, 1]) and (t[0, 0] >= t[2, 2]):
            kx1 = t[0, 0] - t[1, 1] - t[2, 2] + 1  # Nx - Oy - Az + 1
            ky1 = t[1, 0] + t[0, 1]  # Ny + Ox
            kz1 = t[2, 0] + t[0, 2]  # Nz + Ax
            add = kx >= 0
        elif t[1, 1] >= t[2, 2]:
            kx1 = t[1, 0] + t[0, 1]  # Ny + Ox
            ky1 = t[1, 1] - t[0, 0] - t[2, 2] + 1  # Oy - Nx - Az + 1
            kz1 = t[2, 1] + t[1, 2]  # Oz + Ay
            add = ky >= 0
        else:
            kx1 = t[2, 0] + t[0, 2]  # Nz + Ax
            ky1 = t[2, 1] + t[1, 2]  # Oz + Ay
            kz1 = t[2, 2] - t[0, 0] - t[1, 1] + 1  # Az - Nx - Oy + 1
            add = kz >= 0

        if add:
            kx = kx + kx1
            ky = ky + ky1
            kz = kz + kz1
        else:
            kx = kx - kx1
            ky = ky - ky1
            kz = kz - kz1

        kv = np.matrix([kx, ky, kz])
        nm = np.linalg.norm(kv)
        if nm == 0:
            self.s = 1.0
            self.v = np.matrix([0.0, 0.0, 0.0])

        else:
            self.s = qs
            self.v = (sqrt(1 - qs**2) / nm) * kv

    ############### OPERATORS #########################################
    # PLUS Add two quaternion objects
    #
    # Invoked by the + operator
    #
    # q1+q2 standard quaternion addition
    def __add__(self, q):
        """
        Return a new quaternion that is the element-wise sum of the operands.
        """
        if isinstance(q, quaternion):
            qr = quaternion()
            qr.s = 0

            qr.s = self.s + q.s
            qr.v = self.v + q.v

            return qr
        else:
            raise ValueError

    # MINUS Subtract two quaternion objects
    #
    # Invoked by the - operator
    #
    # q1-q2 standard quaternion subtraction

    def __sub__(self, q):
        """
        Return a new quaternion that is the element-wise difference of the operands.
        """
        if isinstance(q, quaternion):
            qr = quaternion()
            qr.s = 0

            qr.s = self.s - q.s
            qr.v = self.v - q.v

            return qr
        else:
            raise ValueError

    # q * q  or q * const
    def __mul__(self, q2):
        """
        Quaternion product. Several cases are handled

            - q * q   quaternion multiplication
            - q * c   element-wise multiplication by constant
            - q * v   quaternion-vector multiplication q * v * q.inv();
        """
        qr = quaternion()

        if isinstance(q2, quaternion):
            # Multiply unit-quaternion by unit-quaternion
            #
            #   QQ = qqmul(Q1, Q2)

            # decompose into scalar and vector components
            s1 = self.s
            v1 = self.v
            s2 = q2.s
            v2 = q2.v

            # form the product
            qr.s = s1 * s2 - v1 * v2.T
            qr.v = s1 * v2 + s2 * v1 + np.cross(v1, v2)

        elif type(q2) is np.matrix:
            # Multiply vector by unit-quaternion
            #
            #   Rotate the vector V by the unit-quaternion Q.

            if q2.shape == (1, 3) or q2.shape == (3, 1):
                qr = self * quaternion(q2) * self.inv()
                return qr.v
            else:
                raise ValueError

        else:
            qr.s = self.s * q2
            qr.v = self.v * q2

        return qr

    def __rmul__(self, c):
        """
        Quaternion product. Several cases are handled

            - c * q   element-wise multiplication by constant
        """
        qr = quaternion()
        qr.s = self.s * c
        qr.v = self.v * c

        return qr

    def __imul__(self, x):
        """
        Quaternion in-place multiplication

            - q *= q2

        """

        if isinstance(x, quaternion):
            s1 = self.s
            v1 = self.v
            s2 = x.s
            v2 = x.v

            # form the product
            self.s = s1 * s2 - v1 * v2.T
            self.v = s1 * v2 + s2 * v1 + np.cross(v1, v2)

        elif np.isscalar(x):
            self.s *= x
            self.v *= x

        return self

    def __pow__(self, p):
        """
        Quaternion exponentiation.  Only integer exponents are handled.  Negative
        integer exponents are supported.
        """

        # check that exponent is an integer
        if not isinstance(p, int):
            raise ValueError

        qr = quaternion()
        q = quaternion(self)

        # multiply by itself so many times
        for i in range(0, abs(p)):
            qr *= q

        # if exponent was negative, invert it
        if p < 0:
            qr = qr.inv()

        return qr

    def double(self):
        """Return the quaternion as 4-element vector.

        @rtype: 4-vector
        @return: the quaternion elements
        """
        return np.concatenate((np.mat(self.s), self.v), 1)  # Debug present

    def unit_Q(self):
        """
        Function asserts a quaternion Q to be a unit quaternion
        s is unchanged as the angle of rotation is assumed to be controlled by the user
        v is cast into a unit vector based on |v|^2 = 1 - s^2 and dividing by |v|
        """

        # Still have some errors with linalg.norm(self.v)
        qr = quaternion()

        try:
            nm = np.linalg.norm(self.v) / sqrt(1 - pow(self.s, 2))
            qr.s = self.s
            qr.v = self.v / nm

        except:
            qr.s = self.s
            qr.v = self.v

        return qr


# QINTERP Interpolate rotations expressed by quaternion objects
#
#   QI = qinterp(Q1, Q2, R)
#
# Return a unit-quaternion that interpolates between Q1 and Q2 as R moves
# from 0 to 1.  This is a spherical linear interpolation (slerp) that can
# be interpretted as interpolation along a great circle arc on a sphere.
#
# If r is a vector, QI, is a cell array of quaternions, each element
# corresponding to sequential elements of R.
#
# See also: CTRAJ, QUATERNION.

# MOD HISTORY
# 2/99 convert to use of objects
# $Log: qinterp.m,v $
# Revision 1.3  2002/04/14 11:02:54  pic
# Changed see also line.
#
# Revision 1.2  2002/04/01 12:06:48  pic
# General tidyup, help comments, copyright, see also, RCS keys.
#
# $Revision: 1.3 $
#
# Copyright (C) 1999-2002, by Peter I. Corke


def interpolate(Q1, Q2, r):
    q1 = Q1.double()
    q2 = Q2.double()

    theta = acos(q1 * q2.T)
    q = []
    count = 0

    if np.isscalar(r):
        if r < 0 or r > 1:
            raise Exception("R out of range")
        if theta == 0:
            q = quaternion(Q1)
        else:
            Qq = np.copy((sin((1 - r) * theta) * q1 + sin(r * theta) * q2) / sin(theta))
            q = quaternion(Qq[0, 0], Qq[0, 1], Qq[0, 2], Qq[0, 3])
    else:
        for R in r:
            if theta == 0:
                qq = Q1
            else:
                qq = quaternion(
                    (sin((1 - R) * theta) * q1 + sin(R * theta) * q2) / sin(theta)
                )
            q.append(qq)
    return q


def linear_translation(A, B, T):
    """
    Interpolates between 2 start points A an dB linearly

    Input:
    A - Start point
    B - Ending point
    T - intermediate points within a range from 0 - 1, 0 representing point A and 1 representing point B

    Output
    C - intermediate pose as an array
    """
    V_AB = B - A
    C = A + T * (V_AB)
    return C


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def linear_pose_interp(A_trans, A_rot, B_trans, B_rot, T):
    """
    Pose interpolator that calculates intermediate poses of a vector connecting 2 points
    Interpolation is done linearly for both translation and rotation
    Translation is done assuming that point A is rotationally invariant.
    Rotation is done about the point A. Quaternion SLERP rotation is used to give the quickest rotation from A -> B
    ** "Roll" or twisting of the arm is not taken into account in this calculation. Separate interpolations have to be done for the roll angle

    Input:
    Starting points start_A [X,Y,Z,roll,pitch,yaw] list
    Ending points end_B in [X',Y',Z',roll',pitch',yaw'] list
        Yaw, Pitch, Roll calculated in radians within bounds [0, 2*pi]
        Sequence of rotation: Roll - 1st, Pitch - 2nd, Yaw - 3rd
    T = no of intermediate poses

    Output:
    list of positions and rotations stored into the variable track
        track is a dictionary with keys 'lin' and 'rot'

    track['lin'] - Linear interpolation of interval T of starting positions from A -> B
    track['rot'] - Slerp interpolation of quaternion of interval T, arranged as a list in [w x y z]
    # track['rot'] - Intermediate Yaw-Pitch-Roll poses of interval T, in sequence YPR

    """

    track = {"lin": [], "rot": []}

    # ra = start_A[3]; pa = start_A[4]; ya = start_A[5]  # Yaw/pitch/Roll for A and B
    # rb = end_B[3]; pb = end_B[4]; yb = end_B[5]

    # A = array(start_A[:3]); B = array(end_B[:3])
    # [vxa, vya, vza, wa] = Rotation(Vector(A_rot[0, 0], A_rot[0, 1], A_rot[0, 2]),
    # Vector(A_rot[1, 0], A_rot[1, 1], A_rot[1, 2]),
    # Vector(A_rot[2, 0], A_rot[2, 1], A_rot[2, 2])).GetQuaternion()  # Quaternion representation of start and end points

    # [vxb, vyb, vzb, wb] = Rotation(Vector(B_rot[0, 0], B_rot[0, 1], B_rot[0, 2]),
    # Vector(B_rot[1, 0], B_rot[1, 1], B_rot[1, 2]),
    # Vector(B_rot[2, 0], B_rot[2, 1], B_rot[2, 2])).GetQuaternion()

    q_a = rotmat2qvec(A_rot)  # [4,]
    q_b = rotmat2qvec(B_rot)  # [4,]

    QA = quaternion(q_a[0], q_a[1], q_a[2], q_a[3])
    QB = quaternion(q_b[0], q_b[1], q_b[2], q_b[3])

    track["lin"] = linear_translation(A_trans, B_trans, T).tolist()
    q = interpolate(QA, QB, T)
    track["rot"] = [q.s] + (q.v).tolist()[0]  # List of quaternion [w x y z]

    return qvec2rotmat(track["rot"]), np.array(track["lin"])
