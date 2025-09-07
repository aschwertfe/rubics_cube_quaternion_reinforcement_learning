import gymnasium as gym
import numpy as np
import quaternion as quaternion
import itertools
from typing_extensions import List

face_colors = {
    0: (255,0,0), # "red",
    1: (0,255,255), # "turquoise",
    2: (255,255,0), # "yellow",
    3: (50,255,0), # "green",
    4: (0,0,255), # "blue",
    5: (255,0,255), # "violet"
} 

class RubicsCube(gym.Env):
    def __init__(self, mode="quat"):
        if mode == "quat":
            self.init_quat()
        elif mode=="matrix":
            False
        else:
            KeyError("This mode does not exist.")

        self.edge_length = 3
        self.decimal_precision = 8

    def init_quat(self):

        pos_with_center = list(itertools.product([-1, 0, 1], repeat=3)) # generate pos of 3x3x3 cube with center 0,0,0
        pos_floats = [ [0]+list(pos) for pos in pos_with_center]
        pos_floats = sorted( pos_floats, key = lambda p: (p[1],p[2],p[3]) )
        pos_quats = quaternion.as_quat_array(pos_floats)

        # Rotation Cube: carries current positions as quaternions at entries of the starting position
        self.rotation_cube = np.reshape(pos_quats, (3,3,3))
        self.init_cube = self.rotation_cube.copy()

        # Indices of the cube:
        indices = np.array(list(itertools.product([0, 1, 2], repeat=3)))

        ### Depreceated: was for rotate_option2 only
        # Position Cube: carries starting positions as indices at entries of the current position
        # self.position_cube = np.reshape(indices, (3,3,3,3))

        # Mapping quaternion -> Index
        self.quat_to_idx = {pos_quats[i]: indices[i] for i in range(indices.shape[0])}

        # indices for convenient iteration:
        self.indices = [tuple(idx) for idx in indices]

        angle = np.pi/2
        self.cos_half = np.round(np.cos(angle/2), decimals=14)
        self.sin_half = np.round(np.sin(angle/2), decimals=14)

        # define objects... define the assignment of faces
        self.make_parts()

    def step(self, action):

        axis = np.round(action[0])
        depth = np.round(action[1])
        dir = np.round(action[2])
        
        self.rotate(axis,depth,dir, False)

        done, reward, state = self.get_reward()

        return state, reward, done, {}

    def rotate(self, axis, depth, dir, with_faces=False):

        depth = depth-1
        # 1. select plane elements
        mask = np.reshape(np.array([depth-10**(-self.decimal_precision) <= p.imag[axis] and p.imag[axis]<=depth+10**(-self.decimal_precision)
                                     for p in np.reshape( self.rotation_cube, (27) ) ] ),(3,3,3))
        plane = self.rotation_cube[mask]

        # 2. rotate these elements
        q = np.quaternion(self.cos_half,0,0,0) + dir * self.sin_half * np.quaternion(0,0==axis,1==axis,2==axis)
        rotation = lambda p: q * p * q ** (-1)
        plane = rotation(plane)
        self.rotation_cube[mask] = plane

        # 3. rotate the faces of each box, too:
        if with_faces:
            boxes_plane = self.boxes[mask].flatten()
            for box in boxes_plane:
                for face in box.faces:
                    # print("old: ", face)
                    face.set_position(rotation(face.position))
                    new_axis = rotation(face.axis)
                    face.axis = new_axis
                    
        print(self.rotation_cube)

    def get_reward(self):

        # Default reward
        reward = -1
        done = False

        # Initializations
        checks = []
        positions = self.get_positions_fast()
        
        for axis_pos in range(3):
            for depth in [0,2]:
                #print(f"Check plane for axis {axis_pos} with depth {depth}")
                plane = np.take(positions, depth, axis = axis_pos)
                any_equal = []
                for axis_quat in range(3):
                    plane_imag = np.take(plane, axis_quat, axis = -1)
                    check = np.all(np.isclose(plane_imag, plane_imag[-1,-1], atol=10**(-self.decimal_precision+2)))
                    any_equal.append(check)
                checks.append(np.any(any_equal))

        # all planes must be correct:
        if np.all(checks):
            reward += 11
            done=True

        return done, reward, positions
    
    def iterative_rotation(self):
        # update rotation iteratively for visualization
        return
    
    def get_positions_fast(self):
        positions = np.zeros((3,3,3,3))
        for idx in self.indices:
            p = self.rotation_cube[idx]
            positions[idx] = p.imag
        return positions
    
    def get_positions(self, quats: bool = False):

        self.clean_cube(True)

        if quats:
            return self.indices, self.rotation_cube
        else:
            positions = {}
            for idx in self.indices:
                p = self.rotation_cube[idx]
                positions[idx] = p.imag

            self.get_positions_fast()
            
            return self.indices, positions
    
    def get_faces(self):
        # get faces from all boxes
        return [self.boxes[idx].faces for idx in self.incides]
        
    def get_faces(self, idx):
        # get faces only from box of the idx (corner:3, edge:2, middle:1)
        return self.boxes[idx].faces

    def make_parts(self):
        
        self.boxes = np.empty((3,3,3), dtype=object)
        # proceed like that:
            # idx[dim]==-1 # -x face gets color dim*2+(idx[dim]-1)/2 = 0*2+-1=0 from the face_colors
            # idx[dim]==1 # x face gets color dim*2+(idx[dim]-1)/2 = 0*2+1=1 from the face_colors
        for idx in self.indices:
            position_box = self.rotation_cube[idx]
            faces = []
            for dim, i in enumerate(idx):
                pos = i-1 # pos gives information in what direction 1,-1 a front is 
                if pos != 0:
                    face_idx = dim*2+i/2
                    color = face_colors[face_idx]
                    # position = [idx[0],idx[1],idx[2]]
                    axis = dim
                    orientation = pos
                    # shift the faces toward the surface of the cube
                    quat_face = position_box + np.quaternion(0,pos*0.51*(axis==0),pos*0.51*(axis==1),pos*0.51*(axis==2))
                    faces.append(Face(color,quat_face,axis,orientation))
            self.boxes[idx] = Box(position_box, faces, idx)


    def clean_cube(self, with_faces=False):
        # remove the numeric errors from quaternion computation with multiples of pi and cosine etc.
        for idx in self.indices:
            p = self.rotation_cube[idx]
            rounded_p = np.quaternion(
                np.round(p.real, decimals=self.decimal_precision),
                np.round(p.imag[0], decimals=self.decimal_precision),
                np.round(p.imag[1], decimals=self.decimal_precision),
                np.round(p.imag[2], decimals=self.decimal_precision)
            )
            self.rotation_cube[idx] = rounded_p

            if with_faces:
                for face in self.boxes[idx].faces:
                    p = face.position
                    rounded_p = np.quaternion(
                        np.round(p.real, decimals=self.decimal_precision),
                        np.round(p.imag[0], decimals=self.decimal_precision),
                        np.round(p.imag[1], decimals=self.decimal_precision),
                        np.round(p.imag[2], decimals=self.decimal_precision)
                    )
                    face.position = rounded_p
                    a = face.axis
                    rounded_a = np.quaternion(
                        np.round(a.real, decimals=self.decimal_precision),
                        np.round(a.imag[0], decimals=self.decimal_precision),
                        np.round(a.imag[1], decimals=self.decimal_precision),
                        np.round(a.imag[2], decimals=self.decimal_precision)
                    )
                    face.axis = rounded_a

            # self.rotation_cube[tuple(idx)].imag = np.round(p.imag, decimals=1)
            # self.rotation_cube[tuple(idx)].real = np.round(p.real, decimals=1)
        return 

    
class Face:
    def __init__(self,
                 color:tuple[float,float,float],
                 position: quaternion,
                 axis: int,
                 orientation: int):
        
        self.color = color
        self.position = position
        self.axis = np.quaternion(0,0==axis,1==axis,2==axis)
        self.dir = orientation

    def set_position(self, position: quaternion):
        self.position = position

    def get_position(self) -> List[int]:
        return self.position.imag
    
    def get_axis(self) -> List[int]:
        return self.axis.imag
    
    def __str__(self):
        return str(self.axis)

class Box: 
    def __init__(self, position: quaternion, faces: List[Face], idx: List[int]):
        self.position = position
        self.faces = faces
        self.type = len(faces) # 3:corner, 2:edge, 1: middle
        self.idx = idx

    def get_faces(self) -> List[Face]:
        return self.faces

class Quat:
    def __init__():
        False



    '''def rotate_option2(self, axis, depth):
        # axis = {1:ix, 2: jy, 3:kz}
        # depth = {1: , 2: ,3: }

        # process:
        # 0. current indices of plane
        # 1. Position Cube (current indices of plane) -> initial indices for plane
        # 2. Rotation Cube (initial indices for plane) -> quaternions
        # 3. act on qarternions
        # 4. update plane indices from position cube according to new quarternions 

        # 1. Position Cube
        # idx = [range(self.edge_length),range(self.edge_length),range(self.edge_length)]
        # idx[axis] = depth
        # self.position_cube[tuple(idx)]
        plane_pos = np.take(self.position_cube, depth, axis)
        print(plane_pos)
        

        # 2. Rotation Cube


        # 3. Rotate Rotation Cube
        print("Not flat: ", plane_pos)
        plane_pos_flat = np.reshape(plane_pos,(int(plane_pos.size/3),3))
        print("Flat:" , plane_pos_flat)
        indices = list(zip(*plane_pos_flat))
        print(list(zip(*plane_pos_flat)))
        plane_quat = self.rotation_cube[indices[0],indices[1],indices[2]] # rotation missing, but for now see how it translates

        print(plane_quat)

        # 4. update Position Cube
        #print("Plane: ",plane_quat[1])
        #print("Elements:")
        #[[print(x) for x in row] for row in plane_quat]
        new_idxs = np.array([[self.quat_to_idx[x] for x in row] for row in plane_quat])
        plane_pos_flat = plane_pos_flat[new_idxs]

        #print(plane_pos)
        

        print(f"success with axis: {axis} and depth: {depth}")
        
        return True'''


