import numpy as np
import quaternion as quat
import itertools

front_colors = {
    0: (255,0,0), # "red",
    1: (255,255,255), # "white",
    2: (0,255,255), # "yellow",
    3: (0,255,0), # "green",
    4: (0,0,255), # "blue",
    5: (255,100,100), # "orange"
} 

class Rubics_Cube:
    def __init__(self, mode="quat"):
        if mode == "quat":
            self.init_quat()
        elif mode=="matrix":
            False
        else:
            KeyError("This mode does not exist.")

        self.edge_length = 3
        self.visual_precision = 8

    def init_quat(self):

        pos_with_center = list(itertools.product([-1, 0, 1], repeat=3)) # generate pos of 3x3x3 cube with center 0,0,0
        pos_floats = [ [0]+list(pos) for pos in pos_with_center]
        pos_floats = sorted( pos_floats, key = lambda p: (p[1],p[2],p[3]) )
        pos_quats = quat.as_quat_array(pos_floats)

        # Rotation Cube: carries current positions as quaternions at entries of the starting position
        self.rotation_cube = np.reshape(pos_quats, (3,3,3))
        self.init_cube = self.rotation_cube.copy()

        # Position Cube: carries starting positions as indices at entries of the current position
        indices = np.array(list(itertools.product([0, 1, 2], repeat=3)))
        self.position_cube = np.reshape(indices, (3,3,3,3))

        # Mapping quaternion -> Index
        self.quat_to_idx = {pos_quats[i]: indices[i] for i in range(indices.shape[0])}

        # indices for convenient iteration:
        self.indices = [tuple(idx) for idx in indices]

        angle = np.pi/2
        self.cos_half = np.round(np.cos(angle/2), decimals=14)
        self.sin_half = np.round(np.sin(angle/2), decimals=14)

        # define the assignment of fronts
        self.box_fronts = {}
        self.make_fronts()

    def rotate_option1(self, axis, depth, dir):

        # 1. select plane elements
        mask = np.reshape(np.array([depth-10**(-self.visual_precision) <= p.imag[axis] and p.imag[axis]<=depth+10**(-self.visual_precision)
                                     for p in np.reshape( self.rotation_cube, (27) ) ] ),(3,3,3))
        plane = self.rotation_cube[mask]

        # 2. rotate these elements
        
        q = np.quaternion(self.cos_half,0,0,0) + self.sin_half * np.quaternion(0,0==axis,1==axis,2==axis)
        rotation = lambda p : q * p * q ** (-1)
        plane = rotation(plane)

        self.rotation_cube[mask] = plane

        #print(f"success with axis: {axis} and depth: {depth}")
    
    def get_positions(self, quats: bool = False):

        self.clean_cube()

        if quats:
            return self.indices, self.rotation_cube
        else:
            positions = {}
            for idx in self.indices:
                p = self.rotation_cube[idx]
                positions[idx] = p.imag
            
            return self.indices, positions
    
    def get_fronts(self):
        # get fronts from all boxes
        return self.box_fronts
        
    def get_fronts(self, idx):
        # get fronts only from box of the idx (corner:3, edge:2, middle:1)
        return self.box_fronts[idx]

    def make_fronts(self):
        
        # proceed like that:
            # idx[dim]==-1 # -x front gets color dim*2+(idx[dim]-1)/2 = 0*2+-1=0 from the front_colors
            # idx[dim]==1 # x front gets color dim*2+(idx[dim]-1)/2 = 0*2+1=1 from the front_colors
        for idx in self.indices: 
            fronts = []
            for dim,i in enumerate(idx):
                pos = i-1
                if pos != 0:
                    front_idx = dim*2+i/2
                    color = front_colors[front_idx]
                    quat = self.rotation_cube[idx] #
                    # position = [idx[0],idx[1],idx[2]]
                    axis = dim
                    orientation = pos
                    fronts.append(Front(color,quat,axis,orientation))

            self.box_fronts[idx] = fronts

        ### Depreceated:
        # color_idx = self.rotation_cube[idx].imag[0] + 1
        # color = front_colors[color_idx]
        return 

    def clean_cube(self):
        # remove the numeric errors from quaternion computation with multiples of pi and cosine etc.
        for idx in self.indices:
            p = self.rotation_cube[idx]
            rounded_p = np.quaternion(
                np.round(p.real, decimals=self.visual_precision),
                np.round(p.imag[0], decimals=self.visual_precision),
                np.round(p.imag[1], decimals=self.visual_precision),
                np.round(p.imag[2], decimals=self.visual_precision)
            )
            self.rotation_cube[idx] = rounded_p
            # self.rotation_cube[tuple(idx)].imag = np.round(p.imag, decimals=1)
            # self.rotation_cube[tuple(idx)].real = np.round(p.real, decimals=1)
        return 


    def rotate_option2(self, axis, depth):
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
        
        return True
    
    def assign_planes():
        return 
    
class Front:

    def __init__(self,
                 color:tuple[float,float,float],
                 quat: quat,
                 axis: int,
                 orientation: int):
        
        self.color = color
        self.quat = quat
        self.position = quat.imag
        self.axis = axis
        self.dir = orientation


class quart:
    def __init__():
        False





