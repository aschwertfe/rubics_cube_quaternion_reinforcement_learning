from viser import ViserServer
import numpy as np


class Visualizer(ViserServer):
    
    def __init__(self, cube):

        super().__init__(port=9000)

        self.cube = cube
        self.indices = cube.indices
        self.scale_cube = 1.0
        self.face_thickness = 0.02
        self.interior_color = (150,150,150)

        self.initial_drawing()
        self.update_position()
        print("boxes created")

        # box_height = self.gui.add_slider(
        #     "Box height", min=-1.0, max=1.0, step=0.1, initial_value=0.0
        # )
        # @box_height.on_update
        # def _(_):
        #     for idx in indices:
        #         self.boxes[idx].position = (1.0, 0.0, box_height.value)
        # print("Event decoator is declared")

        print("Server running")

    def update_position(self):
        
        indices, positions = self.cube.get_positions()

        f = 0
        for idx in indices:
            # Update boxes
            self.boxes[idx].position = self.scale_cube * positions[idx]

            # Update faces
            faces_data = self.cube.boxes[idx].get_faces()
            for k,face_data in enumerate(faces_data):
                #print(dir(self.scene))
                #print("\n\n")
                #print(dir(faces_data[f]))
                self.scene.remove_by_name("/face{k}of{idx}")
                dimension = [1.0,1.0,1.0] - np.abs(0.98*face_data.get_axis())
                #print(dimension)
                #print(0.92*face_data.get_axis())
                dimension = [ entry * self.scale_cube for entry in dimension]
                next_face = self.scene.add_box(
                    name=f"/face{k}of{idx}",
                    dimensions=dimension,
                    color=face_data.color, # (100, 255, 100)
                    position=face_data.get_position(),
                )
                self.faces[f] = next_face
                # self.faces[f].position = face_data.get_position()
                # dimension = [1.0,1.0,1.0] - 0.92*face_data.get_axis()
                # print(dimension)
                # dimension = [ entry * self.scale_cube for entry in dimension]
                # self.faces[f].dimension = dimension
                f += 1
        
        return

    def initial_drawing(self):
 
        self.boxes = {}
        self.faces = []
        dim_interior = [self.scale_cube*entry for entry in [1.0,1.0,1.0]]
        for idx in self.indices:
            # Draw interior box:
            self.boxes[idx] = self.scene.add_box(
                name=f"/box{idx}",
                dimensions=dim_interior,
                color=self.interior_color, # (100, 255, 100)
                position=(1.0,1.0,1.0),
            )

            # Draw faces by slim boxes:
            faces_data = self.cube.boxes[idx].get_faces()
            for k,face_data in enumerate(faces_data):
                dimension = [1.0,1.0,1.0] - 0.98*face_data.get_axis()
                dimension = [ entry * self.scale_cube for entry in dimension]
                next_face = self.scene.add_box(
                    name=f"/face{k}of{idx}",
                    dimensions=dimension,
                    color=face_data.color, # (100, 255, 100)
                    position=face_data.get_position(),
                )
                self.faces.append(next_face)
        
        return