from viser import ViserServer


class Visualizer(ViserServer):
    
    def __init__(self, cube):

        super().__init__(port=9000)

        self.cube = cube
        self.indices = cube.indices
        self.scale_cube = 1.0
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
            self.boxes[idx].position = self.scale_cube * positions[idx]

            fronts_data = self.cube.get_fronts(idx)
            for front_data in fronts_data:
                self.fronts[f].position = front_data.position
                f += 1
        
        return

    def initial_drawing(self):
 
        self.boxes = {}
        self.fronts = []
        dim_interior = [self.scale_cube*entry for entry in [1.0,1.0,1.0]]
        for idx in self.indices:
            # Draw interior box:
            self.boxes[idx] = self.scene.add_box(
                name=f"/box{idx}",
                dimensions=dim_interior,
                color=self.interior_color, # (100, 255, 100)
                position=(1.0,1.0,1.0),
            )

            # Draw fronts by slim boxes:
            fronts_data = self.cube.get_fronts(idx)
            for front_data in fronts_data:
                dimension = dim_interior
                dimension[front_data.axis] = 0.1*self.scale_cube
                next_front = self.scene.add_box(
                    name=f"/box{idx}",
                    dimensions=dimension,
                    color=front_data.color, # (100, 255, 100)
                    position=front_data.position,
                )
                self.fronts.append(next_front)
        
        return