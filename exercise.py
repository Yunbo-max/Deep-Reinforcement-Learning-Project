import math
robot_state = [0.721156895160675, 1.273087538167239e-12, 0.028953753411769867]
object_state = [ 0.723,    -0.08 ,     0.102421]

square_dx = (object_state[0] - object_state[0]) ** 2
square_dy = (robot_state[1] - object_state[1]) ** 2
square_dz = (robot_state[2] - object_state[2]) ** 2


distance = math.sqrt(square_dx + square_dy + square_dz)

print(distance)
