from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import numpy as np

class Room:
    def __init__(self,
        top,
        size,
        entryDoorPos,
        exitDoorPos
    ):
        self.top = top
        self.size = size
        self.entryDoorPos = entryDoorPos
        self.exitDoorPos = exitDoorPos

class MultiRoomEnv(MiniGridEnv):
    """
    Environment with multiple rooms (subgoals)
    """

    def __init__(self,
        minNumRooms,
        maxNumRooms,
        maxRoomSize=10,
        max_steps=500,
    ):
        assert minNumRooms > 0
        assert maxNumRooms >= minNumRooms
        assert maxRoomSize >= 4

        self.minNumRooms = minNumRooms
        self.maxNumRooms = maxNumRooms
        self.maxRoomSize = maxRoomSize

        self.rooms = []

        super(MultiRoomEnv, self).__init__(
            grid_size=25,
            max_steps=self.maxNumRooms * max_steps
        )

    def _gen_grid(self, width, height):
        roomList = []

        # Choose a random number of rooms to generate
        numRooms = self._rand_int(self.minNumRooms, self.maxNumRooms+1)

        while len(roomList) < numRooms:
            curRoomList = []

            entryDoorPos = (
                self._rand_int(0, width - 2),
                self._rand_int(0, width - 2)
            )

            # Recursively place the rooms
            self._placeRoom(
                numRooms,
                roomList=curRoomList,
                minSz=4,
                maxSz=self.maxRoomSize,
                entryDoorWall=2,
                entryDoorPos=entryDoorPos
            )

            if len(curRoomList) > len(roomList):
                roomList = curRoomList

        # Store the list of rooms in this environment
        assert len(roomList) > 0
        self.rooms = roomList

        # Create the grid
        self.grid = Grid(width, height)
        wall = Wall()

        prevDoorColor = None

        # For each room
        for idx, room in enumerate(roomList):

            topX, topY = room.top
            sizeX, sizeY = room.size

            # Draw the top and bottom walls
            for i in range(0, sizeX):
                self.grid.set(topX + i, topY, wall)
                self.grid.set(topX + i, topY + sizeY - 1, wall)

            # Draw the left and right walls
            for j in range(0, sizeY):
                self.grid.set(topX, topY + j, wall)
                self.grid.set(topX + sizeX - 1, topY + j, wall)

            # If this isn't the first room, place the entry door
            if idx > 0:
                # Pick a door color different from the previous one
                doorColors = set(COLOR_NAMES)
                if prevDoorColor:
                    doorColors.remove(prevDoorColor)
                # Note: the use of sorting here guarantees determinism,
                # This is needed because Python's set is not deterministic
                doorColor = self._rand_elem(sorted(doorColors))

                entryDoor = Door(doorColor)
                self.grid.set(*room.entryDoorPos, entryDoor)
                prevDoorColor = doorColor

                prevRoom = roomList[idx-1]
                prevRoom.exitDoorPos = room.entryDoorPos

        # Randomize the starting agent position and direction
        self.place_agent(roomList[0].top, roomList[0].size)

        # Place the final goal in the last room
        self.goal_pos = self.place_obj(Goal(), roomList[-1].top, roomList[-1].size)

        self.mission = 'traverse the rooms to get to the goal'

    def _placeRoom(
        self,
        numLeft,
        roomList,
        minSz,
        maxSz,
        entryDoorWall,
        entryDoorPos
    ):
        # Choose the room size randomly
        sizeX = self._rand_int(minSz, maxSz+1)
        sizeY = self._rand_int(minSz, maxSz+1)

        # The first room will be at the door position
        if len(roomList) == 0:
            topX, topY = entryDoorPos
        # Entry on the right
        elif entryDoorWall == 0:
            topX = entryDoorPos[0] - sizeX + 1
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the south
        elif entryDoorWall == 1:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1] - sizeY + 1
        # Entry wall on the left
        elif entryDoorWall == 2:
            topX = entryDoorPos[0]
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the top
        elif entryDoorWall == 3:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1]
        else:
            assert False, entryDoorWall

        # If the room is out of the grid, can't place a room here
        if topX < 0 or topY < 0:
            return False
        if topX + sizeX > self.width or topY + sizeY >= self.height:
            return False

        # If the room intersects with previous rooms, can't place it here
        for room in roomList[:-1]:
            nonOverlap = \
                topX + sizeX < room.top[0] or \
                room.top[0] + room.size[0] <= topX or \
                topY + sizeY < room.top[1] or \
                room.top[1] + room.size[1] <= topY

            if not nonOverlap:
                return False

        # Add this room to the list
        roomList.append(Room(
            (topX, topY),
            (sizeX, sizeY),
            entryDoorPos,
            None
        ))

        # If this was the last room, stop
        if numLeft == 1:
            return True

        # Try placing the next room
        for i in range(0, 8):

            # Pick which wall to place the out door on
            wallSet = set((0, 1, 2, 3))
            wallSet.remove(entryDoorWall)
            exitDoorWall = self._rand_elem(sorted(wallSet))
            nextEntryWall = (exitDoorWall + 2) % 4

            # Pick the exit door position
            # Exit on right wall
            if exitDoorWall == 0:
                exitDoorPos = (
                    topX + sizeX - 1,
                    topY + self._rand_int(1, sizeY - 1)
                )
            # Exit on south wall
            elif exitDoorWall == 1:
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY + sizeY - 1
                )
            # Exit on left wall
            elif exitDoorWall == 2:
                exitDoorPos = (
                    topX,
                    topY + self._rand_int(1, sizeY - 1)
                )
            # Exit on north wall
            elif exitDoorWall == 3:
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY
                )
            else:
                assert False

            # Recursively create the other rooms
            success = self._placeRoom(
                numLeft - 1,
                roomList=roomList,
                minSz=minSz,
                maxSz=maxSz,
                entryDoorWall=nextEntryWall,
                entryDoorPos=exitDoorPos
            )

            if success:
                break

        return True

class MultiRoomEnvN2S4(MultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=2,
            maxNumRooms=2,
            maxRoomSize=4
        )

class MultiRoomEnvN4S5(MultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=4,
            maxNumRooms=4,
            maxRoomSize=5
        )

class MultiRoomEnvN6(MultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=6,
            maxNumRooms=6
        )

class MultiRoomEnvN2(MultiRoomEnv):
    def __init__(self, seeds):
        self.seeds = seeds
        # print(seeds)
        super().__init__(
            minNumRooms=2,
            maxNumRooms=2, # 2, 4
            max_steps=500,
        )

    def reset(self):
        if type(self.seeds)==int:
            self.seed(self.seeds)
        else:
            self.seed(self.seeds[np.random.randint(0,len(self.seeds))])
        return super().reset()

# simple: (10), (8,16)
# complex: (10), (2,8)
class MultiRoomEnvN20(MultiRoomEnvN2):
    def __init__(self):
        super().__init__(
            seeds=list(range(8, 16))
        )

class MultiRoomEnvN2v0(MultiRoomEnvN2):
    def __init__(self):
        super().__init__(
            seeds=0
        )

class MultiRoomEnvN2v1(MultiRoomEnvN2):
    def __init__(self):
        super().__init__(
            seeds=1
        )

class MultiRoomEnvN2v2(MultiRoomEnvN2):
    def __init__(self):
        super().__init__(
            seeds=2
        )

class MultiRoomEnvN2v3(MultiRoomEnvN2):
    def __init__(self):
        super().__init__(
            seeds=3
        )

class MultiRoomEnvN2v4(MultiRoomEnvN2):
    def __init__(self):
        super().__init__(
            seeds=4
        )

class MultiRoomEnvN2v5(MultiRoomEnvN2):
    def __init__(self):
        super().__init__(
            seeds=5
        )

class MultiRoomEnvN2v6(MultiRoomEnvN2):
    def __init__(self):
        super().__init__(
            seeds=6
        )

class MultiRoomEnvN2v7(MultiRoomEnvN2):
    def __init__(self):
        super().__init__(
            seeds=7
        )

class MultiRoomEnvN2v8(MultiRoomEnvN2):
    def __init__(self):
        super().__init__(
            seeds=8
        )

class MultiRoomEnvN2v9(MultiRoomEnvN2):
    def __init__(self):
        super().__init__(
            seeds=9
        )

class MultiRoomEnvN2v10(MultiRoomEnvN2):
    def __init__(self):
        super().__init__(
            seeds=10
        )

class MultiRoomEnvN2v11(MultiRoomEnvN2):
    def __init__(self):
        super().__init__(
            seeds=11
        )

class MultiRoomEnvN2v12(MultiRoomEnvN2):
    def __init__(self):
        super().__init__(
            seeds=12
        )

class MultiRoomEnvN2v13(MultiRoomEnvN2):
    def __init__(self):
        super().__init__(
            seeds=13
        )

class MultiRoomEnvN2v14(MultiRoomEnvN2):
    def __init__(self):
        super().__init__(
            seeds=14
        )

class MultiRoomEnvN2v15(MultiRoomEnvN2):
    def __init__(self):
        super().__init__(
            seeds=15
        )

class MultiRoomEnvN2v16(MultiRoomEnvN2):
    def __init__(self):
        super().__init__(
            seeds=16
        )

class MultiRoomEnvN2v17(MultiRoomEnvN2):
    def __init__(self):
        super().__init__(
            seeds=17
        )

class MultiRoomEnvN2v18(MultiRoomEnvN2):
    def __init__(self):
        super().__init__(
            seeds=18
        )

class MultiRoomEnvN2v19(MultiRoomEnvN2):
    def __init__(self):
        super().__init__(
            seeds=19
        )

register(
    id='MiniGrid-MultiRoom-N2-S4-v0',
    entry_point='gym_minigrid.envs:MultiRoomEnvN2S4'
)

register(
    id='MiniGrid-MultiRoom-N4-S5-v0',
    entry_point='gym_minigrid.envs:MultiRoomEnvN4S5'
)

register(
    id='MiniGrid-MultiRoom-N6-v0',
    entry_point='gym_minigrid.envs:MultiRoomEnvN6'
)

register(
    id='MiniGrid-MultiRoom-N2-v00',
    entry_point='gym_minigrid.envs:MultiRoomEnvN20'
)

for i in range(20):
    register(
        id='MiniGrid-MultiRoom-N2-v' + str(i),
        entry_point='gym_minigrid.envs:MultiRoomEnvN2v' + str(i)
    )
