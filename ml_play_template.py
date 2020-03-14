"""
The template of the main script of the machine learning process
"""

import games.arkanoid.communication as comm
from games.arkanoid.communication import ( \
    SceneInfo, GameStatus, PlatformAction
)

def ml_loop():
    """
    The main loop of the machine learning process

    This loop is run in a separate process, and communicates with the game process.

    Note that the game process won't wait for the ml process to generate the
    GameInstruction. It is possible that the frame of the GameInstruction
    is behind of the current frame in the game process. Try to decrease the fps
    to avoid this situation.
    """

    # === Here is the execution order of the loop === #
    # 1. Put the initialization code here.
    ball_served = False
    prev_ball_coor = (0,0)
    prev_plat_coor = (0,0)

    # 2. Inform the game process that ml process is ready before start the loop.
    comm.ml_ready()

    # 3. Start an endless loop.
    while True:
        # 3.1. Receive the scene information sent from the game process.
        scene_info = comm.get_scene_info()

        # 3.2. If the game is over or passed, the game process will reset
        #      the scene and wait for ml process doing resetting job.
        if scene_info.status == GameStatus.GAME_OVER or \
            scene_info.status == GameStatus.GAME_PASS:
            # Do some stuff if needed
            ball_served = False

            # 3.2.1. Inform the game process that ml process is ready
            comm.ml_ready()
            continue

        # 3.3. Put the code here to handle the scene information
        esti_ball_x = calculate(prev_ball_coor[0], prev_ball_coor[1], scene_info.ball[0], scene_info.ball[1], scene_info.platform[0], scene_info.platform[1])

        # 3.4. Send the instruction for this frame to the game process
        if not ball_served:
            comm.send_instruction(scene_info.frame, PlatformAction.SERVE_TO_LEFT)
            ball_served = True
        else:
            if esti_ball_x > scene_info.platform[0]:
                comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)
            elif esti_ball_x < scene_info.platform[0]:
                comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
            else:
                comm.send_instruction(scene_info.frame, PlatformAction.NONE)

# calculate the possible x posotion when ball is at y=400
def calculate(prev_ball_x, prev_ball_y, cur_ball_x, cur_ball_y, prev_plat, cur_plat):
    # change pivot to center
    prev_ball_x += 2
    cur_ball_x += 2
    prev_plat += 20

    if prev_ball_y > cur_ball_y:
        # use > for the coordination is up side down
        return cur_ball_x -2
    else:
        try:
            m = (cur_ball_y - prev_ball_y)/(cur_ball_x - prev_ball_x)
        except ZeroDivisionError:
            m = (cur_ball_y - prev_ball_y)/(cur_ball_x - prev_ball_x + 1)
        # (y - y0) = m(x - x0)
        # (x - x0) = (y - y0)/m
        # x        = (y - y0)/m + x0
        candidate = (400 - cur_ball_y)/(m if m != 0 else 1) + cur_ball_x -2
        if candidate >= 0 and candidate <= 400:
            return candidate -2
        elif candidate > 400:
            if (candidate/400) % 2 == 1:
                return 400 - candidate % 400 -2
            else:
                return candidate % 400 -2
        else:
            candidate = abs(candidate)
            if (candidate/400) % 2 == 0:
                return candidate % 400 -2
            else:
                return 400 - candidate % 400 -2