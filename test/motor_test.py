from motion_control import (
    init_motion,
    shutdown_motion,
    rotate_camera,
    set_filter,
)

# Ports
MOTOR_PORT = "/dev/ttyUSB0"
FW_PORT    = "/dev/ttyACM0"

# ------------------------------------
# Initialize once
# ------------------------------------
init_motion(MOTOR_PORT, FW_PORT)

# Example usage during imaging:
set_filter(2)
# rotate_camera(0)

# when exiting
shutdown_motion()
