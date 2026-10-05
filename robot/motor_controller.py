"""
4-Motor Differential-Drive Controller

Each side (left/right) has two motors -- front and back -- wired to an
L298N (or equivalent, e.g. TB6612) H-bridge driver, the cheapest, most
common setup for a DIY wheeled-robot kit. Front and back motors on the
same side always drive together, so control-wise this is still just
left/right differential steering -- same interface as the 2-motor version.

Wiring (BCM GPIO numbering -- edit the *_PIN constants below if yours
differs; matches robot/test.py, which verified these against the actual
hardware):

    Pi GPIO 17 --> L298N IN1  \\
    Pi GPIO 13 --> L298N IN2   }  front-left motor
    Pi GPIO 19 --> L298N IN1  \\
    Pi GPIO 23 --> L298N IN2   }  back-left motor
    Pi GPIO 26 --> L298N IN3  \\
    Pi GPIO 25 --> L298N IN4   }  front-right motor
    Pi GPIO 20 --> L298N IN3  \\
    Pi GPIO 21 --> L298N IN4   }  back-right motor
    Pi GND     --> L298N GND     (common ground with the Pi -- required)
    L298N ENA, ENB            --> jumper both to +5V ("always enabled");
                                   gpiozero drives speed+direction through
                                   IN1-4 with software PWM, so the enable
                                   pins don't need to be wired to the Pi
    L298N 12V, GND (power in) --> motor battery pack, NOT the Pi's 5V rail
                                   (the Pi cannot supply motor current)

If `gpiozero` isn't installed/importable (e.g. developing on a laptop with
no GPIO hardware) or `dry_run=True` is passed, every command just prints
what it *would* do instead of touching hardware -- so the rest of the
pipeline (camera, gesture recognition, temporal smoothing) can still be
exercised on a dev machine.
"""
try:
    from gpiozero import Motor
    _GPIOZERO_AVAILABLE = True
except (ImportError, NotImplementedError):
    _GPIOZERO_AVAILABLE = False

FRONT_LEFT_FORWARD_PIN = 17
FRONT_LEFT_BACKWARD_PIN = 12
BACK_LEFT_FORWARD_PIN = 19
BACK_LEFT_BACKWARD_PIN = 23

FRONT_RIGHT_FORWARD_PIN = 26
FRONT_RIGHT_BACKWARD_PIN = 25
BACK_RIGHT_FORWARD_PIN = 20
BACK_RIGHT_BACKWARD_PIN = 21


class MotorController:
    def __init__(self, speed: float = 0.7, turn_speed: float = None, dry_run: bool = False,
                 left_trim: float = 1.0, right_trim: float = 1.0):
        """
        left_trim / right_trim: per-side multipliers (0.0-1.0) applied on
        top of `speed`, to compensate for one side being physically
        stronger than the other (common with cheap DC motors -- no two are
        perfectly matched). Trim the STRONGER side down rather than trying
        to trim the weaker side up, since speed is already capped at 1.0.

        e.g. left side visibly stronger -> left_trim=0.8 (right stays 1.0)
        so `forward()` drives the left motors at 0.8x, right at 1.0x.

        turn_speed: separate, usually lower, speed used for turn_left/
        turn_right -- pivoting in place is more prone to tipping/skidding
        than driving straight, so it's common to want a gentler turn than
        the forward/backward speed. Defaults to `speed` if not given.
        """
        if not 0.0 < speed <= 1.0:
            raise ValueError("speed must be in (0.0, 1.0]")
        if turn_speed is None:
            turn_speed = speed
        if not 0.0 < turn_speed <= 1.0:
            raise ValueError("turn_speed must be in (0.0, 1.0]")
        if not 0.0 < left_trim <= 1.0 or not 0.0 < right_trim <= 1.0:
            raise ValueError("left_trim/right_trim must be in (0.0, 1.0]")
        self.speed = speed
        self.turn_speed = turn_speed
        self.left_trim = left_trim
        self.right_trim = right_trim
        self.dry_run = dry_run or not _GPIOZERO_AVAILABLE

        if self.dry_run:
            if not _GPIOZERO_AVAILABLE and not dry_run:
                print("[MotorController] gpiozero not available -- running in dry-run mode (printing only)")
            self._front_left = self._back_left = None
            self._front_right = self._back_right = None
        else:
            self._front_left = Motor(forward=FRONT_LEFT_FORWARD_PIN, backward=FRONT_LEFT_BACKWARD_PIN)
            self._back_left = Motor(forward=BACK_LEFT_FORWARD_PIN, backward=BACK_LEFT_BACKWARD_PIN)
            self._front_right = Motor(forward=FRONT_RIGHT_FORWARD_PIN, backward=FRONT_RIGHT_BACKWARD_PIN)
            self._back_right = Motor(forward=BACK_RIGHT_FORWARD_PIN, backward=BACK_RIGHT_BACKWARD_PIN)

    def forward(self):
        self._drive(self.speed, self.speed, "FORWARD")

    def backward(self):
        self._drive(-self.speed, -self.speed, "BACKWARD")

    def turn_left(self):
        """Pivot in place: left side back, right side forward."""
        self._drive(-self.turn_speed, self.turn_speed, "TURN_LEFT")

    def turn_right(self):
        self._drive(self.turn_speed, -self.turn_speed, "TURN_RIGHT")

    def stop(self):
        self._drive(0.0, 0.0, "STOP")

    def _drive(self, left_val: float, right_val: float, label: str):
        left_val *= self.left_trim
        right_val *= self.right_trim
        if self.dry_run:
            print(f"    [motors] {label}  (left={left_val:+.2f}, right={right_val:+.2f})")
            return
        self._set_motor(self._front_left, left_val)
        self._set_motor(self._back_left, left_val)
        self._set_motor(self._front_right, right_val)
        self._set_motor(self._back_right, right_val)

    @staticmethod
    def _set_motor(motor, value: float):
        if value > 0:
            motor.forward(value)
        elif value < 0:
            motor.backward(-value)
        else:
            motor.stop()

    def close(self):
        self.stop()
        if not self.dry_run:
            self._front_left.close()
            self._back_left.close()
            self._front_right.close()
            self._back_right.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
