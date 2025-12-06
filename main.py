import pygame
from pygame.math import Vector2
import math
import random


def cross(a: Vector2, b: Vector2) -> float:
    """ 2D 벡터 외적 """
    return a.x * b.y - a.y * b.x


def cross_sv(s: float, v: Vector2) -> Vector2:
    """ 각속도 x 위치벡터 """
    return Vector2(-s * v.y, s * v.x)


def point_in_convex_polygon(pt: Vector2, verts) -> bool:
    """ 점이 상자 내부에 있는지 체크 """
    sign = None
    n = len(verts)
    for i in range(n):
        a = verts[i]
        b = verts[(i + 1) % n]
        edge = b - a
        to_pt = pt - a
        c = cross(edge, to_pt)
        if c == 0:
            continue
        s = 1 if c > 0 else -1
        if sign is None:
            sign = s
        else:
            if s != sign:
                return False
    return True


class RigidBody:
    def __init__(self, x, y, width, height, mass=1.0, angle=0.0):
        self.position = Vector2(x, y)
        self.velocity = Vector2(0.0, 0.0)

        self.angle = angle
        self.angular_velocity = 0.0

        self.force = Vector2(0.0, 0.0)
        self.torque = 0.0

        self.width = width
        self.height = height

        self.mass = mass
        if mass == 0:
            # static body
            self.inv_mass = 0.0
            self.inertia = float("inf")
            self.inv_inertia = 0.0
        else:
            self.inv_mass = 1.0 / mass
            self.inertia = (mass * (width * width + height * height)) / 12.0
            self.inv_inertia = 1.0 / self.inertia

        self.color = (132, 132, 132)

    def apply_force(self, f: Vector2):
        self.force += f

    def apply_torque(self, tau: float):
        self.torque += tau

    def apply_impulse(self, impulse: Vector2, contact_point: Vector2):
        """ Linear + angular impulse at contact_point """
        if self.inv_mass == 0:
            return

        self.velocity += impulse * self.inv_mass
        r = contact_point - self.position
        self.angular_velocity += cross(r, impulse) * self.inv_inertia

    def world_vertices(self):
        """ 모서리 세계 좌표 계산 """
        hw = self.width / 2.0
        hh = self.height / 2.0

        local = [
            Vector2(-hw, -hh),
            Vector2(hw, -hh),
            Vector2(hw, hh),
            Vector2(-hw, hh),
        ]
        ca = math.cos(self.angle)
        sa = math.sin(self.angle)

        verts = []
        for v in local:
            r = Vector2(
                ca * v.x - sa * v.y,
                sa * v.x + ca * v.y,
            )
            verts.append(self.position + r)
        return verts



class PhysicsWorld:
    def __init__(self, gravity=Vector2(0, 600), ground_y=550):
        self.gravity = gravity
        self.bodies = []
        self.ground_y = ground_y
        self.restitution = 0.3
        self.linear_damping = 0.02
        self.angular_damping = 0.02

    def add_body(self, b: RigidBody):
        self.bodies.append(b)

    def step(self, dt: float):
        # 1. force -> velocity
        for b in self.bodies:
            if b.inv_mass == 0:
                continue

            acc = self.gravity + b.force * b.inv_mass
            b.velocity += acc * dt
            b.angular_velocity += b.torque * b.inv_inertia * dt

            # damping
            b.velocity *= (1.0 - self.linear_damping)
            b.angular_velocity *= (1.0 - self.angular_damping)

            b.force.update(0, 0)
            b.torque = 0.0

        # 2. velocity -> position
        for b in self.bodies:
            b.position += b.velocity * dt
            b.angle += b.angular_velocity * dt

        # 3. ground collision
        for b in self.bodies:
            self.resolve_ground_collision(b)

    def resolve_ground_collision(self, b: RigidBody):
        if b.inv_mass == 0:
            return

        verts = b.world_vertices()
        deepest_vertex = None
        max_pen = 0.0

        # find deepest vertex
        for v in verts:
            pen = v.y - self.ground_y
            if pen > max_pen:
                max_pen = pen
                deepest_vertex = v

        if deepest_vertex is None or max_pen <= 0.0:
            return

        n = Vector2(0, -1)  # upward normal
        r = deepest_vertex - b.position

        v_contact = b.velocity + cross_sv(b.angular_velocity, r)
        v_rel = v_contact.dot(n)

        if v_rel < 0.0:
            e = self.restitution
            rn = cross(r, n)
            denom = b.inv_mass + (rn * rn) * b.inv_inertia
            if denom == 0:
                return

            j = -(1.0 + e) * v_rel / denom
            impulse = j * n

            b.velocity += impulse * b.inv_mass
            b.angular_velocity += cross(r, impulse) * b.inv_inertia

            # correcting position
            percent = 0.8
            slop = 0.01
            correction = max(max_pen - slop, 0.0) * percent
            b.position += correction * n



WIDTH, HEIGHT = 800, 600


def create_random_box():
    x = random.randint(200, 600)
    y = random.randint(20, 120)
    w = random.randint(40, 80)
    h = random.randint(20, 60)
    m = random.uniform(1.0, 5.0)
    a = random.uniform(-0.5, 0.5)
    return RigidBody(x, y, w, h, mass=m, angle=a)


def draw_body(screen, b: RigidBody):
    verts = b.world_vertices()
    pts = [(v.x, v.y) for v in verts]
    pygame.draw.polygon(screen, b.color, pts)
    pygame.draw.line(screen, (255, 255, 255), pts[0], pts[1], 4)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2D Rigid Body Physics Engine")
    clock = pygame.time.Clock()

    world = PhysicsWorld()

    for _ in range(4):
        world.add_body(create_random_box())

    font = pygame.font.SysFont("arial", 18)

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # spawn
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    world.add_body(create_random_box())

            # click -> torque impulse
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse = Vector2(event.pos)
                clicked = None

                for b in world.bodies:
                    if point_in_convex_polygon(mouse, b.world_vertices()):
                        clicked = b
                        break

                if clicked is not None:
                    r = mouse - clicked.position
                    perp = Vector2(-r.y, r.x)
                    if perp.length() == 0:
                        perp = Vector2(1, 0)
                    else:
                        perp = perp.normalize()

                    impulse = perp * 300.0
                    clicked.apply_impulse(impulse, mouse)

        world.step(dt)

        screen.fill((25, 25, 35))

        pygame.draw.line(
            screen,
            (12, 204, 252),
            (0, world.ground_y),
            (WIDTH, world.ground_y),
            3,
        )

        for b in world.bodies:
            draw_body(screen, b)

        screen.blit(font.render("SPACE: Spawn | Click: Torque", True, (255, 255, 255)), (10, 10))
        screen.blit(font.render("Rigid Body + Impulse-based Collision with Torque", True, (255, 255, 255)), (10, 30))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
