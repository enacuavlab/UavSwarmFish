import numpy as np
import math
from dataclasses import dataclass

@dataclass
class SwarmParams:
    # wall
    y_w: float
    l_w: float
    e1_w: float
    e2_w: float
    y_z_w: float
    dz_w: float
    # attraction
    y_att: float
    l_att: float
    d0_att: float
    a_att: float
    b1_att: float
    b2_att: float
    # alignment
    y_ali: float
    l_ali: float
    d0_ali: float
    a_ali: float
    b1_ali: float
    b2_ali: float
    # speed
    y_acc: float
    l_acc: float
    d0_v: float
    # vertical
    y_z: float
    l_z: float
    a_z: float
    d0_z: float
    sigma_z: float
    # nav
    y_nav: float
    y_z_nav: float
    y_vz_nav: float
    # intruders
    y_intruder: float
    y_z_intruder: float
    l_intruder: float
    # obstacles
    y_obs: float
    t_obs: float
    e1_obs: float
    e2_obs: float

    use_heading: bool = False

@dataclass
class NavParams:
    # collective motion parameters
    max_velocity: float = None
    min_velocity: float = None
    zmax: float = None
    zmin: float = None


def wrap_to_pi(angle) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass
class State:
    pos: np.ndarray
    speed: np.ndarray
    heading: float
    timestamp: float
    name: str = "none"

    def get_distance_2d(self, other: 'State') -> float:
        return np.linalg.norm(self.pos[0:2] - other.pos[0:2])

    def get_distance_3d(self, other: 'State') -> float:
        return np.linalg.norm(self.pos - other.pos)

    def get_distance_coupled(self, other: 'State', params: SwarmParams) -> float:
        return np.sqrt((self.pos[0] - other.pos[0])**2 + (self.pos[1] - other.pos[1])**2 + ((self.pos[2] - other.pos[2]) / params.sigma_z)**2)

    def get_speed_2d(self) -> float:
        return np.linalg.norm(self.speed[0:2])

    def get_speed_3d(self) -> float:
        return np.linalg.norm(self.speed)

    def get_vz(self) -> float:
        return self.speed[2]

    def get_course(self, use_heading: bool = False) -> float:
        if use_heading:
            return self.heading
        return math.atan2(self.speed[1], self.speed[0])

    def get_course_diff(self, other: 'State', use_heading: bool = False) -> float:
        diff = other.get_course(use_heading) - self.get_course(use_heading)
        return wrap_to_pi(diff)

    def get_viewing_angle(self, other: 'State', use_heading: bool = False) -> float:
        ''' Viewing angle in 2D
        '''
        diff = other.pos[0:2] - self.pos[0:2]
        direction = math.atan2(diff[1], diff[0])
        viewing_angle = direction - self.get_course(use_heading)
        return wrap_to_pi(viewing_angle)

    def get_elevation_angle(self, other: 'State') -> float:
        ''' Vertical viewing angle
        '''
        dz = other.pos[2] - self.pos[2]
        dh = self.get_distance_2d(other)
        elevation = math.atan2(dz, dh)
        return elevation

    def __str__(self):
        out = f'State {self.name}: '
        out += f'pos= {self.pos[0]:.2f}, {self.pos[1]:.2f}, {self.pos[2]:.2f} | '
        out += f'speed= {self.speed[0]:.2f}, {self.speed[1]:.2f}, {self.speed[2]:.2f} | '
        out += f'vel= {self.get_speed_3d():.2f} course= {np.degrees(self.get_course()):0.2f} '
        out += f'heading= {np.degrees(self.heading):.2f} | '
        out += f'time= {self.timestamp:.3f}'
        return out

@dataclass
class SwarmCommands:
    delta_course: float = 0.
    delta_speed: float = 0.
    delta_vz: float = 0.

    def __add__(self, other):
        return SwarmCommands(
                delta_course = wrap_to_pi(self.delta_course + other.delta_course),
                delta_speed = self.delta_speed + other.delta_speed,
                delta_vz = self.delta_vz + other.delta_vz)

    def __iadd__(self, other):
        self.delta_course = wrap_to_pi(self.delta_course + other.delta_course)
        self.delta_speed += other.delta_speed
        self.delta_vz += other.delta_vz
        return SwarmCommands(self.delta_course, self.delta_speed, self.delta_vz)

    def __str__(self):
        out = f'Cmd: '
        out += f'd_course= {np.degrees(self.delta_course):0.2f} '
        out += f'd_speed= {self.delta_speed:0.2f} '
        out += f'd_vz= {self.delta_vz:0.2f}'
        return out


##
## interaction functions
##

def interaction_wall(agent: State, params: SwarmParams, wall: tuple[float, float] = None, z_min: float = None, z_max: float = None) -> SwarmCommands:
    ''' Interaction with a wall defined by a distance and relative orientation
        and with floor and ceiling
    '''
    cmd = SwarmCommands()
    if wall is not None:
        dist, angle = wall
        fw = math.exp(-(dist / params.l_w)**2)
        ow = params.e1_w * math.cos(angle) + params.e2_w * math.cos(2. * angle)
        cmd.delta_course = params.y_w * math.sin(angle) * (1. + ow) * fw
    if z_min is not None:
        dz = agent.pos[2] - z_min
        cmd.delta_vz += 2. * params.y_z_w / (1. + math.exp((dz - params.dz_w) / params.dz_w))
    if z_max is not None:
        dz = z_max - agent.pos[2]
        cmd.delta_vz +=  - 2. * params.y_z_w / (1. + math.exp((dz - params.dz_w) / params.dz_w))
    #TODO a speed variation to slow down on obstacles ?
    return cmd

def interaction_social(agent: State, params: SwarmParams, other: State, dt: float, r_w: float = 100.) -> SwarmCommands:
    ''' Social interaction of alignment, attraction, speed and vertical speed
        with wall distance r_w for attenuation
    '''
    dij = agent.get_distance_3d(other)
    if dij < 1e-3:
        return SwarmCommands() # too close to do something, better avoid division by zero
    dij_c = agent.get_distance_coupled(other, params)          # distance between agents
    dphi = agent.get_course_diff(other, params.use_heading)  # course/heading difference
    psi = agent.get_viewing_angle(other, params.use_heading) # viewing angle
    c_psi = math.cos(psi)
    s_psi = math.sin(psi)
    print(f"attrib {agent.name} <-> {other.name} | dij={dij:0.2f}  dphi={np.degrees(dphi):0.2f} psi={np.degrees(psi):0.2f}")

    attenuation = 1. - math.exp(-(r_w/params.l_w)**2)        # wall attenuation
    # alignment
    f_ali = params.y_ali * (dij_c / params.d0_ali + 1.) * math.exp(-(dij_c / params.l_ali)**2) * math.sin(dphi) * (1. + params.b1_ali * c_psi) * attenuation
    # attraction
    f_att = params.y_att * ((dij_c / params.d0_att - 1.) / (1. + (dij_c / params.l_att)**2)) * (1. + params.a_att * c_psi) * attenuation

    e_para = np.array([c_psi, s_psi, 0.])
    e_perp = np.array([-s_psi, c_psi, 0.])
    e_z = np.array([0., 0., 1.])
    e_ij = (other.pos - agent.pos) / dij
    f_social = f_ali + f_att

    dyaw = f_social * np.dot(e_perp, e_ij) * (dt / max(agent.get_speed_2d(), 0.1))
    dv = f_social * np.dot(e_para, e_ij) * dt
    dvz = f_social * np.dot(e_z, e_ij) * dt
    #print(f"   att {params.d0_att:.2f} {dij:0.2f} | {(dij / params.d0_att - 1.):0.4f} | f(dij)={f_att:0.2f} o(psi)={o_att:0.2f} e(dphi)={e_att:0.2f}")
    #print(f"   ali {params.d0_ali:.2f} {dij:0.2f} | {(dij / params.d0_ali + 1.):0.4f} | f(dij)={f_ali:0.2f} o(dphi)={o_ali:0.2f} e(psi)={e_ali:0.2f}")
    #print(f'   dyaw_att {np.degrees(dyaw_att):0.2f} | dyaw_ali {np.degrees(dyaw_ali):0.2f} | social {np.degrees(dyaw_att+dyaw_ali):.2f}')
    print(f"  f_ali {f_ali:0.3f}, f_att {f_att:0.3f} | dyaw {np.degrees(dyaw):0.2f} | dv {dv:0.2f} | dvz {dvz:0.2f}")

    return SwarmCommands(delta_course=dyaw, delta_speed=dv, delta_vz=dvz)

def interaction_nav(agent: State, params: SwarmParams, direction: float = None, altitude: float = None) -> tuple[float, float]:
    ''' Navigation interaction with a specified direction and altitude
        also add the vertical speed damping
    '''
    cmd = SwarmCommands()
    if direction is not None:# and agent.get_speed_2d() > 0.5:
        cmd.delta_course = params.y_nav * math.atan(direction - agent.get_course(params.use_heading))
    if altitude is not None:
        cmd.delta_vz += -params.y_z_nav * math.tanh((agent.pos[2] - altitude) / params.a_z)
    # add vertical speed damping
    speed = agent.get_speed_3d()
    if speed > 0.1:
        cmd.delta_vz += - params.y_vz_nav * agent.get_vz() / speed
    #TODO speed attraction to setpoint
    return cmd

def interaction_intruder(agent: State, params: SwarmParams, other: State) -> SwarmCommands:
    ''' Interaction of repulsion with an intruder
    '''
    dij = agent.get_distance_coupled(other, params)
    dphi = agent.get_course_diff(other, params.use_heading)
    psi = agent.get_viewing_angle(other, params.use_heading)
    psi_int = other.get_viewing_angle(agent, params.use_heading)
    speed = agent.get_speed_2d()
    speed_int = other.get_speed_2d()
    relative_vel_ratio = (speed * math.cos(psi) + speed_int * math.cos(psi_int)) / max(speed + speed_int, 0.1) 
    #print(f'{agent.name} -> {other.name}: {relative_vel_ratio:0.2f} | {np.degrees(psi):0.2f} {np.degrees(psi_int):0.2f}')
    #FIXME use dphi or psi for even function ?
    #dyaw = -params.y_intruder * math.exp(-(dij / params.l_intruder)**2) * (1. + params.e1_w * math.cos(psi)) * math.sin(dphi)
    F_rep = params.y_intruder * math.exp(-(dij / params.l_intruder)**2) * (1. + relative_vel_ratio)
    dyaw = - F_rep * math.sin(psi) / max(speed, 0.1)
    dv = - F_rep * math.cos(psi)
    dz = agent.pos[2] - other.pos[2]
    dvz = params.y_z_intruder * math.tanh(dz / params.a_z) * math.exp(-(dij / params.l_z)**2)
    return SwarmCommands(delta_course=dyaw, delta_speed=dv, delta_vz=dvz)

def interaction_obstacle(agent: State, params: SwarmParams, obstacle: tuple[float, float] = None) -> SwarmCommands:
    ''' Interaction with an obstacle defined by a distance and relative orientation
    '''
    cmd = SwarmCommands()
    if obstacle is None:
        return cmd

    dist, angle = obstacle
    speed = agent.get_speed_2d()
    collision_speed = speed * math.cos(angle)
    #print(f'obs speed {speed:.2f}, {dist:.2f}, {collision_speed:.2}')
    if collision_speed < 0.1:
        return cmd # speed is too small or negative
    tau = dist / collision_speed
    fw = math.exp(-(tau / params.t_obs)**2)
    ow = params.e1_obs * math.cos(angle) + params.e2_obs * math.cos(2. * angle)
    F_obs = params.y_obs * (1. + ow) * fw
    cmd.delta_course = F_obs * math.sin(angle) / max(speed, 0.1)
    cmd.delta_speed = - F_obs * math.cos(angle)
    #print(f'obs delta {np.degrees(cmd.delta_course):.2f}, {tau}, {fw}, {ow}')
    return cmd


def compute_interactions(
        agent: State,
        params: SwarmParams,
        neighbors: list[State] = [],
        nb_influent: int = 1,
        direction: float = None,
        altitude: float = None,
        wall: tuple[float, float] = None,
        z_min: float = None,
        z_max: float = None,
        obstacles: list[(float, float)] = [],
        intruders: list[State] = []) -> tuple[SwarmCommands, list[tuple[str, str]]]:
    # closest wall distance
    closest_wall = np.min(np.array([ w[0] for w in [wall] + obstacles if w is not None ]), initial=100.)
    # social
    social = []
    for neighbor in neighbors:
        social.append((agent.name, neighbor.name, interaction_social(agent, params, neighbor, dt=1., r_w=closest_wall)))
    speed_2d = agent.get_speed_2d()
    def compute_influence(x):
        return math.sqrt(x.delta_speed**2 + (x.delta_course * speed_2d)**2 + x.delta_vz**2)
        #return np.fabs(x.delta_course)
    influential = sorted(social, key=lambda s: compute_influence(s[2]), reverse=True)
    d_social = SwarmCommands()
    most_influentials = []
    for i, s in enumerate(influential):
        if i == nb_influent:
            break
        #print(f"Most influential for {s[0]}: {s[1]}")
        most_influentials.append((s[0], s[1]))
        d_social += s[2]
    # wall and borders
    d_borders = interaction_wall(agent, params, wall, z_min, z_max)
    # navigation
    d_nav = interaction_nav(agent, params, direction, altitude)
    # intruders
    d_intruders = SwarmCommands()
    for intruder in intruders:
        d_intruders += interaction_intruder(agent, params, intruder)
    # obstacles
    d_obstacles = SwarmCommands()
    for obstacle in obstacles:
        d_obstacles += interaction_obstacle(agent, params, obstacle)

    cmd = d_social + d_borders + d_nav + d_intruders + d_obstacles
    #print(f'  delta_yaw {np.degrees(delta_yaw):.2f} | s={np.degrees(dyaw_s):.2f}, w={np.degrees(dyaw_w):.2f}, n={np.degrees(dyaw_nav):.2f}, i={np.degrees(dyaw_i):.2f}')
    #print(f'  delta_vz {delta_vz:.2f} | s={dvz_s:.2f}, w={dvz_w:.2f}, n={dvz_nav:.2f}, i={dvz_i:.2f}')
    return cmd, most_influentials


if __name__ == '__main__':
    # run tests
    pass

