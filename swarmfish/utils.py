from swarmfish.swarm_control import SwarmParams, SwarmAccel, State, NavParams
from swarmfish.obstacles import PolygonObstacle
import yaml
import json
import logging
import numpy as np

log = logging.getLogger(__name__)

def load_params_from_yaml(file_name: str, node: str = 'SwarmParams') -> SwarmParams | None:
    try:
        f = open(file_name)
    except FileNotFoundError:
        log.error(f'File {file_name} not found')
    else:
        try:
            log.info('Parsing yaml config')
            yaml_string = f.read()
            parsed = yaml.load(yaml_string, Loader=yaml.FullLoader)
            if node == 'SwarmParams':
                return SwarmParams(**parsed['Agent']['SwarmParams'])
            elif node == 'NavParams':
                return NavParams(**parsed['Agent']['NavParams'])
            else:
                return None
        except Exception as e:
            log.error(f'Parsing error {e}')
        finally:
            f.close()
    return None

def load_params_from_json(file_name: str, node: str = 'SwarmParams') -> SwarmParams | None:
    try:
        f = open(file_name)
    except FileNotFoundError:
        log.error(f'File {file_name} not found')
    else:
        try:
            log.info('Parsing json config')
            json_string = f.read()
            parsed = json.loads(json_string)
            if node == 'SwarmParams':
                return SwarmParams(**parsed['Agent']['SwarmParams'])
            elif node == 'NavParams':
                return NavParams(**parsed['Agent']['NavParams'])
            else:
                return None
        except Exception as e:
            log.error(f'Parsing error {e}')
        finally:
            f.close()
    return None

def load_polygons_from_json(file_name: str) -> list[PolygonObstacle]:
    ''' load a list of polygons obstacles from json file
        generated with scenebuilder
    '''
    try:
        f = open(file_name)
    except FileNotFoundError:
        log.error(f'File {file_name} not found')
    else:
        try:
            log.info('Parsing json scene')
            json_string = f.read()
            parsed = json.loads(json_string)
            buildings = parsed['scenebuilder']['buildings']
            obstacles = []
            for building in buildings:
                name = building['ID']
                vertices = np.array(building['vertices'])
                obstacle = PolygonObstacle(name=name,
                        vertices=vertices[:,0:2],
                        z_min=0.,
                        z_max=vertices[0][2])
                obstacles.append(obstacle)
            return obstacles
        except Exception as e:
            log.error(f'Parsing error {e}')
        finally:
            f.close()
    return []

def compute_speed_from_accel(cmd: SwarmAccel, state: State, nav_params: NavParams, dt: float, use_heading: bool = False) -> np.ndarray:
    def saturate(cmd, increment, limit_min, limit_max):
        if limit_max is not None and cmd > limit_max and increment > 0.:
            #print(limit_max)
            return limit_max
        elif limit_min is not None and cmd < limit_min and increment < 0.:
            return limit_min
        else:
            return cmd

    ch = np.cos(state.get_course(use_heading))
    sh = np.sin(state.get_course(use_heading))
    if use_heading:
        speed_incr = cmd.a_bx * dt
        speed_cmd = saturate(state.get_speed_2d() + speed_incr, speed_incr, nav_params.speed_min, nav_params.speed_max)
        vz_incr = cmd.a_z * dt
        vz_cmd = saturate(state.get_vz() + vz_incr, vz_incr, nav_params.vz_min, nav_params.vz_max)
        yaw_incr = cmd.a_by / max(state.get_speed_2d(), 0.1)
        yaw_cmd = saturate(yaw_incr, yaw_incr, nav_params.yaw_min, nav_params.yaw_max)
        return np.array([speed_cmd * ch, speed_cmd * sh, vz_cmd, yaw_cmd])
    else:
        # FIXME make a nicer bounding of speed
        a_x = ch * cmd.a_bx - sh * cmd.a_by
        a_y = sh * cmd.a_bx + ch * cmd.a_by
        #print(state.name, a_x, a_y, cmd.a_bx, cmd.a_by, state.get_course(use_heading))
        #vx_cmd = state.speed[0] + a_x * dt
        #vy_cmd = state.speed[1] + a_y * dt
        vx_cmd = saturate(state.speed[0] + a_x * dt, a_x, -nav_params.speed_max, nav_params.speed_max)
        vy_cmd = saturate(state.speed[1] + a_y * dt, a_y, -nav_params.speed_max, nav_params.speed_max)
        vz_incr = cmd.a_z * dt
        vz_cmd = saturate(state.get_vz() + vz_incr, vz_incr, nav_params.vz_min, nav_params.vz_max)
        yaw_cmd = 0. # FIXME track course ?
        return np.array([vx_cmd, vy_cmd, vz_cmd, yaw_cmd])

# data processing functions

def compute_quantification(positions: np.ndarray, velocities: np.ndarray, skip: int = 0) -> np.ndarray:
    ''' Compute quantification (polarization, dispersion, milling) of the group
        and its fluctuations

        positions = [nb_uav, 3, nb_steps] array
        velocities = [nb_uav, 3, nb_steps] array

        #0: dispersion
        #1: polarization
        #2: milling
        #3: fluctuation of dispersion
        #4: fluctuation of polarization
        #5: fluctuation of milling

    '''
    nb_steps = positions.shape[2] - skip
    nb_uav = positions.shape[0]

    pos = positions[:,:,skip:]
    pos_bary = np.mean(pos, axis=0)
    dpos = pos - pos_bary
    vel = velocities[:,:,skip:]
    vel_bary = np.mean(vel, axis=0)
    dvel = vel - vel_bary

    data = np.zeros((nb_steps,3), dtype=np.float64)
    # dispersion
    data[:,0] = np.sqrt(np.sum(np.linalg.norm(dpos, axis=1)**2, axis=0) / nb_uav)
    # polarization
    vel_unit = vel / np.linalg.norm(vel, axis=1).reshape(nb_uav,1,nb_steps)
    data[:,1] = np.linalg.norm(np.sum(vel_unit , axis=0), axis=0) / nb_uav
    # milling
    theta = np.arctan2(dpos[:,1,:], dpos[:,0,:])
    phi = np.arctan2(dvel[:,1,:], dvel[:,0,:])
    data[:,2] = np.fabs(np.sum(np.sin(theta - phi), axis=0) / nb_uav)

    quant = np.zeros(6, dtype=np.float64)
    quant[0] = np.mean(data[:, 0])
    quant[1] = np.mean(data[:, 1])
    quant[2] = np.mean(data[:, 2])
    quant[3] = np.mean(data[:, 0] ** 2) - np.mean(data[:, 0]) ** 2
    quant[4] = np.mean(data[:, 1] ** 2) - np.mean(data[:, 1]) ** 2
    quant[5] = np.mean(data[:, 2] ** 2) - np.mean(data[:, 2]) ** 2
    return quant

def compute_quantification_from_log(filename: str, skip_ratio: float = 0.5, pos_idx: int = 0, vel_idx: int = 3) -> np.ndarray:
    data = np.load(filename)
    traj = data['states']
    size = traj.shape[2]
    return compute_quantification(traj[:,pos_idx:pos_idx+3,:], traj[:,vel_idx:vel_idx+3,:], int(size * skip_ratio))

def compute_quantification_from_dronesim_log(filename: str, skip_ratio: float = 0.5) -> np.ndarray:
    return compute_quantification_from_log(filename, skip_ratio, 0, 10)

def plot_traj_from_log(filename: str):
    import matplotlib.pyplot as plt

    data = np.load(filename)
    traj = data['states']
    nb_uav = traj.shape[0]
    for i in range(nb_uav):
        plt.plot(traj[i,0,:], traj[i,1,:])
    plt.show()

def plot_traj_3D_from_log(filename: str):
    import matplotlib.pyplot as plt

    data = np.load(filename)
    traj = data['states']
    nb_uav = traj.shape[0]
    ax = plt.figure().add_subplot(projection='3d')
    for i in range(nb_uav):
        ax.plot(traj[i,0,:], traj[i,1,:], traj[i,2,:], )
    plt.show()

