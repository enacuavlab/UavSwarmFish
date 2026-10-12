from swarmfish.swarm_control import SwarmParams
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

