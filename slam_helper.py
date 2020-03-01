import numpy as np
import load_data
import p2_utils
from scipy.special import logsumexp


class LIDAR_SENSOR:
    def __init__(self, lfilename, jfilename):
        self.lfilename = lfilename
        self.lidar = load_data.get_lidar(self.lfilename)

        self.jfilename = jfilename
        self.joint = load_data.get_joint(self.jfilename)
        self.j_t = self.joint['ts']
        self.j_ang = self.joint['head_angles']
        self.j_n = self.j_ang[0]
        self.j_h = self.j_ang[1]
        self.pose = np.array([0.0, 0.0, 0.0])
        self.scan = None

    def dead_reckoning(self):
        pose = [[] for _ in range(len(self.lidar) + 1)]
        curp = [0.0, 0.0, 0.0]
        pose[0].extend(curp)

        for i in range(0, len(self.lidar)):
            deltapose = self.lidar[i]['delta_pose']
            dx, dy, dt = deltapose[0]
            curp[0] += dx
            curp[1] += dy
            curp[2] += dt
            pose[i + 1].extend(curp)

        pose_arr = np.array(pose)
        return pose_arr

    def pose_to_world(self, deltapose):
        self.pose += deltapose[0]
        return self.pose

    def transformation_matrices(self, R, p):
        last_row = np.array([[0.0, 0.0, 0.0, 1.0]])
        T = np.hstack((R, p))
        T = np.vstack((T, last_row))
        return T

    def scan_to_world(self, pose, t):
        HpL = np.array([[0.0], [0.0], [0.15]])
        BpH = np.array([[0.0], [0.0], [0.33]])
        HRL = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        HTL = self.transformation_matrices(HRL, HpL)

        lidar_scan = self.lidar[t]['scan']
        theta = np.array([np.arange(-135, 135.25, 0.25) * np.pi / 180.])

        indValid = np.logical_and((lidar_scan < 30), (lidar_scan > 0.1))
        lidar_scan = lidar_scan[indValid]
        theta = theta[indValid]

        scan_x = lidar_scan * np.cos(theta)
        scan_y = lidar_scan * np.sin(theta)
        scan_z = np.zeros((scan_x.shape[0]))


        homo = np.ones((scan_x.shape[0]))
        hL = np.vstack((scan_x, scan_y))
        hL = np.vstack((hL, scan_z))
        hL = np.vstack((hL, homo))

        l_t = self.lidar[t]['t']
        t_diff = self.j_t - l_t[0][0]
        t_diff = t_diff.flatten()
        ind = np.where(t_diff > 0.0)
        id = ind[0][0] - 1


        psi = self.j_n[id]
        phi = self.j_h[id]
        R_n = np.array([[np.cos(psi), -np.sin(psi), 0.0], [np.sin(psi), np.cos(psi), 0.0], [0.0, 0.0, 1.0]])
        R_h = np.array([[np.cos(phi), 0.0, np.sin(phi)], [0.0, 1.0, 0.0], [-np.sin(phi), 0.0, np.cos(phi)]])
        BRH = np.matmul(R_n, R_h)
        BTH = self.transformation_matrices(BRH, BpH)

        th = pose[2]
        WRB = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
        WpB = np.array([[pose[0]], [pose[1]], [0.93]])
        WTB = self.transformation_matrices(WRB, WpB)

        hH = np.matmul(HTL, hL)

        hB = np.matmul(BTH, hH)

        hW = np.matmul(WTB, hB)
        assert int(np.sum(hW, axis=1)[3]) == hW.shape[1]

        pW = np.delete(hW, 3, 0)
        idValid = np.logical_and(True, (pW[2] > 0.1))
        self.scan = np.vstack((pW[0][idValid], pW[1][idValid]))
        return self.scan


class MAP:
    def __init__(self, size, res):
        self.res = res
        self.size = int(np.ceil(size * 2 / self.res) + 1)
        self.grid = np.zeros((self.size, self.size), dtype=np.float16)
        self.v_grid_occ = np.vectorize(self.grid_occ)
        self.v_grid_free = np.vectorize(self.grid_free)
        self.v_bresenham = np.vectorize(self.bresenham)
        self.xmin = -40
        self.xmax = 40
        self.ymin = -40
        self.ymax = 40

    def grid_occ(self, x, y):
        # print("In grid occ Map")
        # print(x, y)
        if self.grid[x, y] < 30:
            self.grid[x, y] = self.grid[x, y] + np.log(4)

    def grid_free(self, x, y):
        # print("In grid free Map")
        # print(x, y)
        if self.grid[x, y] > -30:
        # print(self.grid[x, y])
            self.grid[x, y] = self.grid[x, y] - np.log(4)
        # print(self.grid[x, y])

    def bresenham(self, a, b, c, d):


        cells = p2_utils.bresenham2D(c, d, a, b).astype(int)
        self.grid_occ(cells[0, -1], cells[1, -1])
        cells = cells[:, :-1]

        self.v_grid_free(cells[0], cells[1])
        # print("bresenham")

    def update_map(self, scan, pose):


        self.v_bresenham(scan[0, :], scan[1, :], pose[0], pose[1])





class PARTICLE:
    def __init__(self, num):
        self.num = num
        self.weights = np.ones(self.num)/self.num
        self.pose = np.zeros((self.num, 3))
        self.cov = [0.005, 0.005, 0.0005]
        self.Nth = 0.1 * self.num
        self.traj = []

    def best_particle(self):
        pose = self.pose[np.argmax(self.weights), :]
        self.traj.append(np.copy(pose))
        return pose

    def particle_pose(self, pose, res):
        for n in range(self.num):
            pose[n][0] = np.ceil((pose[n][0] / res)).astype(np.int16) + 800
            pose[n][1] = np.ceil((pose[n][1] / res)).astype(np.int16) + 800
        return pose



    def particle_predict(self, n, deltapose):
        self.pose[n] += deltapose
        noise = np.random.multivariate_normal([0, 0, 0], np.diag(self.cov), size=1)
        noise = noise.flatten()
        # print(noise.shape, self.pose[n].shape)
        self.pose[n] += noise

    def particle_update(self, l, m, t):
        x_im = np.arange(m.xmin, m.xmax + m.res, m.res)  # x-positions of each pixel of the map
        y_im = np.arange(m.ymin, m.ymax + m.res, m.res)  # y-positions of each pixel of the map

        x_range = np.arange(-0.2, 0.2 + 0.05, 0.05)
        y_range = np.arange(-0.2, 0.2 + 0.05, 0.05)

        # For positive x, y>0.5
        # binary_map = np.zeros((m.size, m.size), dtype=np.float16)

        binary_map = (np.copy(m.grid) > np.log(4)).astype(np.int)
        corr = []

        for n in range(self.num):
            scan = l.scan_to_world(self.pose[n], t)
            scan_copy = np.copy(scan)
            c = p2_utils.mapCorrelation(binary_map, x_im, y_im, scan_copy, x_range, y_range)
            max_c = c.max()
            corr.append(np.copy(max_c))
        corr = np.array(corr)
        # print("corr:", corr)
        corr = corr/2
        # p_obs = softmax(corr - np.max(corr))
        # self.weights = self.weights * p_obs / np.sum(self.weights * p_obs)

        log_w = np.log(self.weights) + corr
        log_w -= np.max(log_w) + logsumexp(log_w - np.max(log_w))
        self.weights = np.exp(log_w)

    def resample(self):
        N_eff = 1/np.sum(self.weights * self.weights)
        if N_eff > self.Nth:
            return


        nums = self.num
        # normalize weight and get cum sum
        sum_weights = np.cumsum(self.weights)
        sum_weights /= sum_weights[-1]

        random = (np.linspace(0, nums - 1, nums) + np.random.uniform(size=nums)) / nums
        new_sample = np.zeros(self.pose.shape)
        sample = 0
        idx = 0
        while sample < nums:
            while sum_weights[idx] < random[sample]:
                idx += 1
            new_sample[sample, :] = self.pose[idx, :]
            sample += 1
        self.pose = new_sample
        self.weights = np.ones(nums) / nums