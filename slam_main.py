# Venkatesh Prasad Venkataramanan
# PID : A53318036

import numpy as np
import matplotlib.pyplot as plt
import slam_helper as f
from scipy.special import expit


def main():
    size = 40
    res = 0.05
    num = 100
    step = 100
    data_index = 4

    lidar_data = '../data/lidar/train_lidar' + str(data_index)
    joint_data = '../data/joint/train_joint' + str(data_index)
    save_name = '../data/output/data' + str(data_index) + '/dead_reck' + str(data_index)

    l = f.LIDAR_SENSOR(lidar_data, joint_data)
    m = f.MAP(size, res)
    p = f.PARTICLE(num)
    dead_reck = l.dead_reckoning()
    np.save(save_name + '.npy', dead_reck)

    plt.figure()
    plt.plot(dead_reck[:, 0], dead_reck[:, 1])
    plt.savefig(save_name + '.png')
    plt.show(block=False)


    # fig = plt.figure(figsize=(18, 6))
    plt.figure()

    for t in range(0, len(l.lidar), step):
        print("Time: ", t)
        pose = p.best_particle()

        # print(pose.shape)
        scan = l.scan_to_world(pose, t)
        scan_w = np.copy(scan)
        # Cells convertion
        scan_w[0] = (np.ceil((scan[0] + size) / m.res)).astype(np.int16)
        scan_w[1] = (np.ceil((scan[1] + size) / m.res)).astype(np.int16)
        scan_w = scan_w.astype(np.int)

        pose_w = np.copy(pose)
        # print(pose)
        pose_w[0] = np.ceil((pose[0] + size) / m.res).astype(np.int16)
        pose_w[1] = np.ceil((pose[1] + size) / m.res).astype(np.int16)
        pose_w = pose_w.astype(np.int)

        m.update_map(scan_w, pose_w)

        p.resample()

        deltapose = dead_reck[t] - dead_reck[t - step]
        for i in range(num):
            p.particle_predict(i, deltapose)

        p.particle_update(l, m, t)

        if t % 1000 == 0:

            # plot original lidar points
            # pose_copy = np.copy(p.pose)
            traj_copy = list(p.traj)
            traj_copy = np.array(traj_copy)
            # pose_plt = p.particle_pose(pose_copy, res)
            # traj_plt = p.traj_plot(traj_copy, res)

            plt.autoscale(False)
            plot = 1 - expit(m.grid)
            # plot_R = rotate(plot, 90, reshape=False)
            # ax1 = fig.add_subplot(111)
            plt.imshow(plot, cmap='gray')
            plt.plot(pose_w[0], pose_w[1], '.r')
            # plt.scatter(pose_plt[:, 0], pose_plt[:, 1], s=30, c='g')
            # plt.scatter(traj_copy[:, 0]+800, traj_copy[:, 1]+800, s=1, c='b', marker=',')
            #plt.scatter(pose_copy[:, 0], pose_copy[:, 1], s=30, c='b')
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Simultaneous Localisation and Mapping")
            plt.axis('equal')
            save_dir = '../data/output/data' + str(data_index) + '/slam' + str(data_index) + 't' + str(t) + '.png'
            plt.savefig(save_dir)
            plt.show(block=False)

    save_file = '../data/output/data' + str(data_index) + '/slam' + str(data_index) + '.png'
    plot = 1-expit(m.grid)
    # print(m.grid.shape)
    plt.figure()
    plt.imshow(plot, cmap='gray')
    plt.savefig(save_file)
    plt.show()


if __name__ == '__main__':
    main()
