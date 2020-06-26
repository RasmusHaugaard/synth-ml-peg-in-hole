import argparse

import numpy as np
import rtde_control
import rtde_receive

from transform import Transform


def kabsch(P, Q):  # P, Q: [N, d]
    assert P.shape == Q.shape, '{}, {}'.format(P.shape, Q.shape)
    d = P.shape[1]
    Pc, Qc = P.mean(axis=0), Q.mean(axis=0)
    P, Q = P - Pc, Q - Qc
    H = P.T @ Q
    u, _, vt = np.linalg.svd(H, full_matrices=False)
    s = np.eye(d)
    s[-1, -1] = np.linalg.det(vt.T @ u.T)
    R = vt.T @ s @ u.T
    t = Qc - R @ Pc
    return R, t  # R: [d, d], t: [d]


def get_transform(rtde_r):
    pose = rtde_r.getActualTCPPose()
    return Transform(t=pose[:3], rotvec=pose[3:])


def get_stable_table_pose(rtde_c, rtde_r):
    #  TODO: press against table to obtain more stable pose
    return get_transform(rtde_r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ur-ip', type=str, required=True)
    parser.add_argument('--n', type=int, default=3)
    parser.add_argument('--grid-size', type=float, default=0.05)
    parser.add_argument('--tcp-tool-offset', type=float, default=0.012)
    args = parser.parse_args()

    ur_ip = args.ur_ip
    n = args.n
    grid_size = args.grid_size
    tcp_tool_offset = args.tcp_tool_offset

    rtde_c = rtde_control.RTDEControlInterface(ur_ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(ur_ip)

    rtde_c.teachMode()

    table_poses = []
    input('Insert the calibration tool close to the table origin with aligned tool axes')
    table_poses.append(get_stable_table_pose(rtde_c, rtde_r))
    table_offset_x = (int(input('How many holes to the table origin (x)?')) + 0.5) * grid_size
    table_offset_y = (int(input('How many holes to the table origin (y)?')) + 0.5) * grid_size
    for i in range(1, n):
        input('Insert the calibration tool at another position. ({}/{})'.format(i + 1, n))
        table_poses.append(get_stable_table_pose(rtde_c, rtde_r))
    pts_base = np.array([T @ (0, 0, tcp_tool_offset) for T in table_poses])

    # estimate table_T_base just from the first table pose
    table_T_base = table_poses[0] @ Transform(t=(-table_offset_x, -table_offset_y, tcp_tool_offset))
    base_T_table = table_T_base.inv()
    # find the acutal positions in the table-coordinates
    pts_table = np.array([base_T_table @ pose @ (0, 0, tcp_tool_offset) for pose in table_poses])
    # nearest possible table positions
    pts_table[:, 2] = 0
    pts_table[:, :2] = np.round(((pts_table[:, :2] - grid_size / 2) / grid_size)) * grid_size + grid_size / 2

    base_T_table = kabsch(pts_base, pts_table)
    print(base_T_table)


if __name__ == '__main__':
    main()
