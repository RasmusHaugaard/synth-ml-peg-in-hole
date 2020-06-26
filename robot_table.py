#!/usr/bin/env python3

import time
from data_handling.transformable_frame import TransformableFrame
from cell.cell import Cell
from .plane import Plane
from finger_tips.finger_tip_change import release_finger
from finger_tips.finger_exchange import FingerExchange
import numpy as np
from transforms3d import axangles


# Small hack for Python 2 - 3 cross-compatibility
try:
    input = raw_input
except NameError:
    pass

def calc_orientation_A(robot):
    T_World_RobABase = robot.cell_t_entity()
    R_World_RobABase = T_World_RobABase.getRotation()
    R_RobABase_Tool = TransformableFrame([0, 0, 0, 3.14, 0, 0])
    R_World_Tool = R_World_RobABase * R_RobABase_Tool

    print(R_World_Tool)


def calc_orientation_B(robot):
    T_World_RobBBase = robot.cell_t_entity()
    R_World_RobBBase = T_World_RobBBase.getRotation()
    R_RobBBase_Tool = TransformableFrame([0, 0, 0, 2.222, 0, -2.246])
    R_World_Tool = R_World_RobBBase * R_RobBBase_Tool

    print(R_World_Tool)

    print(T_World_RobBBase * TransformableFrame([-0.53662, -0.04751, 0.33200, 2.047, 0.326, -2.061]))

    print(T_World_RobBBase * TransformableFrame([-0.49040, -0.28153, -0.17921, 2.205, 0, -2.188]))


def auto_calibration(robot):
    T_World_RobBase = robot.cell_t_entity()
    t_end_tool = TransformableFrame([0, 0, 0, 0, 0, 0])

    lin_speed = 0.75
    lin_speed_low = 0.25
    lin_acc = 1.2
    joint_speed = 1.05
    joint_acc = 1.4

    robot.setForceModeParameters(gain=0.25, damping=0.75)

    if robot.name == "robot_a":
        print("Robot A")
        name = "A"

        import applications.calibration.pointSetupA_UR10 as rob
        initialFrame_World = rob.initialFrame_World
        initialFrame_World2 = rob.initialFrame_World2
        print('initialFrame_World', initialFrame_World)
        print('initialFrame_World2', initialFrame_World2)
        calibration_points = rob.calibration_points
        test_points = rob.test_points
        # sidePoints in jointSpace
        sidePoints = rob.sidePoints

        moveUpHeight = rob.moveUpHeight
        heightAdjustment = rob.heightAdjustment
        smallHeight = rob.smallHeight
        smallSideDistance = rob.smallSideDistance

        xd_calib = rob.xd_calib
        xd_test = rob.xd_test
        xd_side = rob.xd_side

        t_end_tool = robot.gripper.t_robot_gripper()

        # return
    elif robot.name == "robot_b":
        print("Robot B")
        name = "B"

        # robot.teachModeStart()
        # raw_input("Wait for ENTER ")
        # robot.teachModeStop()
        # return

        print(robot.getActualTCPPose())

        #import applications.calibration.pointSetupB_UR10 as rob
        import applications.calibration.pointSetupB_UR10_Robotiq as rob
        initialFrame_World = rob.initialFrame_World
        initialFrame_World2 = rob.initialFrame_World2
        calibration_points = rob.calibration_points
        test_points = rob.test_points
        # sidePoints in jointSpace
        sidePoints = rob.sidePoints

        moveUpHeight = rob.moveUpHeight
        heightAdjustment = rob.heightAdjustment
        smallHeight = rob.smallHeight
        smallSideDistance = rob.smallSideDistance

        xd_calib = rob.xd_calib
        xd_test = rob.xd_test
        xd_side = rob.xd_side
        print(sidePoints)

        #t_end_tool = robot.screwdriver.t_robot_screwdriver()
        from old_carryover.parameters import T_RobotBTool_RobotiqClosed
        t_end_tool = T_RobotBTool_RobotiqClosed

    else:
        raise Exception("Unknwon robot!")

    print('T_World_RobBase', T_World_RobBase)
    print('T_World_RobBase.inv()', T_World_RobBase.inv())
    print('initialFrame_World', initialFrame_World)
    print('initialFrame_World2', initialFrame_World2)

    # Move to the initial frame
    # initialFrame_World = TransformableFrame([0.425, 0.075, 0.65, 2.200074231361496, 2.2416465300432464, -0.001851470435887073])
    # initialFrame_World = TransformableFrame([0.425, 0.075, 0.65, 2.2412776449824587, -2.1997123338894355, 0])
    print('Initial Frame:', (T_World_RobBase.inv() * initialFrame_World * t_end_tool.inv()).pose)
    robot.moveL((T_World_RobBase.inv() * initialFrame_World * t_end_tool.inv()).pose, lin_speed, lin_acc)
    #robot.moveL((T_World_RobBase.inv() * initialFrame_World2).pose, 0.25)

    print(robot.getActualTCPPose())

    # T_RobotATool_Weiss = TransformableFrame([0, 0, 0.197, 0, 0, 0])

    # wanted position of fingers: x = 12.5 cm, y = 52.5, z = 25 cm + 0 cm
    #        rotation of fingers: rx = 1.17 rad, ry = 2.895 rad, rz = 0 rad
    # calibration_points = [TransformableFrame([0.025, 0.525, 0, 1.17, 2.895, 0]),
    #                       TransformableFrame([0.125, 0.425, 0, 1.17, 2.895, 0]),
    #                       TransformableFrame([0.125, 0.325, 0, 1.17, 2.895, 0])]
    # calibration_points = [TransformableFrame([0.125, 0.625, 0, 2.2412776449824587, -2.1997123338894355, 0]),
    #                       TransformableFrame([0.025, 0.425, 0, 2.2412776449824587, -2.1997123338894355, 0]),
    #                       TransformableFrame([0.175, 0.075, 0, 2.2412776449824587, -2.1997123338894355, 0])]

    # T_World_Item = TransformableFrame([0.125, 0.525, 0.05, 1.17, 2.895, 0])

    recordedTCPPositions = []

    for T_World_Item in calibration_points:
        print("Point:", T_World_Item)

        # T_RobBase_Item = T_World_RobABase.inv() * T_World_Item * T_RobotATool_Weiss.inv()
        T_RobBase_Item = T_World_RobBase.inv() * T_World_Item * t_end_tool.inv()

        print(T_RobBase_Item)

        print("\nMove the robot")

        # robot_action = pool.submit(robot.moveCartesian, T_RobBase_Item, 0.25, 0, 0)
        # waitFor(robot_action)
        robot.moveL((T_RobBase_Item * heightAdjustment).pose, lin_speed, lin_acc)
        robot.moveL((T_RobBase_Item * smallHeight).pose, lin_speed, lin_acc)

        print("Move done")

        # SpeedUntilForce down into the ground
        print("Move down to ground")

        # xd = [0, 0, -0.01, 0, 0, 0]
        time.sleep(0.5)
        robot.zeroFtSensor()
        time.sleep(0.5)
        robot.speed_until_force(speed_vector=xd_calib, acceleration=1, stop_force=10)

        print("Force met, setting to teach mode")

        #robot.teachMode()
        robot.forceModeStart([0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [0, 0, -10, 0, 0, 0], 2, [0.01, 0.01, 0.5, 0.05, 0.05, 0.05])

        input("Move down in the hole. Press enter to save this pose")

        #robot.endTeachMode()
        robot.forceModeStop()

        currentT = robot.getActualTCPPose()
        print("Current TCP pose:", currentT)

        recordedTCPPositions.append(currentT)

        print("Moving up")
        # Moving up
        newT = TransformableFrame(currentT) * moveUpHeight
        robot.moveL(newT.pose, lin_speed_low, lin_acc)

    print("Move to initial")
    robot.moveL((T_World_RobBase.inv() * initialFrame_World * t_end_tool.inv()).pose, 0.25)

    for pose in recordedTCPPositions:
        print(pose)

    # print "Printing poses in reference to the base"
    #
    # for pose in recordedTCPPositions:
    #     print pose

    print("Printing poses in reference to the world")

    ps = []

    for pose in recordedTCPPositions:
        # ps.append(T_World_RobBase * TransformableFrame(pose) * T_RobotATool_Weiss)
        ps.append(T_World_RobBase * TransformableFrame(pose))

    for pose in ps:
        print(pose)

    plane = Plane(ps[0], ps[1], ps[2])
    n, point = plane.calculate_plane()

    print("n", n, "point", point)

    # Now the three inital points are done. Now move over to the other side of the table to check a point to see if it's
    # on the plane. If that is the case, accept the point, otherwise put it in teachmode and wait for the operator
    # to put it in the proper location.

    # test_points = [TransformableFrame([0.375, 0.625, 0, 2.2412776449824587, -2.1997123338894355, 0]),
    #                TransformableFrame([0.825, 0.225, 0, 2.2412776449824587, -2.1997123338894355, 0]),
    #                TransformableFrame([0.925, 0.125, 0, 2.2412776449824587, -2.1997123338894355, 0]),
    #                TransformableFrame([1.175, 0.275, 0, 2.2412776449824587, -2.1997123338894355, 0]),
    #                TransformableFrame([0.825, 0.575, 0, 2.2412776449824587, -2.1997123338894355, 0])]
    # T_World_RobBase * TransformableFrame([-0.4603, 0.83643, -0.435, 1.196, 1.23, -1.25])] # The point on the side of the table

    print("\n##########\nNow on to the test points\n##########\n")

    for i, T_World_Item in enumerate(test_points):
        if i > 0:
            initialFrame_World = initialFrame_World2
        # if i == 5:
        #     initialFrame_World = T_World_RobBase * TransformableFrame([-0.783, 0.830, -0.128, 1.078, 1.932, -0.699])

        print("Move back to initial")

        robot.moveL((T_World_RobBase.inv() * initialFrame_World * t_end_tool.inv()).pose, lin_speed, lin_acc)

        T_RobBase_Item = T_World_RobBase.inv() * T_World_Item * t_end_tool.inv()

        robot.moveL((T_RobBase_Item * heightAdjustment).pose, lin_speed, lin_acc)
        robot.moveL((T_RobBase_Item * smallHeight).pose, lin_speed, lin_acc)

        print("Move done")
        print("Move down to ground")

        # xd = [0, 0, -0.01, 0, 0, 0]
        time.sleep(0.5)
        robot.zeroFtSensor()
        time.sleep(0.5)
        robot.speed_until_force(speed_vector=xd_test, acceleration=1, stop_force=10)

        print("Force met. Checking if point is on the plane of the three first.")

        T_RobABase_RobotATool = robot.getActualTCPPose()
        T_World_RobotATool = T_World_RobBase * TransformableFrame(T_RobABase_RobotATool)  # * T_RobotATool_Weiss

        d = plane.distance_to_point(T_World_RobotATool)
        print(d)

        #if d > 0.005:
        # More than 5 mm away from the plane: Set into teach mode, then enter, then save. Otherwise just save
        # robot.teachMode()
        robot.forceModeStart([0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [0, 0, -10, 0, 0, 0], 2, [0.01, 0.01, 0.5, 0.05, 0.05, 0.05])
        input("Move down in the hole. Press enter to save this pose")
        # robot.endTeachMode()
        robot.forceModeStop()

        print("Point is on the plane")
        currentT = robot.getActualTCPPose()
        print("Current TCP pose:", currentT)
        recordedTCPPositions.append(currentT)

        print("Moving up")
        # Moving up
        newT = TransformableFrame(currentT) * moveUpHeight
        robot.moveL(newT.pose, lin_speed_low, lin_acc)

    print("Move to initial")
    robot.moveL((T_World_RobBase.inv() * initialFrame_World2 * t_end_tool.inv()).pose, 0.25)

    # The only point missing now is the point on the side.
    # The point on the side has to be moved to in joint space, as the IK
    # propably would do some crazy stuff
    # The following positions are given in a vector Q with joint values

    # sidePoints = [[3.11296925385709, -1.00373885282194, -2.06385184048329, -1.65893545402061, 1.52454510161705, 0.150621914447111],
    #               [1.59453280462202, -1.00373885282194, -2.22494573044237, -1.49958689331353, 1.52454510161705, 0.150621914447111],
    #               [1.49714343236074, -1.50168128841592, -2.43054551632730, -1.30061935858617, 3.61021355775027, 0.150621914447111],
    #               [1.86121911432675, -2.24170089126152, -2.44642801252045, -1.61826928244914, 3.44440727881081, -0.013962634015955]]

    print(sidePoints)

    if sidePoints:  # if sidePoints list is not empty
        print("Moving to side point")
        for qConfig in sidePoints:
            print("moving q", qConfig)
            robot.moveJ(qConfig, joint_speed, joint_acc)

        time.sleep(0.5)
        robot.zeroFtSensor()
        time.sleep(0.5)
        robot.speed_until_force(speed_vector=xd_side, acceleration=1, stop_force=10)

        print("Force met, setting to teach mode")

        robot.teachMode()

        input("Move down in the hole. Press enter to save this pose")

        robot.endTeachMode()

        currentT = robot.getActualTCPPose()
        print("Current TCP pose:", currentT)

        recordedTCPPositions.append(currentT)

        print("Move back to initial")
        pos = TransformableFrame(robot.getActualTCPPose())
        pos *= smallSideDistance  # TransformableFrame([0, 0, -0.05, 0, 0, 0])
        robot.moveL(pos.pose, lin_speed, lin_acc)  # move back a bit

        for qConfig in reversed(sidePoints):
            print("moving q", qConfig)
            robot.moveJ(qConfig, joint_speed, joint_acc)

        robot.moveL((T_World_RobBase.inv() * initialFrame_World * t_end_tool.inv()).pose, lin_speed, lin_acc)

    print("\nAll the recorded points:")
    print("T_RobBase_Tool, T_World_Item")

    for T_RobBase_Tool in recordedTCPPositions:
        # print T_RobABase_Tool, ",\t\t", (T_World_RobBase * TransformableFrame(T_RobABase_Tool) * T_RobotATool_Weiss).pose
        print(T_RobBase_Tool, ",\t\t", (T_World_RobBase * TransformableFrame(T_RobBase_Tool)).pose)

    print("Done")

    # Save poses to a csv file
    outFile = open("applications/calibration/robotTableCalibrationData" + name + ".csv", 'w+')
    outFile.write("x,y,z,Rx,Ry,Rz\n")

    for T in recordedTCPPositions:
        tmp = ['%.17f' % x for x in T]
        outFile.write(",".join(tmp) + '\n')

    outFile.close()


def findTransformation(pointSetOneP, pointSetTwoQ):
    # Calculates the rigid transform between two sets of points
    sizeOfPoints = pointSetOneP.shape

    pMean = np.transpose(np.array([0, 0, 0]))
    qMean = np.transpose(np.array([0, 0, 0]))

    for i in range(sizeOfPoints[1]):
        pMean = pMean + pointSetOneP[:, i]
        qMean = qMean + pointSetTwoQ[:, i]

    pMean /= sizeOfPoints[1]
    qMean /= sizeOfPoints[1]

    pMean = pMean.reshape(-1, 1)
    qMean = qMean.reshape(-1, 1)

    # print pMean
    # print qMean

    pCentered = pointSetOneP - pMean
    qCentered = pointSetTwoQ - qMean

    # print pCentered
    # print qCentered

    M = np.eye(sizeOfPoints[1])

    # print M

    # print pCentered.shape, M.shape, np.transpose(qCentered).shape

    S = np.linalg.multi_dot([pCentered, M, np.transpose(qCentered)])
    # print S

    U, Sresult, V = np.linalg.svd(S)
    V = np.transpose(V)
    # print V
    # print "U: \n", U
    # print "Sresult: \n", Sresult
    # print "V: \n", V

    # print U, Sresult, V
    # print len(U)

    E = np.eye(len(U))
    detUT = np.linalg.det(np.dot(U, np.transpose(V)))
    E[len(U) - 1, len(U) - 1] = detUT

    R = np.linalg.multi_dot([V, E, np.transpose(U)])
    t = qMean - np.dot(R, pMean)

    # print R
    # print t

    # print np.concatenate((R, t), axis=1), (np.concatenate((R, t), axis=1)).shape
    # print np.array([0, 0, 0, 1]).reshape(1,-1), (np.array([0, 0, 0, 1]).reshape(1,-1)).shape
    transform = np.concatenate((np.concatenate((R, t), axis=1), np.array([0, 0, 0, 1]).reshape(1, -1)))
    return transform


def compute_calibration_parameters(setup):
    calibration_data_robot = np.loadtxt(open("applications/calibration/robotTableCalibrationData" + setup + ".csv", "rb"),
                                        delimiter=",", skiprows=1)
    points_on_table_robot = np.loadtxt(open("applications/calibration/pointsOnTableRobot" + setup + "_Robotiq.csv", "rb"), delimiter=",",
                                       skiprows=0)

    posRobot = np.empty((3, calibration_data_robot.shape[0]))

    # print posRobot

    # Coordinate transformation of calibration tool
    # (here assumed to only be translation)
    # T_TCP_Tool = [0, 0, 0.012]
    if setup == "A":
        #rot_tool = TransformableFrame([0, 0, 0, 0.00310609, -0.00032931, 0.00941169])
        #rot_tool = TransformableFrame([0, 0, 0, 0.0033114369*2, -0.000386604156*2, 0.0093046936*2]) * TransformableFrame([0, 0, 0, -0.0075, 0, 0])
        rot_tool = TransformableFrame([0, 0, 0, -0.00245736, -0.0006192, 0.01119357])
        T_TCP_Tool = [0, 0, 0.197]  # For Robot A
        T_TCP_Tool = [0, 0, 0.215]  # For Robot A
        T_TCP_Tool = [0, 0, 0.1801]  # For Robot A
        T_TCP_Tool = rot_tool.inv() * TransformableFrame([0, 0, 0.1796, 0, 0, 0])
    elif setup == "B":
        #T_TCP_Tool = TransformableFrame([0.1505, -0.0515, 0.0285, 0, np.pi/2, 0]) # For Robot B Screwdriver
        #T_TCP_Tool = TransformableFrame([0, 0, 0.26282, 0, 0, 3.14])  # For Robot B Robotiq setup A
        T_TCP_Tool = TransformableFrame([-0.00022, 0.00063, 0.26128, 0, 0, 3.14])  # For Robot B Robotiq setup B


    #print(T_TCP_Tool)
    #print(rot_tool * T_TCP_Tool)
    #print("T_TCP_Tool Weiss: ", rot_tool.inv() * TransformableFrame([0, 0, 0.197, 0, 0, 0]))

    # start = height of data - 1
    # end = -1, because it stops when this is reached
    # step = -1 to count down
    print("for")
    for i in range(calibration_data_robot.shape[0] - 1, -1, -1):
        # print i, calibration_data_robot[i]
        # 0.315091558821 mm and a standard deviation of 0.0771040428194 mm
        posRobot[:, i] = (TransformableFrame(calibration_data_robot[i]) * T_TCP_Tool)[0:3]

    T_World_Robot = findTransformation(posRobot, np.transpose(points_on_table_robot[:, 0:3]))

    # print T_World_Robot

    # Verify transformation
    pWorld = np.transpose(
        np.linalg.multi_dot([
            np.eye(3, 4),
            T_World_Robot,
            np.concatenate((posRobot, np.ones((1, calibration_data_robot.shape[0]))), axis=0)
        ])
    )

    eEst = np.zeros(pWorld.shape[0])
    eRotEst = np.zeros(pWorld.shape[0])
    avgRot = np.zeros(3)

    for i in range(calibration_data_robot.shape[0] - 1, -1, -1):
        out = axangles.mat2axangle(T_World_Robot[0:3, 0:3])  # rotm2axang(T_World_Robot[0:3, 0:3])
        T_World_Robot_pose = np.append(T_World_Robot[0:3, 3], out[0] * out[1]).tolist()
        T_World_TCP_ref = TransformableFrame(points_on_table_robot[i].tolist())
        T_World_TCP_act = TransformableFrame(T_World_Robot_pose) * TransformableFrame(calibration_data_robot[i].tolist()) * T_TCP_Tool
        T_ref_act = T_World_TCP_ref.inv()*T_World_TCP_act
        eEst[i] = np.linalg.norm(T_ref_act[0:3])
        print(T_ref_act[0:3])
        eRotEst[i] = np.linalg.norm(T_ref_act[3:6])
        avgRot += T_ref_act[3:6]

    avgRot /= calibration_data_robot.shape[0]

    #for i in range(pWorld.shape[0] - 1, -1, -1):
    #    eEst[i] = np.linalg.norm(pWorld[i, :] - points_on_table_robot[i, 0:3])

    print("The calibration of the World to Robot has a mean error of", np.mean(eEst) * 1e3, \
          "mm and a standard deviation of", np.std(eEst) * 1e3, "mm")
    print("The calibration of the World to Robot has a rotational mean error of", np.mean(eRotEst), \
          "rad and a standard deviation of", np.std(eRotEst), "rad")
    print("Average rotation: ", avgRot)

    # Convert Transformation to axis angle coordinates
    out = axangles.mat2axangle(T_World_Robot[0:3, 0:3])  # rotm2axang(T_World_Robot[0:3, 0:3])
    # out = np.concatenate(out[0], out[1])
    out4 = np.append(out[0], out[1])

    print("\nT_World_Robot")
    print(T_World_Robot)

    print("\nT_World_Robot in axis angle coordinates")
    print(out4)

    print("\nT_World_Robot position")
    print(T_World_Robot[0:3, 3])

    print("")
    print("T_World_Robot pose")
    print(np.append(T_World_Robot[0:3, 3], out[0] * out[1]).tolist())

    return out


if __name__ == "__main__":
    cell = Cell.initialize()
    robot_a = cell.robot_a
    robot_b = cell.robot_b
    robot_a.move_home()
    #robot_b.move_to_config("bit_changer")
    robot_b.move_home()
    robot_to_calibrate = robot_b

    if robot_to_calibrate is robot_a:
        cur_finger = robot_a.gripper.current_finger()
        if cur_finger is not None:
            release_finger(robot_a, cur_finger)
            FingerExchange.release()
            robot_a.gripper.move_to_desired(25, 420, 0)
            input("Insert calibration plate. Press enter to lock and proceed after 2 seconds.")
            FingerExchange.lock()
            time.sleep(2)
            robot_a.move_home()
        else:
            choice = input("Is calibration tool inserted [y/n]? ").lower()
            if choice[:1] == 'y':
                pass
            elif choice[:1] == 'n':
                FingerExchange.release()
                robot_a.gripper.move_to_desired(25, 420, 0)
                input("Insert calibration plate. Press enter to lock and proceed after 2 seconds.")
                FingerExchange.lock()
                time.sleep(2)
            else:
                raise Exception("Not understood.")

        auto_calibration(robot_a)
        compute_calibration_parameters("A")

    if robot_to_calibrate is robot_b:
        auto_calibration(robot_b)
        compute_calibration_parameters("B")
