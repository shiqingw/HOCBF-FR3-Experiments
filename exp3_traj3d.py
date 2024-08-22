import numpy as np
import pickle
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt


if __name__ == '__main__':
    with open('exp3_marker_data.pickle', 'rb') as f:
        data = pickle.load(f)

    msg_timestamp_all = []
    center_timestamp_all = []
    markers_all = []
    center_all = []
    c2_all = []
    c1_all = []
    c0_all = []
    is_flying_all = []
    future_pos_all = []
    future_vel_all = []
    kf_center_all = []
    kf_vel_all = []

    future_times = np.array([0,0.05,0.1,0.15,0.2,0.25])
    num_prediction = len(future_times)
    for sample in data['dataset']:
        msg_timestamp_all.append(sample[0]) # must exist

        if not sample[1] is None:
            center_timestamp_all.append(sample[1])
        else:
            center_timestamp_all.append(0)
        
        markers_all.append(sample[2])

        if not sample[3] is None:
            center_all.append(sample[3])
        else:
            center_all.append([0, 0, 0])

        if not sample[4] is None:
            c2_all.append(sample[4])
        else:
            c2_all.append([0, 0, 0])
        
        if not sample[5] is None:
            c1_all.append(sample[5])
        else:
            c1_all.append([0, 0, 0])
        
        if not sample[6] is None:
            c0_all.append(sample[6])
        else:
            c0_all.append([0, 0, 0])
        
        is_flying_all.append(sample[7])

        if not sample[8] is None:
            future_pos_all.append(sample[8])
        else:
            future_pos_all.append(np.zeros((num_prediction,3)))
        
        if not sample[9] is None:
            future_vel_all.append(sample[9])
        else:
            future_vel_all.append(np.zeros((num_prediction,3)))

        if not sample[10] is None:
            kf_center_all.append(sample[10])
        else:
            kf_center_all.append([0, 0, 0])
        
        if not sample[11] is None:
            kf_vel_all.append(sample[11])
        else:
            kf_vel_all.append([0, 0, 0])

    msg_timestamp_all = np.array(msg_timestamp_all)
    msg_timestamp_all -= msg_timestamp_all[0]
    center_timestamp_all = np.array(center_timestamp_all)
    center_timestamp_all -= center_timestamp_all[0]
    center_all = np.array(center_all)
    c2_all = np.array(c2_all)
    c1_all = np.array(c1_all)
    c0_all = np.array(c0_all)
    is_flying_all = np.array(is_flying_all)
    future_pos_all = np.array(future_pos_all)
    future_vel_all = np.array(future_vel_all)
    kf_center_all = np.array(kf_center_all)
    kf_vel_all = np.array(kf_vel_all)

    idx_start = np.argmax(center_timestamp_all > 17.3)
    idx_end = np.argmax(center_timestamp_all > 17.9)
    
    predict_idx = 4
    time_ahead = future_times[predict_idx]
    future_pos = future_pos_all[idx_start:idx_end,predict_idx]
    idx_delta = int(time_ahead*240)
    center = center_all[idx_start+idx_delta:idx_end+idx_delta]
    print("time ahead: ", time_ahead)
    print("idx_delta: ", idx_delta)

    # plot 3d trajectory
    fig = plt.figure(0)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(center[:,0], center[:,1], center[:,2], label='ground truth')
    ax.plot(future_pos[:,0], future_pos[:,1], future_pos[:,2], label='prediction')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Set the axes to be equal
    max_range = np.array([center[:,0].max() - center[:,0].min(), 
                        center[:,1].max() - center[:,1].min(), 
                        center[:,2].max() - center[:,2].min()]).max() / 2.0

    mid_x = (center[:,0].max() + center[:,0].min()) * 0.5
    mid_y = (center[:,1].max() + center[:,1].min()) * 0.5
    mid_z = (center[:,2].max() + center[:,2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # plot 2d error
    fig = plt.figure(1)
    time = msg_timestamp_all[idx_start+idx_delta:idx_end+idx_delta]
    error = np.linalg.norm(future_pos - center, axis=1)
    plt.plot(time, error)
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('error')
    plt.show()
