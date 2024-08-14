import numpy as np
import pickle
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

    msg_timestamp_all = np.array(msg_timestamp_all)
    msg_timestamp_all -= msg_timestamp_all[0]
    center_timestamp_all = np.array(center_timestamp_all)
    center_timestamp_all -= center_timestamp_all[0]
    center_all = np.array(center_all)
    c2_all = np.array(c2_all)
    c1_all = np.array(c1_all)
    c0_all = np.array(c0_all)
    is_flying_all = np.array(is_flying_all)
  
    plt.plot(msg_timestamp_all, center_all[:, 0], label='center x')
    plt.plot(msg_timestamp_all, center_all[:, 1], label='center y')
    plt.plot(msg_timestamp_all, center_all[:, 2], label='center z')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('Center of the sphere')
    plt.show()