


with open('nvgesture_train_correct_cvpr2016_v2.lst', 'r') as f:
    equal = 0
    count = 0
    for line in f:
        rgb_frame = line.split(' ')[2].split(':')[-2:]
        depth_frame = line.split(' ')[1].split(':')[-2:]
        if rgb_frame == depth_frame:
            equal += 1
        else:
            print(rgb_frame)
            print(depth_frame)
        count += 1
    print('{}/{} frames are aligned.'.format(equal, count))