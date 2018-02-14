import pathlib
import os


def update_paths(args):

    # Create (if necessary) folder for summary file write and checkpoints
    p1 = os.path.realpath(os.getcwd())
    sfw = os.path.join(p1, args.sfw_dir)
    pathlib.Path(sfw).mkdir(parents=True, exist_ok=True)
    cp = os.path.join(p1, args.cp_dir)
    pathlib.Path(cp).mkdir(parents=True, exist_ok=True)

    # Update the txt files containing the paths to the data sets (train, val, test)
    train_p = args.train_f
    val_p = args.val_f
    test_p = args.test_f
    flag = 0
    while flag < 3:
        p1 = os.path.realpath(os.getcwd())
        p2 = os.path.join(p1, args.ds_dir)  # only 'train' folder has imgs
        if flag == 0:
            file = train_p
        elif flag == 1:
            file = val_p
        else:
            file = test_p
        with open(file, 'r+') as f:
            lines = f.readlines()
            wfile = open(file, 'w')
            for line in lines:
                items = line.split('\t')
                imgname = items[0].split('/')[-1]
                path = os.path.join(p2, imgname)
                wfile.write((path+'\t'))
                wfile.write(items[-1])
        wfile.close()
        flag += 1
