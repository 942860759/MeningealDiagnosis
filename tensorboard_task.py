import os

def tensorboard_task():
    log_path = 'logs/'

    # os.system('tensorboard --logdir {}'.format('logs/'))
    os.system('C:\\ProgramData\\Anaconda3\\envs\\tensorflow\\Scripts\\tensorboard.exe --logdir {}'.format(log_path))


if __name__ == '__main__':
    tensorboard_task()