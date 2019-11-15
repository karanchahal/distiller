import os


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

alphas = [0.1, 0.3, 0.7, 0.9]
temps = [1, 5, 10, 15, 20]


num_epochs =100
student_model = 'resnet8'
teacher_model = 'resnet110'
teacher_path = '../no_kd_baseline/lightning_logs/' + \
                'default/version_1/checkpoints/_ckpt_epoch_25.ckpt'
save_dir = './hyperparam_tuning'
optim = 'adam'
learning_rate = 0.001
weight_decay = 1e-4
momentum = 0.9
mkdir(save_dir)
start_version = 0

for a in alphas:
    for t in temps:
        start_version += 1
        test_file_name = "{}_{}_test.s".format(a,t)
        n = open(test_file_name, "w")
        f = open("slurm_test.s","r") 
        n.write(f.read())
        command = ("python kd_baseline_trainer.py" + \
                  " --student-model {} --teacher-model {}" + \
                  " --path-to-teacher {} --save-dir {}" + \
                  " --version {} --optim {} --learning-rate {}" + \
                  " --weight-decay {} --momentum {} --alpha {}" + \
                  " --temperature {} --epochs {}").format(student_model,
                   teacher_model, teacher_path, save_dir, start_version,
                   optim, learning_rate, weight_decay, momentum, a, t, num_epochs)
         
        n.write('\n')
        n.write(command)
        n.close()
        os.system("sbatch " + test_file_name) 
