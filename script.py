import os
lr, T = 1e-4, 2
source_length, target_length, comment_length = 512, 512, 64
batch_size, beam_size, epochs = 16, 4, 30
autonum = 20
lamda = 0.1

postfix = {"Java":"java", "C#":"cs", "C++":"cpp", "C":"c", "Python":"py", "PHP":"php", "Javascript":"js"}
cudanum = str(input("CUDA: "))
output_dir = "./save_models/"
os.system("mkdir -p " + output_dir)
os.system("CUDA_VISIBLE_DEVICES=" + cudanum + " python run.py" + \
        " --do_train" + \
        " --do_eval" + \
        " --model_type roberta" + \
        " --output_dir " + output_dir + \
        " --max_source_length " + str(source_length) + \
        " --max_target_length " + str(target_length) + \
        " --max_comment_length " + str(comment_length) + \
        " --beam_size " + str(beam_size) + \
        " --train_batch_size " + str(batch_size) + \
        " --eval_batch_size 69 " + \
        " --learning_rate " + str(lr) + \
        " --temperature " + str(T) + \
        " --autonum " + str(autonum) + \
        " --lamda " + str(lamda) + \
        " --load_model_path ./save_models/checkpoint-best/pytorch_model.bin" + \
        " --num_train_epochs %s 2>&1| tee %s/epoch-%s.log" % (epochs,output_dir,epochs)
)

